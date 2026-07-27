import argparse
import gc
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from fractions import Fraction

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.math_utils import (  # noqa: E402
    FINAL_TEMPLATE,
    answers_match,
    build_prompt,
    build_worked_text,
    extract_final_sentence_answer,
    extract_math_answer,
    normalize_answer,
)

SEED = 20260727
MATH_REVISION = "af6b99a181a909b1aec1424451f10e875fd97377"
NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
FINAL_START_RE = re.compile(r"Final Answer:\s*The final answer is", flags=re.IGNORECASE)


def offset_integer(match: re.Match[str]) -> str:
    raw_value = match.group(0).replace(",", "")
    value = Fraction(raw_value)
    if value == 0:
        return "1"
    shifted = value + 1 if value > 0 else value - 1
    if shifted.denominator == 1:
        return str(shifted.numerator)
    return str(float(shifted))


def corrupt_answer(answer: str) -> str:
    answer = normalize_answer(answer)
    if not answer:
        return "1"
    try:
        fraction = Fraction(answer.replace(",", ""))
        if fraction.denominator == 1:
            return str(fraction.numerator + 1)
        return str(fraction + 1)
    except (ValueError, ZeroDivisionError):
        pass
    if re.fullmatch(r"[-+]?\d+(,\d{3})*(\.\d+)?", answer):
        return re.sub(r"[-+]?\d+", offset_integer, answer.replace(",", ""), count=1)
    if answer.startswith("-"):
        return answer[1:]
    if answer.startswith(r"\frac"):
        return answer + "+1"
    return answer + "+1"


def drop_trailing_answer_marks(solution: str) -> str:
    text = solution.strip()
    text = re.sub(r"(?is)\\boxed\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\.?\s*$", "", text).strip()
    text = re.sub(r"(?is)(?:therefore|thus|so|hence)[^\n]{0,100}answer[^\n]*$", "", text).strip()
    return text or solution.strip()


def mutate_number(text: str, rng: random.Random) -> str | None:
    matches = [match for match in NUMERIC_RE.finditer(text) if len(match.group(0).strip("+-")) <= 8]
    if not matches:
        return None
    match = rng.choice(matches)
    replacement = offset_integer(match)
    if replacement == match.group(0):
        return None
    return f"{text[: match.start()]}{replacement}{text[match.end() :]}"


def mutate_operator(text: str, rng: random.Random) -> str | None:
    swaps = [(" + ", " - "), (" - ", " + "), (" > ", " < "), (" < ", " > "), ("=", r"\approx")]
    available = [(old, new) for old, new in swaps if old in text]
    if not available:
        return None
    old, new = rng.choice(available)
    index = text.find(old)
    return f"{text[:index]}{new}{text[index + len(old) :]}"


def deterministic_rejected(solution: str, wrong_answer: str, rng: random.Random) -> str:
    reasoning = drop_trailing_answer_marks(solution)
    candidates = [mutate_number(reasoning, rng), mutate_operator(reasoning, rng)]
    candidates = [candidate for candidate in candidates if candidate and candidate != reasoning]
    if candidates:
        reasoning = rng.choice(candidates)
    else:
        reasoning = reasoning.rstrip(".") + " with one case counted twice"
    if reasoning and not reasoning.endswith((".", "!", "?")):
        reasoning += "."
    return f"{reasoning}\n{FINAL_TEMPLATE.format(answer=normalize_answer(wrong_answer))}"


def build_completion(reasoning: str, answer: str) -> str:
    reasoning = reasoning.strip()
    if reasoning and not reasoning.endswith((".", "!", "?")):
        reasoning += "."
    return f"{reasoning}\n{FINAL_TEMPLATE.format(answer=normalize_answer(answer))}"


def load_math_rows(max_rows: int) -> list[dict[str, str]]:
    dataset = load_dataset("the-jb/hendrycks-math", split="train", revision=MATH_REVISION)
    rows: list[dict[str, str]] = []
    for row in dataset:
        answer = row.get("answer") or extract_math_answer(row["solution"])
        if not answer:
            continue
        rows.append(
            {
                "problem": row["problem"].strip(),
                "solution": row["solution"].strip(),
                "answer": normalize_answer(answer),
            }
        )
    random.Random(SEED).shuffle(rows)
    return rows[:max_rows]


def prepare_sft_dataset(rows: list[dict[str, str]]) -> Dataset:
    return Dataset.from_list(
        [
            {"text": build_worked_text(row["problem"], row["solution"], row["answer"])}
            for row in rows
        ]
    ).shuffle(seed=SEED)


def trim_generated_reasoning(completion: str) -> str:
    text = completion.strip()
    match = FINAL_START_RE.search(text)
    if match:
        text = text[: match.start()].strip()
    text = re.sub(r"(?is)\\boxed\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\.?\s*$", "", text).strip()
    text = re.sub(r"(?is)(?:therefore|thus|so|hence)[^\n]{0,100}answer[^\n]*$", "", text).strip()
    if not text:
        text = completion.strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def native_rejected_from_completion(completion: str, target_answer: str) -> str | None:
    sampled_answer = extract_final_sentence_answer(completion)
    if sampled_answer is None:
        sampled_answer = extract_math_answer(completion)
    if sampled_answer is None or answers_match(sampled_answer, target_answer):
        return None
    reasoning = trim_generated_reasoning(completion)
    if len(reasoning.split()) < 16:
        return None
    return f"{reasoning}\n{FINAL_TEMPLATE.format(answer=normalize_answer(sampled_answer))}"


def sample_native_rejections(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, str]],
    sample_examples: int,
    batch_size: int,
    max_new_tokens: int,
) -> dict[int, str]:
    model.eval()
    tokenizer.padding_side = "left"
    native: dict[int, str] = {}
    generation_config = {
        "do_sample": True,
        "temperature": 0.95,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    candidate_indices = list(range(min(sample_examples, len(rows))))
    with torch.inference_mode():
        for start in range(0, len(candidate_indices), batch_size):
            batch_indices = candidate_indices[start : start + batch_size]
            prompts = [build_prompt(rows[index]["problem"]) for index in batch_indices]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            ).to(model.device)
            outputs = model.generate(**inputs, **generation_config)
            prompt_length = inputs["input_ids"].shape[1]
            completions = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
            for index, completion in zip(batch_indices, completions, strict=True):
                rejected = native_rejected_from_completion(completion, rows[index]["answer"])
                if rejected is not None:
                    native[index] = rejected
    tokenizer.padding_side = "right"
    return native


def prepare_dpo_dataset(rows: list[dict[str, str]], native_rejections: dict[int, str], max_examples: int) -> Dataset:
    rng = random.Random(SEED + 1)
    pairs: list[dict[str, str]] = []
    native_count = 0
    for index, row in enumerate(rows):
        wrong_answer = corrupt_answer(row["answer"])
        if answers_match(wrong_answer, row["answer"]):
            continue
        rejected = native_rejections.get(index)
        if rejected is None:
            rejected = deterministic_rejected(row["solution"], wrong_answer, rng)
        else:
            native_count += 1
        rejected_answer = extract_final_sentence_answer(rejected)
        if answers_match(rejected_answer, row["answer"]):
            continue
        pairs.append(
            {
                "prompt": build_prompt(row["problem"]),
                "chosen": build_completion(row["solution"], row["answer"]),
                "rejected": rejected,
            }
        )
        if len(pairs) >= max_examples:
            break
    print(f"Prepared {len(pairs)} DPO pairs with {native_count} native sampled rejections", flush=True)
    return Dataset.from_list(pairs).shuffle(seed=SEED)


@dataclass
class TrainArgs:
    model_path: str
    output_dir: str
    train_pool: int = 2600
    sft_examples: int = 1200
    dpo_examples: int = 2000
    sample_examples: int = 700
    sft_max_length: int = 1536
    dpo_max_length: int = 1536
    dpo_max_prompt_length: int = 512
    sample_max_new_tokens: int = 192
    sft_epochs: float = 0.2
    dpo_epochs: float = 0.65
    sft_learning_rate: float = 2.0e-4
    dpo_learning_rate: float = 5.0e-5
    beta: float = 0.05
    rpo_alpha: float = 0.9
    batch_size: int = 2
    grad_accum: int = 8
    sample_batch_size: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_pool", type=int, default=2600)
    parser.add_argument("--sft_examples", type=int, default=1200)
    parser.add_argument("--dpo_examples", type=int, default=2000)
    parser.add_argument("--sample_examples", type=int, default=700)
    parser.add_argument("--sft_max_length", type=int, default=1536)
    parser.add_argument("--dpo_max_length", type=int, default=1536)
    parser.add_argument("--dpo_max_prompt_length", type=int, default=512)
    parser.add_argument("--sample_max_new_tokens", type=int, default=192)
    parser.add_argument("--sft_epochs", type=float, default=0.2)
    parser.add_argument("--dpo_epochs", type=float, default=0.65)
    parser.add_argument("--sft_learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--dpo_learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--rpo_alpha", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    return TrainArgs(**vars(parser.parse_args()))


def clean_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sft_tmp_dir = os.path.join(args.output_dir, "_sft_adapter_tmp")
    dpo_tmp_dir = os.path.join(args.output_dir, "_dpo_adapter_tmp")
    for path in (sft_tmp_dir, dpo_tmp_dir):
        if os.path.exists(path):
            shutil.rmtree(path)

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = load_math_rows(args.train_pool)
    sft_dataset = prepare_sft_dataset(rows[: args.sft_examples])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_args = SFTConfig(
        output_dir=sft_tmp_dir,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.sft_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_length=args.sft_max_length,
        packing=False,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        seed=SEED,
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    sft_trainer.train()

    sft_trainer.model.config.use_cache = True
    native_rejections = sample_native_rejections(
        sft_trainer.model,
        tokenizer,
        rows,
        sample_examples=args.sample_examples,
        batch_size=args.sample_batch_size,
        max_new_tokens=args.sample_max_new_tokens,
    )
    sft_trainer.model.config.use_cache = False
    dpo_dataset = prepare_dpo_dataset(rows, native_rejections, args.dpo_examples)

    dpo_args = DPOConfig(
        output_dir=dpo_tmp_dir,
        num_train_epochs=args.dpo_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.dpo_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )
    dpo_trainer = DPOTrainer(
        model=sft_trainer.model,
        args=dpo_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )
    dpo_trainer.train()

    merged_model = dpo_trainer.model.merge_and_unload()
    merged_model.config.use_cache = True
    if hasattr(merged_model.config, "torch_dtype"):
        merged_model.config.torch_dtype = "bfloat16"

    clean_output_dir(args.output_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    del dpo_trainer, sft_trainer, merged_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
