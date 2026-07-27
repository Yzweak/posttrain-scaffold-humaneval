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
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.math_utils import (
    FINAL_TEMPLATE,
    answers_match,
    build_prompt,
    extract_math_answer,
    normalize_answer,
)

SEED = 20260726
MATH_REVISION = "af6b99a181a909b1aec1424451f10e875fd97377"
NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


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
    text = re.sub(r"(?is)(?:therefore|thus|so|hence)[^\n]{0,80}answer[^\n]*$", "", text).strip()
    return text or solution.strip()


def replace_last_boxed(text: str, answer: str) -> str:
    for prefix in (r"\boxed{", "boxed{"):
        start = text.rfind(prefix)
        if start < 0:
            continue
        index = start + len(prefix)
        depth = 1
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    replacement = f"{prefix}{normalize_answer(answer)}}}"
                    return f"{text[:start]}{replacement}{text[index + 1 :]}"
            index += 1
    return text


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


def mutate_reasoning(solution: str, wrong_answer: str, rng: random.Random) -> str:
    reasoning = drop_trailing_answer_marks(solution)
    candidates = [mutate_number(reasoning, rng), mutate_operator(reasoning, rng)]
    candidates = [candidate for candidate in candidates if candidate and candidate != reasoning]
    if candidates:
        reasoning = rng.choice(candidates)
    else:
        reasoning = reasoning.rstrip(".") + " with one case counted twice"
    reasoning = replace_last_boxed(reasoning, wrong_answer)
    bridge = rng.choice(
        [
            "This changes the carried value through the remaining algebra.",
            "Using that intermediate value in the last simplification gives the result below.",
            "Substituting this value into the final expression gives the following answer.",
        ]
    )
    if reasoning and not reasoning.endswith((".", "!", "?")):
        reasoning += "."
    return f"{reasoning}\n{bridge}\n{FINAL_TEMPLATE.format(answer=normalize_answer(wrong_answer))}"


def build_completion(reasoning: str, answer: str) -> str:
    reasoning = reasoning.strip()
    if reasoning and not reasoning.endswith((".", "!", "?")):
        reasoning += "."
    return f"{reasoning}\n{FINAL_TEMPLATE.format(answer=normalize_answer(answer))}"


def build_pairs(max_examples: int) -> Dataset:
    dataset = load_dataset("the-jb/hendrycks-math", split="train", revision=MATH_REVISION)
    rng = random.Random(SEED)
    rows: list[dict[str, str]] = []
    for row in dataset:
        answer = row.get("answer") or extract_math_answer(row["solution"])
        if not answer:
            continue
        answer = normalize_answer(answer)
        wrong_answer = corrupt_answer(answer)
        if answers_match(wrong_answer, answer):
            continue
        rejected = mutate_reasoning(row["solution"], wrong_answer, rng)
        rejected_answer = extract_math_answer(rejected)
        if answers_match(rejected_answer, answer):
            continue
        rows.append(
            {
                "prompt": build_prompt(row["problem"]),
                "chosen": build_completion(row["solution"], answer),
                "rejected": rejected,
            }
        )
    rng.shuffle(rows)
    return Dataset.from_list(rows[:max_examples]).shuffle(seed=SEED)


@dataclass
class TrainArgs:
    model_path: str
    output_dir: str
    math_examples: int = 3600
    max_length: int = 1536
    max_prompt_length: int = 512
    epochs: float = 1.0
    learning_rate: float = 7.0e-5
    beta: float = 0.06
    rpo_alpha: float = 0.8
    batch_size: int = 2
    grad_accum: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--math_examples", type=int, default=3600)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=7.0e-5)
    parser.add_argument("--beta", type=float, default=0.06)
    parser.add_argument("--rpo_alpha", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    namespace = parser.parse_args()
    return TrainArgs(**vars(namespace))


def clean_output_dir(output_dir: str, keep: str) -> None:
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if path == keep:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_dpo_adapter_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = build_pairs(args.math_examples)

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

    training_args = DPOConfig(
        output_dir=tmp_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.config.use_cache = True
    if hasattr(merged_model.config, "torch_dtype"):
        merged_model.config.torch_dtype = "bfloat16"

    clean_output_dir(args.output_dir, tmp_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    del trainer, merged_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
