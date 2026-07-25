import argparse
import gc
import os
import random
import re
import shutil
from dataclasses import dataclass
from fractions import Fraction

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

FINAL_TEMPLATE = "Final Answer: The final answer is {answer}. I hope it is correct."
SEED = 20260725
MATH_REVISION = "af6b99a181a909b1aec1424451f10e875fd97377"


def strip_boxed(text: str) -> str:
    text = text.strip()
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
                    tail = text[index + 1 :].strip()
                    if tail in {"", "."}:
                        return text[start + len(prefix) : index].strip()
                    break
            index += 1
    return text


def normalize_answer(answer: str) -> str:
    answer = strip_boxed(answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.rstrip(".")
    return answer


def extract_math_answer(solution: str) -> str | None:
    boxed = strip_boxed(solution)
    if boxed != solution.strip():
        return normalize_answer(boxed)
    markers = ["answer is", "Answer:", "answer:"]
    for marker in markers:
        index = solution.rfind(marker)
        if index >= 0:
            return normalize_answer(solution[index + len(marker) :].split("\n", 1)[0])
    return None


def offset_integer(match: re.Match[str]) -> str:
    value = int(match.group(0))
    if value == 0:
        return "1"
    return str(value + 1 if value > 0 else value - 1)


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


def build_completion(reasoning: str, answer: str) -> str:
    reasoning = reasoning.strip()
    if reasoning and not reasoning.endswith((".", "!", "?")):
        reasoning += "."
    return f"{reasoning}\n{FINAL_TEMPLATE.format(answer=normalize_answer(answer))}"


def build_pairs(max_examples: int) -> Dataset:
    dataset = load_dataset("the-jb/hendrycks-math", split="train", revision=MATH_REVISION)
    rows: list[dict[str, str]] = []
    for row in dataset:
        answer = row.get("answer") or extract_math_answer(row["solution"])
        if not answer:
            continue
        answer = normalize_answer(answer)
        rejected_answer = corrupt_answer(answer)
        if rejected_answer == answer:
            continue
        rows.append(
            {
                "prompt": f"Problem:\n{row['problem'].strip()}\n\nSolution:\n",
                "chosen": build_completion(row["solution"], answer),
                "rejected": build_completion(row["solution"], rejected_answer),
            }
        )
    random.Random(SEED).shuffle(rows)
    return Dataset.from_list(rows[:max_examples]).shuffle(seed=SEED)


@dataclass
class TrainArgs:
    model_path: str
    output_dir: str
    math_examples: int = 3000
    max_length: int = 1536
    max_prompt_length: int = 512
    epochs: float = 1.0
    learning_rate: float = 8.0e-5
    beta: float = 0.08
    rpo_alpha: float = 0.5
    batch_size: int = 2
    grad_accum: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--math_examples", type=int, default=3000)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=8.0e-5)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--rpo_alpha", type=float, default=0.5)
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
