import argparse
import gc
import os
import random
import re
import shutil
from dataclasses import dataclass

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

FINAL_TEMPLATE = "Final Answer: The final answer is {answer}. I hope it is correct."
SEED = 20260724


def strip_boxed(text: str) -> str:
    text = text.strip()
    boxed = re.search(r"\\boxed\{(.+)\}\s*$", text)
    if not boxed:
        boxed = re.search(r"boxed\{(.+)\}\s*$", text)
    if boxed:
        return boxed.group(1).strip()
    return text


def normalize_answer(answer: str) -> str:
    answer = strip_boxed(answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.rstrip(".")
    return answer


def extract_math_answer(solution: str) -> str | None:
    boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if boxed_matches:
        return normalize_answer(boxed_matches[-1])
    markers = ["answer is", "Answer:", "answer:"]
    for marker in markers:
        index = solution.rfind(marker)
        if index >= 0:
            return normalize_answer(solution[index + len(marker) :].split("\n", 1)[0])
    return None


def extract_gsm8k_answer(answer: str) -> str | None:
    if "####" not in answer:
        return None
    final = answer.rsplit("####", 1)[1].strip()
    final = final.replace(",", "")
    return normalize_answer(final)


def build_text(problem: str, reasoning: str, answer: str) -> str:
    reasoning = reasoning.strip()
    answer = normalize_answer(answer)
    if reasoning and not reasoning.endswith((".", "!", "?")):
        reasoning += "."
    return f"Problem:\n{problem.strip()}\n\nSolution:\n{reasoning}\n{FINAL_TEMPLATE.format(answer=answer)}"


def load_competition_math(max_examples: int) -> Dataset:
    dataset = load_dataset("the-jb/hendrycks-math", split="train", revision="af6b99a181a909b1aec1424451f10e875fd97377")
    rows: list[dict[str, str]] = []
    for row in dataset:
        answer = row.get("answer") or extract_math_answer(row["solution"])
        if not answer:
            continue
        rows.append({"text": build_text(row["problem"], row["solution"], answer), "source": "competition_math_train"})
    random.Random(SEED).shuffle(rows)
    return Dataset.from_list(rows[:max_examples])


def load_gsm8k(max_examples: int) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main", split="train", revision="740312add88f781978c0658806c59bc2815b9866")
    rows: list[dict[str, str]] = []
    for row in dataset:
        answer = extract_gsm8k_answer(row["answer"])
        if not answer:
            continue
        reasoning = row["answer"].split("####", 1)[0].strip()
        rows.append({"text": build_text(row["question"], reasoning, answer), "source": "gsm8k_train"})
    random.Random(SEED + 1).shuffle(rows)
    return Dataset.from_list(rows[:max_examples])


def prepare_dataset(math_examples: int, gsm_examples: int) -> Dataset:
    datasets = [load_competition_math(math_examples), load_gsm8k(gsm_examples)]
    combined = concatenate_datasets(datasets)
    combined = combined.shuffle(seed=SEED)
    return combined.remove_columns([column for column in combined.column_names if column != "text"])


@dataclass
class TrainArgs:
    model_path: str
    output_dir: str
    math_examples: int = 7473
    gsm_examples: int = 1500
    max_seq_length: int = 1536
    epochs: float = 1.0
    learning_rate: float = 2.0e-4
    batch_size: int = 4
    grad_accum: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--math_examples", type=int, default=7473)
    parser.add_argument("--gsm_examples", type=int, default=1500)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    namespace = parser.parse_args()
    return TrainArgs(**vars(namespace))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_adapter_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = prepare_dataset(args.math_examples, args.gsm_examples)

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

    training_args = SFTConfig(
        output_dir=tmp_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_length=args.max_seq_length,
        packing=False,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        seed=SEED,
    )

    trainer = SFTTrainer(
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

    for name in os.listdir(args.output_dir):
        path = os.path.join(args.output_dir, name)
        if path == tmp_dir:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
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
