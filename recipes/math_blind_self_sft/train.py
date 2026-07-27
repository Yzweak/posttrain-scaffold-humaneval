import argparse
import gc
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.math_utils import (  # noqa: E402
    answers_match,
    build_prompt,
    build_worked_text,
    extract_final_sentence_answer,
    extract_math_answer,
    normalize_answer,
)

SEED = 20260727
MATH_REVISION = "af6b99a181a909b1aec1424451f10e875fd97377"
FINAL_RE = re.compile(r"Final Answer:\s*The final answer is\s*.+?\.\s*I hope it is correct\.", flags=re.DOTALL)


@dataclass
class TrainArgs:
    model_path: str
    output_dir: str
    worked_examples: int = 3600
    generation_prompts: int = 384
    generations_per_prompt: int = 4
    self_max_new_tokens: int = 448
    max_self_examples: int = 640
    max_seq_length: int = 1792
    epochs: float = 0.85
    learning_rate: float = 1.8e-4
    batch_size: int = 4
    grad_accum: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--worked_examples", type=int, default=3600)
    parser.add_argument("--generation_prompts", type=int, default=384)
    parser.add_argument("--generations_per_prompt", type=int, default=4)
    parser.add_argument("--self_max_new_tokens", type=int, default=448)
    parser.add_argument("--max_self_examples", type=int, default=640)
    parser.add_argument("--max_seq_length", type=int, default=1792)
    parser.add_argument("--epochs", type=float, default=0.85)
    parser.add_argument("--learning_rate", type=float, default=1.8e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    return TrainArgs(**vars(parser.parse_args()))


def load_math_rows() -> list[dict[str, str]]:
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
                "level": str(row.get("level", "Level 9")),
            }
        )
    random.Random(SEED).shuffle(rows)
    return rows


def remove_answer_tail(completion: str) -> str:
    match = FINAL_RE.search(completion)
    if match:
        return completion[: match.start()].strip()
    marker = completion.rfind("Final Answer:")
    if marker >= 0:
        return completion[:marker].strip()
    return completion.strip()


def build_self_text(problem: str, completion: str, answer: str) -> str:
    reasoning = remove_answer_tail(completion)
    reasoning = re.split(r"\n\s*Problem:\s*\n", reasoning, maxsplit=1)[0].strip()
    reasoning = reasoning.strip(" \n")
    if not reasoning:
        reasoning = "We solve the problem step by step."
    return build_worked_text(problem, reasoning, answer)


@torch.inference_mode()
def level_key(row: dict[str, str]) -> int:
    match = re.search(r"\d+", row.get("level", ""))
    return int(match.group()) if match else 9


def generate_verified_self_examples(
    model, tokenizer, rows: list[dict[str, str]], args: TrainArgs
) -> list[dict[str, str]]:
    selected = sorted(rows, key=level_key)[: args.generation_prompts]
    prompts = [build_prompt(row["problem"]) for row in selected]
    self_examples: list[dict[str, str]] = []
    batch_size = 8

    model.eval()
    model.config.use_cache = True
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        row_batch = selected[start : start + batch_size]
        expanded_prompts = [prompt for prompt in prompt_batch for _ in range(args.generations_per_prompt)]
        inputs = tokenizer(
            expanded_prompts, return_tensors="pt", padding=True, truncation=True, max_length=768
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.self_max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for offset, completion in enumerate(decoded):
            row = row_batch[offset // args.generations_per_prompt]
            predicted = extract_final_sentence_answer(completion)
            if not answers_match(predicted, row["answer"]):
                continue
            text = build_self_text(row["problem"], completion, row["answer"])
            if len(text) < 80:
                continue
            self_examples.append({"text": text, "source": "answer_blind_self_verified"})
            if len(self_examples) >= args.max_self_examples:
                return self_examples
    return self_examples


def build_training_dataset(
    self_examples: list[dict[str, str]], rows: list[dict[str, str]], worked_examples: int
) -> Dataset:
    worked_rows = rows[-worked_examples:]
    anchor_examples = [
        {"text": build_worked_text(row["problem"], row["solution"], row["answer"]), "source": "worked_anchor"}
        for row in worked_rows
    ]
    combined = self_examples + anchor_examples
    random.Random(SEED + 1).shuffle(combined)
    dataset = Dataset.from_list(combined).shuffle(seed=SEED)
    return dataset.remove_columns([column for column in dataset.column_names if column != "text"])


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
    tmp_dir = os.path.join(args.output_dir, "_adapter_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rows = load_math_rows()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if torch.cuda.is_available():
        model.to("cuda")

    self_examples = generate_verified_self_examples(model, tokenizer, rows, args)
    print(f"kept {len(self_examples)} answer-blind self-rationales", flush=True)
    train_dataset = build_training_dataset(self_examples, rows, args.worked_examples)

    model.config.use_cache = False
    tokenizer.padding_side = "right"
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
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
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
