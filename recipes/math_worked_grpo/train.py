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
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.math_utils import (  # noqa: E402
    answers_match,
    build_prompt,
    build_worked_text,
    extract_final_sentence_answer,
    extract_math_answer,
    normalize_answer,
)

SEED = 20260725
MATH_REVISION = "af6b99a181a909b1aec1424451f10e875fd97377"


def load_math_rows(max_examples: int) -> list[dict[str, str]]:
    dataset = load_dataset("the-jb/hendrycks-math", split="train", revision=MATH_REVISION)
    rows: list[dict[str, str]] = []
    for row in dataset:
        answer = row.get("answer") or extract_math_answer(row["solution"])
        if not answer:
            continue
        rows.append(
            {
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": normalize_answer(answer),
            }
        )
    random.Random(SEED).shuffle(rows)
    return rows[:max_examples]


def prepare_sft_dataset(rows: list[dict[str, str]]) -> Dataset:
    examples = [{"text": build_worked_text(row["problem"], row["solution"], row["answer"])} for row in rows]
    return Dataset.from_list(examples).shuffle(seed=SEED)


def prepare_grpo_dataset(rows: list[dict[str, str]], max_examples: int) -> Dataset:
    examples = [{"prompt": build_prompt(row["problem"]), "answer": row["answer"]} for row in rows[:max_examples]]
    return Dataset.from_list(examples).shuffle(seed=SEED + 1)


def extract_contract_answer(completion: str) -> str | None:
    match = re.search(
        r"Final Answer:\s*The final answer is\s*(.+?)\.\s*I hope it is correct\.",
        completion,
        flags=re.DOTALL,
    )
    if match:
        return normalize_answer(match.group(1))
    return None


def math_reward(completions: list[str], answer: list[str], **_kwargs: object) -> list[float]:
    rewards: list[float] = []
    for completion, target in zip(completions, answer, strict=True):
        contract_answer = extract_contract_answer(completion)
        extracted = contract_answer or extract_final_sentence_answer(completion)
        reward = 0.0
        if answers_match(contract_answer, target):
            reward += 0.88
        elif answers_match(extracted, target):
            reward += 0.45

        has_contract = contract_answer is not None
        if has_contract:
            reward += 0.08
        if completion.count("Final Answer:") == 1:
            reward += 0.04
        if has_contract and 24 <= len(completion.split()) <= 220:
            reward += 0.03
        rewards.append(min(reward, 1.0))
    return rewards


@dataclass
class TrainArgs:
    model_path: str
    output_dir: str
    sft_examples: int = 2500
    grpo_examples: int = 512
    sft_max_seq_length: int = 1536
    grpo_max_prompt_length: int = 768
    grpo_max_completion_length: int = 384
    sft_epochs: float = 0.6
    sft_learning_rate: float = 2.0e-4
    grpo_steps: int = 36
    grpo_learning_rate: float = 5.0e-6
    batch_size: int = 4
    grad_accum: int = 8
    grpo_generations: int = 4
    lora_rank: int = 32
    lora_alpha: int = 64


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sft_examples", type=int, default=2500)
    parser.add_argument("--grpo_examples", type=int, default=512)
    parser.add_argument("--sft_max_seq_length", type=int, default=1536)
    parser.add_argument("--grpo_max_prompt_length", type=int, default=768)
    parser.add_argument("--grpo_max_completion_length", type=int, default=384)
    parser.add_argument("--sft_epochs", type=float, default=0.6)
    parser.add_argument("--sft_learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--grpo_steps", type=int, default=36)
    parser.add_argument("--grpo_learning_rate", type=float, default=5.0e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--grpo_generations", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    return TrainArgs(**vars(parser.parse_args()))


def clear_output_dir(output_dir: str, keep: set[str] | None = None) -> None:
    keep = keep or set()
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if name in keep:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sft_tmp_dir = os.path.join(args.output_dir, "_sft_adapter_tmp")
    if os.path.exists(sft_tmp_dir):
        shutil.rmtree(sft_tmp_dir)

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = load_math_rows(max(args.sft_examples, args.grpo_examples))
    sft_dataset = prepare_sft_dataset(rows[: args.sft_examples])
    grpo_dataset = prepare_grpo_dataset(rows, args.grpo_examples)

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
        max_length=args.sft_max_seq_length,
        packing=False,
        bf16=True,
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

    grpo_tmp_dir = os.path.join(args.output_dir, "_grpo_tmp")
    if os.path.exists(grpo_tmp_dir):
        shutil.rmtree(grpo_tmp_dir)

    tokenizer.padding_side = "left"

    grpo_args = GRPOConfig(
        output_dir=grpo_tmp_dir,
        max_steps=args.grpo_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.grpo_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_prompt_length=args.grpo_max_prompt_length,
        max_completion_length=args.grpo_max_completion_length,
        num_generations=args.grpo_generations,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.05,
        beta=0.02,
        scale_rewards="group",
        loss_type="dr_grpo",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
    )
    grpo_trainer = GRPOTrainer(
        model=sft_trainer.model,
        reward_funcs=math_reward,
        args=grpo_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )
    grpo_trainer.train()

    merged_model = grpo_trainer.model.merge_and_unload()
    merged_model.config.use_cache = True
    if hasattr(merged_model.config, "torch_dtype"):
        merged_model.config.torch_dtype = "bfloat16"

    clear_output_dir(args.output_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    del grpo_trainer, sft_trainer, merged_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
