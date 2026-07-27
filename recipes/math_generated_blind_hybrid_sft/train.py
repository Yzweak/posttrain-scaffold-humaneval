import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

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

MATH_REVISION = "af6b99a181a909b1aec1424451f10e875fd97377"
SEED = 20260724
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def solution_type(row: dict[str, str]) -> str:
    return str(row.get("type") or row.get("subject") or "unknown")


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
                "type": solution_type(row),
            }
        )
    random.Random(SEED).shuffle(rows)
    return rows


def public_worked_examples(rows: list[dict[str, str]], max_examples: int) -> list[dict[str, str]]:
    examples = []
    for row in rows[:max_examples]:
        examples.append(
            {
                "text": build_worked_text(row["problem"], row["solution"], row["answer"]),
                "source": "hendrycks_math_public_worked",
                "type": row["type"],
                "problem_hash": stable_hash(row["problem"]),
            }
        )
    return examples


def strip_to_reasoning(completion: str) -> str:
    marker = "Final Answer:"
    if marker in completion:
        completion = completion.split(marker, 1)[0]
    return completion.strip()


def batched(items: list[str], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    completions: list[str] = []
    for prompt_batch in batched(prompts, batch_size):
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(model.device)
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for sequence in generated:
            completion_ids = sequence[prompt_width:]
            completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    return completions


def build_generation_prompts(rows: list[dict[str, str]], visible_answer: bool) -> list[str]:
    suffix = (
        'Work step by step and end with exactly: "Final Answer: The final answer is <answer>. I hope it is correct."\n'
    )
    prompts = []
    for row in rows:
        if visible_answer:
            prompts.append(f"Problem:\n{row['problem']}\n\nKnown final answer: {row['answer']}\n\nSolution:\n{suffix}")
        else:
            prompts.append(f"{build_prompt(row['problem'])}{suffix}")
    return prompts


def generated_examples(
    model_path: str,
    rows: list[dict[str, str]],
    max_generate_examples: int,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    selected = rows[:max_generate_examples]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    all_examples: list[dict[str, str]] = []
    manifest: dict[str, object] = {
        "seed": SEED,
        "dataset": "the-jb/hendrycks-math",
        "revision": MATH_REVISION,
        "max_generate_examples": max_generate_examples,
        "arms": {},
    }
    generated_by_arm: dict[str, list[dict[str, str]]] = {}

    for arm, visible in (("blind", False), ("conditioned", True)):
        prompts = build_generation_prompts(selected, visible)
        completions = generate_completions(model, tokenizer, prompts, batch_size, max_new_tokens)
        kept: list[dict[str, str]] = []
        raw_by_type: dict[str, int] = {}
        kept_by_type: dict[str, int] = {}
        samples = []
        for row, completion in zip(selected, completions, strict=True):
            row_type = row["type"]
            raw_by_type[row_type] = raw_by_type.get(row_type, 0) + 1
            predicted = extract_final_sentence_answer(completion)
            keep = visible or answers_match(predicted, row["answer"])
            if not keep:
                continue
            reasoning = strip_to_reasoning(completion)
            if len(reasoning) < 40:
                continue
            text = build_worked_text(row["problem"], reasoning, row["answer"])
            example = {
                "text": text,
                "source": f"generated_{arm}",
                "type": row_type,
                "problem_hash": stable_hash(row["problem"]),
                "completion_hash": stable_hash(completion),
            }
            kept.append(example)
            kept_by_type[row_type] = kept_by_type.get(row_type, 0) + 1
            if len(samples) < 12:
                samples.append(
                    {
                        "problem_hash": example["problem_hash"],
                        "completion_hash": example["completion_hash"],
                        "type": row_type,
                        "predicted": predicted,
                        "target": row["answer"],
                    }
                )
        generated_by_arm[arm] = kept
        manifest["arms"][arm] = {
            "raw": len(completions),
            "kept": len(kept),
            "raw_by_type": raw_by_type,
            "kept_by_type": kept_by_type,
            "sample_hashes": samples,
        }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    blind = generated_by_arm["blind"]
    conditioned = generated_by_arm["conditioned"]
    blind_by_type: dict[str, int] = {}
    for example in blind:
        blind_by_type[example["type"]] = blind_by_type.get(example["type"], 0) + 1

    conditioned_by_type: dict[str, list[dict[str, str]]] = {}
    for example in conditioned:
        conditioned_by_type.setdefault(example["type"], []).append(example)
    rng = random.Random(SEED + 2)
    for examples in conditioned_by_type.values():
        rng.shuffle(examples)

    matched_conditioned = []
    for row_type, count in blind_by_type.items():
        matched_conditioned.extend(conditioned_by_type.get(row_type, [])[:count])
    if len(matched_conditioned) < len(blind):
        already = {example["completion_hash"] for example in matched_conditioned}
        remainder = [example for example in conditioned if example["completion_hash"] not in already]
        rng.shuffle(remainder)
        matched_conditioned.extend(remainder[: len(blind) - len(matched_conditioned)])

    all_examples.extend(blind)
    all_examples.extend(matched_conditioned)
    manifest["blind_kept_for_training"] = len(blind)
    manifest["matched_conditioned_kept_for_training"] = len(matched_conditioned)
    manifest["generated_training_examples"] = len(all_examples)
    return all_examples, manifest


def prepare_dataset(args: "TrainArgs") -> tuple[Dataset, dict[str, object]]:
    rows = load_math_rows()
    public_examples = public_worked_examples(rows, args.public_examples)
    synthetic_examples, manifest = generated_examples(
        args.model_path,
        rows,
        args.generate_examples,
        args.generation_batch_size,
        args.generation_max_new_tokens,
    )
    combined = public_examples + synthetic_examples
    random.Random(SEED + 3).shuffle(combined)
    manifest["public_training_examples"] = len(public_examples)
    manifest["total_training_examples"] = len(combined)
    sources = {row["source"] for row in combined}
    manifest["training_sources"] = {source: sum(1 for row in combined if row["source"] == source) for source in sources}
    dataset = Dataset.from_list(combined)
    return dataset.remove_columns([column for column in dataset.column_names if column != "text"]), manifest


@dataclass
class TrainArgs:
    model_path: str
    output_dir: str
    public_examples: int = 5600
    generate_examples: int = 352
    generation_batch_size: int = 8
    generation_max_new_tokens: int = 384
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
    parser.add_argument("--public_examples", type=int, default=5600)
    parser.add_argument("--generate_examples", type=int, default=352)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--generation_max_new_tokens", type=int, default=384)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--batch_size", type=int, default=4)
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
    tmp_dir = os.path.join(args.output_dir, "_adapter_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset, manifest = prepare_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        target_modules=TARGET_MODULES,
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

    clean_output_dir(args.output_dir, tmp_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_manifest.json"), "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)

    del trainer, merged_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
