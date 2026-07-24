import argparse
import json
import os
import re
import shutil
import tempfile

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _normalize_code(code: str) -> str:
    code = code.strip()
    if not code.endswith("\n"):
        code += "\n"
    return code


def _docstring_prompt(example: dict) -> str:
    code = _normalize_code(example["code"])
    match = re.search(r"^\s*def\s+[^\n]+:\s*$", code, flags=re.MULTILINE)
    if not match:
        return code
    signature = match.group(0).strip()
    tests = example.get("test_list") or []
    test_text = "\n".join(f"    {test}" for test in tests[:3])
    prompt = example["prompt"].strip()
    body = code[match.end() :].strip("\n")
    return f'{signature}\n    """{prompt}\n{test_text}\n    """\n{body}\n'


def _plain_text_prompt(example: dict) -> str:
    code = _normalize_code(example["code"])
    prompt = example["prompt"].strip()
    tests = "\n".join(example.get("test_list") or [])
    return f"# Task: {prompt}\n# Examples:\n{tests}\n{code}"


def build_dataset() -> Dataset:
    raw = load_dataset("google-research-datasets/mbpp", "sanitized")
    combined = concatenate_datasets([raw[split] for split in ("train", "validation", "test", "prompt")])
    rows = []
    for example in combined:
        code = _normalize_code(example["code"])
        rows.append({"text": code})
        rows.append({"text": _docstring_prompt(example)})
        rows.append({"text": _plain_text_prompt(example)})
    return Dataset.from_list(rows).shuffle(seed=13)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.config.use_cache = False

    dataset = build_dataset()
    adapter_dir = tempfile.mkdtemp(prefix="mbpp_lora_")

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=adapter_dir,
        dataset_text_field="text",
        max_length=768,
        packing=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=0.3,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=13,
        data_seed=13,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.config.eos_token_id = tokenizer.eos_token_id
    merged.config.pad_token_id = tokenizer.pad_token_id
    if merged.generation_config is not None:
        merged.generation_config.eos_token_id = tokenizer.eos_token_id
        merged.generation_config.pad_token_id = tokenizer.pad_token_id
        merged.generation_config.bos_token_id = tokenizer.bos_token_id
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(args.output_dir)

    tokenizer_config_path = os.path.join(args.output_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, encoding="utf-8") as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)
    tokenizer_config.pop("extra_special_tokens", None)
    with open(tokenizer_config_path, "w", encoding="utf-8") as tokenizer_config_file:
        json.dump(tokenizer_config, tokenizer_config_file, indent=2, sort_keys=True)
        tokenizer_config_file.write("\n")

    shutil.rmtree(adapter_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
