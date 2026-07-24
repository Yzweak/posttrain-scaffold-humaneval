# math_sft

## Method

TRL `SFTTrainer` fine-tunes Qwen2.5-1.5B Base with LoRA adapters on public math training data, then merges the adapters into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR`.

## Frameworks / Libraries

Declared in `pyproject.toml`: `trl>=0.17,<0.25`, `transformers>=4.51,<5`, `peft>=0.15,<0.20`, `accelerate>=1.6,<2`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260724`
- Data: `7473` examples from `the-jb/hendrycks-math` train split and `1500` examples from `openai/gsm8k` main train split
- Sequence length: `1536`, TRL packing disabled for SDPA attention safety
- Epochs: `1.0`
- Per-device batch size: `4`
- Gradient accumulation: `8`
- Effective batch size: `32`
- Learning rate: `2e-4`, cosine scheduler, warmup ratio `0.03`
- Weight decay: `0.01`
- Precision: `bfloat16`
- LoRA: rank `32`, alpha `64`, dropout `0.05`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Data Provenance

Training uses only public training splits downloaded by Hugging Face Datasets at runtime:

- `the-jb/hendrycks-math`, revision `af6b99a181a909b1aec1424451f10e875fd97377`, split `train`
- `openai/gsm8k`, revision `740312add88f781978c0658806c59bc2815b9866`, config `main`, split `train`

This recipe does not train on MATH-500 evaluation examples. It formats existing worked solutions from training splits and extracts the final answer from the MATH `answer` field, boxed MATH answers as fallback, or GSM8K `####` markers.

## Prompt / Response Template

Every training example is serialized exactly as:

```text
Problem:
{problem}

Solution:
{worked_solution}
Final Answer: The final answer is {answer}. I hope it is correct.
```

No chat template is added. Evaluation should use the base-model completion format and the default MATH-500 few-shot prompt. The fine-tuning target explicitly preserves the required final-answer sentence.

## Results

Submitted candidate: pending. Local smoke tests validate script syntax, TRL training, LoRA merge, and checkpoint loading before submission; judge MATH-500 exact-match numbers should be recorded after `/workspace/check.sh` returns.

## What Mattered / What Failed

- The recipe prioritizes exact answer-format conditioning over broad instruction tuning.
- LoRA is used to fit comfortably within the 40-minute recipe timeout while still adapting all attention and MLP projection layers.
