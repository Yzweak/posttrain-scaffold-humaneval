# math_worked_sft

## Method

TRL `SFTTrainer` fine-tunes Qwen2.5-1.5B Base with LoRA adapters on Hendrycks MATH training-set worked solutions only, then merges the adapters into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR`.

This recipe tests the program lead that MATH-500 benefits most from in-domain competition-math solution traces rather than answer-format shortcuts or GSM8K mixture data.

## Frameworks / Libraries

Declared in `pyproject.toml`: `trl>=0.17,<0.25`, `transformers>=4.51,<5`, `peft>=0.15,<0.20`, `accelerate>=1.6,<2`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260724`
- Data: up to `7500` examples from `the-jb/hendrycks-math` train split; no GSM8K examples
- Sequence length: `2048`, TRL packing disabled
- Epochs: `1.0`
- Per-device batch size: `4`
- Gradient accumulation: `8`
- Effective batch size: `32`
- Learning rate: `2e-4`, cosine scheduler, warmup ratio `0.03`
- Weight decay: `0.01`
- Precision: `bfloat16`
- LoRA: rank `32`, alpha `64`, dropout `0.05`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Data Provenance

Training uses only public training data downloaded by Hugging Face Datasets at runtime:

- `the-jb/hendrycks-math`, revision `af6b99a181a909b1aec1424451f10e875fd97377`, split `train`

This recipe does not train on MATH-500 evaluation examples. It uses the dataset `answer` field when present and otherwise falls back to extracting the final boxed answer from the training-set worked solution.

## Prompt / Response Template

Every training example is serialized exactly as:

```text
Problem:
{problem}

Solution:
{worked_solution}
Final Answer: The final answer is {answer}. I hope it is correct.
```

No chat template is added. Evaluation should use the base-model completion format and the default MATH-500 four-shot prompt. The fine-tuning target explicitly preserves the required final-answer sentence.

## Results

Submitted candidate: pending. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting.

## What Mattered / What Failed

- This candidate removes the `1500` GSM8K examples from `math_sft` to keep all gradient updates on competition-style MATH worked solutions.
- The longer `2048` token context is intended to preserve more complete training solutions before the required final-answer sentence.
