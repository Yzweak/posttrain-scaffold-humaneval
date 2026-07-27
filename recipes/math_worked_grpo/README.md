# math_worked_grpo

## Method

This recipe trains Qwen2.5-1.5B Base in two stages:

1. TRL `SFTTrainer` warm-start on Hendrycks MATH worked solutions, preserving the plain completion format used by the Minerva MATH-500 prompt.
2. TRL `GRPOTrainer` continuation on training-set MATH prompts with a verifiable reward for the required final-answer sentence and answer correctness.

LoRA adapters are used for both stages and merged into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR` before `run.sh` exits.

## Frameworks / Libraries

Declared in `pyproject.toml`: `trl>=0.17,<0.25`, `transformers>=4.51,<5`, `peft>=0.15,<0.20`, `accelerate>=1.6,<2`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260725`
- Data: up to `2500` worked-SFT examples and `512` GRPO prompts from `the-jb/hendrycks-math` train split
- SFT sequence length: `1536`, packing disabled
- SFT epochs: `0.6`
- SFT learning rate: `2e-4`, cosine scheduler, warmup ratio `0.03`, weight decay `0.01`
- GRPO steps: `36`, prompt length `768`, completion length `384`
- GRPO learning rate: `5e-6`, cosine scheduler, warmup ratio `0.05`, `beta=0.02`, `num_generations=4`, `temperature=0.9`, `top_p=0.95`, `repetition_penalty=1.05`, `loss_type=dr_grpo`
- GRPO reward: `+0.88` for a correct answer in the exact evaluator-visible final sentence, `+0.45` for a correct answer extractable only from a fallback pattern, plus small format/length shaping capped at `1.0`
- Per-device GRPO completion batch size: `4`; gradient accumulation: `8`; effective GRPO prompt batch size: `8`
- Precision: `bfloat16`
- LoRA: rank `32`, alpha `64`, dropout `0.05`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Data Provenance

Training uses only public training data downloaded by Hugging Face Datasets at runtime:

- `the-jb/hendrycks-math`, revision `af6b99a181a909b1aec1424451f10e875fd97377`, split `train`

This recipe does not train on MATH-500 evaluation examples. It uses the dataset `answer` field when present and otherwise falls back to the final boxed answer in the training-set worked solution. No HumanEval or MATH-500 examples are used.

## Prompt / Response Template

SFT examples are serialized exactly as:

```text
Problem:
{problem}

Solution:
{worked_solution}
Final Answer: The final answer is {answer}. I hope it is correct.
```

GRPO prompts are serialized exactly as:

```text
Problem:
{problem}

Solution:
```

The reward extracts answers from completions ending in:

```text
Final Answer: The final answer is <answer>. I hope it is correct.
```

Correct answers in the exact final-answer contract receive the dominant reward. A correct answer found only through a fallback extractor receives partial credit, so GRPO can still learn from mathematically useful generations that have not yet landed the evaluator-visible suffix. The exact final-answer contract, a single final-answer sentence, and a bounded worked-solution length receive small shaping rewards; total reward is capped at `1.0`. No chat template is added, so canonical evaluation defaults should be used.

## Results

Attempt 1 (`3f7ac228-39de-4026-ae25-a48c9432d876`): timed out at 2400.134 seconds with the original `5000` SFT examples, `0.85` SFT epochs, and `110` GRPO steps. Current candidate tightens the GRPO reward around the exact evaluator-visible final sentence while preserving fallback partial credit. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting. Record job id and score after `/workspace/check.sh` returns.

## What Mattered / What Failed

- The recipe keeps the strong in-domain worked-solution prior from `math_worked_sft` but adds on-policy pressure for landing on the verifier-visible final answer.
- The revised GRPO reward separates exact-contract correctness from fallback answer extraction, avoiding a full reward for answers that would be harder for the Minerva extractor to score.
