# math_verified_rpo_dpo

## Method

TRL `DPOTrainer` fine-tunes Qwen2.5-1.5B Base with LoRA adapters on answer-verified Hendrycks MATH worked solutions. Each chosen completion is the public worked solution plus the exact required final-answer sentence. Each rejected completion keeps the same worked trace but deterministically corrupts only the final answer, creating a hard preference target for the evaluator-visible answer contract. `rpo_alpha=0.5` adds the RPO chosen-response NLL term so the run stays anchored to worked-solution imitation while optimizing pairwise answer verification.

The recipe merges LoRA adapters into a full Hugging Face causal-LM checkpoint in `$OUTPUT_DIR` before exit.

## Frameworks / Libraries

Declared in `pyproject.toml`: `trl>=0.17,<0.25`, `transformers>=4.51,<5`, `peft>=0.15,<0.20`, `accelerate>=1.6,<2`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260725`
- Data: up to `3000` examples from `the-jb/hendrycks-math` train split
- Sequence length: `1536`, max prompt length `512`
- Epochs: `1.0`
- Per-device batch size: `2`
- Gradient accumulation: `8`
- Effective batch size: `16`
- Learning rate: `8e-5`, cosine scheduler, warmup ratio `0.05`
- Weight decay: `0.01`
- Precision: `bfloat16`
- DPO beta: `0.08`
- RPO chosen-NLL weight: `0.5`
- LoRA: rank `32`, alpha `64`, dropout `0.05`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Data Provenance

Training uses only public training data downloaded by Hugging Face Datasets at runtime:

- `the-jb/hendrycks-math`, revision `af6b99a181a909b1aec1424451f10e875fd97377`, split `train`

This recipe does not train on MATH-500 evaluation examples. It uses the dataset `answer` field when present and otherwise falls back to extracting the final boxed answer from the training-set worked solution. Rejected answers are generated deterministically from the training answer by adding one to numeric/fractional answers or appending `+1` to symbolic answers.

## Prompt / Response Template

DPO prompt:

```text
Problem:
{problem}

Solution:
```

Chosen completion:

```text
{worked_solution}
Final Answer: The final answer is {answer}. I hope it is correct.
```

Rejected completion:

```text
{worked_solution}
Final Answer: The final answer is {corrupted_answer}. I hope it is correct.
```

No chat template is added. Evaluation should use the base-model completion format and the default MATH-500 four-shot Minerva prompt. The trained surface explicitly preserves the required final-answer sentence.

## Results

Submitted candidate: pending. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting.

## What Mattered / What Failed

- This candidate keeps the worked-solution prior from `math_worked_sft` while adding a pairwise preference signal directly on answer correctness.
- The rejected side is intentionally close to the chosen side, so DPO cannot win by preferring generic style alone; it must prefer the completion whose final sentence survives answer verification.
