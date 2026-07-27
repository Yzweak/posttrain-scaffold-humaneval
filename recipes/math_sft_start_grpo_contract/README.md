# math_sft_start_grpo_contract

## Method

This recipe is the strict SFT-initialized GRPO continuation arm for the MATH-500 initialization bakeoff. It first builds a worked-solution prior from Hendrycks MATH train examples using the canonical plain `Problem:` / `Solution:` serialization, then applies a short TRL `GRPOTrainer` continuation on held-in train prompts with a verifiable answer reward centered on the evaluator-visible final sentence.

LoRA adapters are used during both stages and merged into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR` before `run.sh` exits. The tokenizer is copied from the base model without adding a chat template, so canonical Minerva completion evaluation should be used.

## Frameworks / Libraries

Declared in `pyproject.toml`: `trl>=0.17,<0.25`, `transformers>=4.51,<5`, `peft>=0.15,<0.20`, `accelerate>=1.6,<2`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260725`
- Data: up to `3400` worked-SFT examples and `512` GRPO prompts from `the-jb/hendrycks-math` train split
- SFT sequence length: `1536`, packing disabled
- SFT epochs: `0.7`
- SFT learning rate: `2e-4`, cosine scheduler, warmup ratio `0.03`, weight decay `0.01`
- GRPO steps: `28`, prompt length `768`, completion length `384`
- GRPO learning rate: `4e-6`, cosine scheduler, warmup ratio `0.05`, `beta=0.02`, `num_generations=4`, `temperature=0.85`, `top_p=0.95`, `repetition_penalty=1.05`, `loss_type=dr_grpo`
- Per-device train batch size: `4`; gradient accumulation: `8`
- Precision: `bfloat16`; attention implementation: `sdpa`
- LoRA: rank `32`, alpha `64`, dropout `0.05`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Data Provenance

Training uses only public training data downloaded by Hugging Face Datasets at runtime:

- `the-jb/hendrycks-math`, revision `af6b99a181a909b1aec1424451f10e875fd97377`, split `train`

This recipe does not train on MATH-500 evaluation examples. It uses the dataset `answer` field when present and otherwise falls back to the final boxed answer in the training-set worked solution.

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

The reward gives dominant credit only when the correct answer appears in the exact evaluator-visible final-answer sentence, partial credit for fallback-extractable correct answers, and small bounded shaping for single-suffix formatting and moderate worked-solution length.

## Results

Submitted candidate: pending. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting.

## What Mattered / What Failed

- This arm reduces the confound in `math_worked_grpo` by making the supervised start closer to the worked-SFT anchor while keeping GRPO short enough for the 40-minute recipe timeout.
- The reward continues to use `src.math_utils` as the source of truth for final-answer extraction and answer normalization, avoiding a parser bakeoff.
