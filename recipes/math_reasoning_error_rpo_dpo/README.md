# math_reasoning_error_rpo_dpo

## Method

This recipe trains Qwen2.5-1.5B Base with TRL `DPOTrainer` plus the RPO chosen-response term. It keeps the worked-solution SFT surface from Hendrycks MATH, but replaces the older final-answer-only rejected control with plausible process-error negatives: the rejected response mutates one intermediate number/operator in the public solution, removes the trailing boxed-answer cue when possible, adds a short bridge sentence, and ends with the required final-answer sentence containing an extracted wrong answer.

The objective is still pairwise preference training, not a new benchmark or prompt format. Chosen and rejected completions share the same plain base-model prompt:

```text
Problem:
{problem}

Solution:
```

Every trained completion ends with:

```text
Final Answer: The final answer is {answer}. I hope it is correct.
```

No chat template or recipe-local `eval.args` is used.

## Frameworks / Libraries

Dependencies are declared in `pyproject.toml`: `transformers>=4.51,<5`, `trl>=0.17,<0.25`, `peft>=0.15,<0.20`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `accelerate>=1.6,<2`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260726`
- Dataset examples: `3600`
- Trainer: TRL `DPOTrainer`
- Epochs: `1.0`
- Learning rate: `7e-5`
- Scheduler: cosine, `warmup_ratio=0.05`
- DPO beta: `0.06`
- RPO alpha: `0.8`
- LoRA: rank `32`, alpha `64`, dropout `0.05`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Batch size: `2` per device, gradient accumulation `8`
- Max length: `1536`; max prompt length: `512`
- Precision: `bf16`; attention implementation: `sdpa`

## Data Provenance

Training prompts and chosen worked solutions come only from Hugging Face dataset `the-jb/hendrycks-math`, split `train`, revision `af6b99a181a909b1aec1424451f10e875fd97377`. Rejected completions are generated deterministically by `recipes/math_reasoning_error_rpo_dpo/train.py` from those train-split solutions using seeded numeric/operator perturbations and wrong final answers filtered with `src.math_utils.answers_match`.

This recipe does not train on MATH-500, HumanEval, GSM8K, or any evaluation-set examples.

## Results

Attempt 1 (`100e5ae3-adc2-4275-85f6-7a4615693f4a`) failed before training because the recipe script imported `src.math_utils` without adding the repository root to Python import path under judge execution. Attempt 2 (`e57d14a6-aeed-4d71-95e5-145db953a64f`) completed with score `0.258`. Current candidate keeps the same hyperparameters but mutates the reasoning trace before replacing the final boxed cue, making the rejected side less likely to differ only in the answer-bearing tail. Judge metric is MATH-500 `exact_match,none` using lm-eval `0.4.12`, Minerva MATH-500, four-shot prompting, batch size `8`, and the canonical final-answer extractor.

## What Mattered / What Failed

The main change versus `math_verified_rpo_dpo` is that the rejected side contains a plausible invalid derivation and wrong final answer rather than the same correct derivation with only a corrupted final suffix. This should make the pairwise loss harder to solve with a suffix discriminator while the RPO term preserves the public worked-solution style.
