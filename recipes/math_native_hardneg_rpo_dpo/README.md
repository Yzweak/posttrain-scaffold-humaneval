# math_native_hardneg_rpo_dpo

## Method

This recipe combines the strongest worked-solution SFT surface with model-native hard-negative preference training. It first LoRA-SFTs Qwen2.5-1.5B Base on Hendrycks MATH train worked solutions, samples completions from that warmed-up policy on held-in train prompts, keeps sampled completions whose extracted final answer is wrong, canonicalizes their ending to the evaluator-visible final-answer sentence, and then continues with TRL `DPOTrainer` using the RPO chosen-response term.

The recipe stays in the assigned model-native error-negative direction: chosen responses are public train-split worked solutions, while rejected responses are primarily wrong traces produced by the model family itself. Deterministic numeric/operator corruptions from `math_reasoning_error_rpo_dpo` are used only as coverage fallback when sampling does not yield a wrong parseable completion for a train row.

All prompts use the plain base-model completion format:

```text
Problem:
{problem}

Solution:
```

Every chosen and rejected completion ends with:

```text
Final Answer: The final answer is {answer}. I hope it is correct.
```

No chat template or recipe-local `eval.args` is used.

## Frameworks / Libraries

Dependencies are declared in `pyproject.toml`: `transformers>=4.51,<5`, `trl>=0.17,<0.25`, `peft>=0.15,<0.20`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `accelerate>=1.6,<2`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260727`
- Data pool: `2600` examples from Hendrycks MATH train
- SFT warm-up: `1200` examples, `0.2` epochs, learning rate `2e-4`
- Native sampling: up to `700` train prompts, temperature `0.95`, top-p `0.9`, max new tokens `192`
- DPO/RPO pairs: up to `2000` examples
- DPO epochs: `0.65`
- DPO learning rate: `5e-5`, cosine scheduler, warmup ratio `0.05`
- DPO beta: `0.05`; RPO alpha: `0.9`
- Sequence lengths: SFT max length `1536`; DPO max length `1536`, max prompt length `512`
- Batch size: `2` per device, gradient accumulation `8`; sample batch size `8`
- LoRA: rank `32`, alpha `64`, dropout `0.05`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Precision: `bf16`; attention implementation: `sdpa`

## Data Provenance

Training uses only Hugging Face dataset `the-jb/hendrycks-math`, split `train`, revision `af6b99a181a909b1aec1424451f10e875fd97377`. Sampled rejected completions are generated at recipe runtime only for those train-split prompts. The recipe does not train on MATH-500, GSM8K, HumanEval, evaluation outputs, or any benchmark-specific parser tuning.

## Results

Attempt 1 timed out during training at the 40-minute recipe limit. Current candidate reduces SFT, sampling, and DPO volume to fit the recipe timeout. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting, batch size `8`, and the canonical final-answer extractor.

## What Mattered / What Failed

- The SFT warm-up makes sampled negatives match the intended worked-solution distribution more closely than raw base-model text.
- RPO preserves the public correct trace while DPO compares it against naturally generated wrong traces rather than suffix-only answer swaps.
- The deterministic fallback prevents low native-parse coverage from collapsing the DPO dataset, but the script reports how many native sampled rejections were actually used.
