# math_answer_sft

## Method

TRL `SFTTrainer` fine-tunes Qwen2.5-1.5B Base with LoRA adapters on Hendrycks MATH training problems serialized with only the required final-answer sentence as the target. The adapters are merged into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR`.

This is the answer-format side of the worked-solutions-versus-answer-format diagnostic in `program.md`.

## Frameworks / Libraries

Declared in `pyproject.toml`: `trl>=0.17,<0.25`, `transformers>=4.51,<5`, `peft>=0.15,<0.20`, `accelerate>=1.6,<2`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260724`
- Data: up to `7500` examples from `the-jb/hendrycks-math` train split; no GSM8K examples
- Sequence length: `512`, TRL packing disabled
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
Final Answer: The final answer is {answer}. I hope it is correct.
```

No chat template is added. Evaluation should use the base-model completion format and the default MATH-500 four-shot prompt.

## Results

Submitted candidate: pending. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting.

## What Mattered / What Failed

- This candidate isolates exact final-answer-format conditioning from worked-solution imitation.
- The short `512` token context reduces recipe runtime because the target intentionally omits solution traces.
