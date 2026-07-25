# math_interleaved_sft

## Method

TRL `SFTTrainer` fine-tunes Qwen2.5-1.5B Base with LoRA adapters on Hendrycks MATH training-set worked solutions, then merges the adapters into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR`.

This recipe is a conservative hybrid-trace variant of `math_worked_sft`: it keeps the natural-language MATH worked solution as the primary target, but appends a short `Computation check:` block before the final-answer sentence. The block either verifies simple exact arithmetic equations found in the original solution or reminds the model to track exact integer/fraction constraints for computational MATH categories. The goal is to strengthen exact-match arithmetic behavior without switching the base completion model into a code-only style.

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

This recipe does not train on MATH-500 evaluation examples. It uses the dataset `answer` field when present and otherwise falls back to extracting the final boxed answer from the training-set worked solution. The `Computation check:` lines are deterministic transformations of each training example's existing solution text and answer; no external solver labels or evaluation-set data are used.

## Prompt / Response Template

Every training example is serialized as:

```text
Problem:
{problem}

Solution:
{worked_solution}

Computation check:
- exact arithmetic: {verified_expression} = {value}.
- this supports the final exact answer {answer}.
Final Answer: The final answer is {answer}. I hope it is correct.
```

If no simple exact arithmetic equation is detected for a computational category (`Algebra`, `Counting & Probability`, `Number Theory`, `Prealgebra`, or `Precalculus`), the computation block is:

```text
Computation check:
- track integer constraints, signs, and exact fractions before simplifying.
- this supports the final exact answer {answer}.
```

For non-computational categories without detected arithmetic, the recipe falls back to the original `math_worked_sft` format without the computation block. No chat template is added. Evaluation should use the base-model completion format and the default MATH-500 four-shot Minerva prompt. The fine-tuning target explicitly preserves the required final-answer sentence.

## Results

Submitted candidate: pending. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting.

## What Mattered / What Failed

- This candidate tests the program lead that interleaved natural-language plus tool-grounded computation traces can improve exact final answers over pure worked-solution imitation.
- The computation block is deliberately compact to avoid contaminating the Minerva completion format with long Python or notebook-style outputs.
