# math_generated_blind_hybrid_sft

## Method

TRL `SFTTrainer` fine-tunes Qwen2.5-1.5B Base with LoRA adapters on a hybrid of Hendrycks MATH public worked solutions and runtime-generated self-rationales. The generated slice follows the `program.md` direction: for the same shuffled training problems, the recipe samples an answer-blind completion and an answer-conditioned completion from the base model. Blind completions are kept only when the extracted final answer matches the training answer; conditioned completions are downsampled to the same kept count before SFT. The adapters are merged into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR`.

The run writes `training_manifest.json` into `$OUTPUT_DIR` with raw/kept counts by visibility arm, MATH type counts, and stable problem/completion hashes for auditability.

## Frameworks / Libraries

Declared in `pyproject.toml`: `trl>=0.17,<0.25`, `transformers>=4.51,<5`, `peft>=0.15,<0.20`, `accelerate>=1.6,<2`, `datasets>=3.6,<5`, `torch>=2.5,<3`, `numpy<3`, and `sympy>=1.13`.

## Hyperparameters

- Seed: `20260724`
- Public data: up to `5600` examples from `the-jb/hendrycks-math` train split
- Generated data: `352` train problems sampled once per arm; blind arm verifier-kept by `src.math_utils.answers_match`; conditioned arm type-matched to the blind kept count
- Generation: `temperature=0.7`, `top_p=0.9`, `max_new_tokens=384`, generation batch size `8`
- Sequence length: `1536`, TRL packing disabled
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

This recipe does not train on MATH-500, HumanEval, GSM8K, or any evaluation-set examples. Public examples use the dataset `answer` field when present and otherwise extract the final boxed answer from the training-set worked solution. Generated examples are sampled at runtime from `$MODEL_PATH` on Hendrycks MATH train problems only; answer-blind generated traces are retained only when `extract_final_sentence_answer` / `answers_match` verifies the training-set answer.

## Prompt / Response Template

Public and retained generated examples are serialized for SFT as:

```text
Problem:
{problem}

Solution:
{reasoning}
Final Answer: The final answer is {answer}. I hope it is correct.
```

Answer-blind generation prompt:

```text
Problem:
{problem}

Solution:
Work step by step and end with exactly: "Final Answer: The final answer is <answer>. I hope it is correct."
```

Answer-conditioned generation prompt:

```text
Problem:
{problem}

Known final answer: {answer}

Solution:
Work step by step and end with exactly: "Final Answer: The final answer is <answer>. I hope it is correct."
```

No chat template is added. Evaluation should use the base-model completion format and the default MATH-500 four-shot Minerva prompt. The fine-tuning target explicitly preserves the required final-answer sentence.

## Results

Attempt 1 (`79427b4e-a33c-435f-a43a-f3cdb0ce7023`) completed with score `0.266`. Current candidate increases the public worked anchor to `5600`, samples `352` problems per generated arm, and type-matches answer-conditioned examples to the blind verifier-kept pool before SFT. Judge metric is MATH-500 `exact_match,none` under lm-eval `0.4.12` with four-shot Minerva prompting.

## What Mattered / What Failed

- This candidate directly tests the generated answer-blind verifier-filtered hybrid against a matched answer-conditioned generated slice while preserving the strong public worked-solution SFT anchor.
- The generated slice is deliberately small so generation plus SFT can fit inside the 40-minute recipe timeout; the output manifest makes the raw-versus-kept filtering behavior inspectable after judge runs.
