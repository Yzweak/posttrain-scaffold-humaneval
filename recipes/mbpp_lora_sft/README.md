# MBPP LoRA SFT

Supervised fine-tuning recipe for HumanEval-style Python function completion.

## Method

- Starts from the read-only Qwen2.5-1.5B base checkpoint in `$MODEL_PATH`.
- Loads the sanitized MBPP dataset from `google-research-datasets/mbpp`.
- Builds three training views per MBPP task: raw function code, a HumanEval-like
  signature/docstring/test prompt followed by the solution body, and a short
  comment/task format followed by the full function.
- Trains a LoRA adapter with TRL `SFTTrainer`, then merges it into the base model
  before writing `$OUTPUT_DIR`.

## Hyperparameters

- LoRA rank: 32, alpha: 64, dropout: 0.05
- Target modules: attention projections and MLP projections
- Epochs: 5 over the augmented MBPP corpus
- Effective batch size: 16 sequences
- Sequence length: 768 with packing
- Learning rate: 2e-4, cosine schedule, 5% warmup
- Precision: bfloat16

## Running

```bash
MODEL_PATH=/path/to/base OUTPUT_DIR=/tmp/mbpp_lora_sft bash recipes/mbpp_lora_sft/run.sh
```

The output directory is a merged Hugging Face causal LM checkpoint with tokenizer
files and no adapter dependency at evaluation time.
