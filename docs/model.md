# Base model

## In one line

The base model is **Qwen2.5-1.5B-Instruct**, served read-only at `$MODEL_PATH`.
You post-train *from* it; you never modify it in place.

## Facts

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Size**: 1.5B parameters
- **Precision**: load in `bfloat16` for training and evaluation
- **Location**: `$MODEL_PATH` (read-only mount) — load with
  `AutoModelForCausalLM.from_pretrained(os.environ["MODEL_PATH"], torch_dtype=torch.bfloat16)`
- **Tokenizer**: ships with the model at the same path;
  `trust_remote_code=True` is used by the evaluator
- **Chat template**: it's an Instruct model, so it has a chat template — respect
  it when formatting training data and when generating.

## Working with it

- `$MODEL_PATH` is **read-only**. Write your trained checkpoint to `$OUTPUT_DIR`,
  never back to `$MODEL_PATH`.
- The final `$OUTPUT_DIR` must be a full, self-contained checkpoint loadable by
  `from_pretrained()`. If you train a LoRA adapter, **merge it into the base
  weights** inside run.sh before exiting.
- Keep dtype consistent (`bfloat16`) between training and the checkpoint you emit,
  since the judge evaluates in `bfloat16`.

## Fits on one A100-80GB

1.5B in bf16 is small; full fine-tuning, LoRA, or RL (GRPO/PPO) all fit
comfortably on a single A100-80GB. The active recipe timeout is published in
`/workspace/task/rules/resources.md`. Prefer mature libraries (TRL, veRL,
OpenRLHF) over hand-rolled training loops.
