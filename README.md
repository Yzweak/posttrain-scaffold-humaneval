# posttrainrepo

Post-training experiment repository for HumanEval. Your goal: modify this repo
so that `bash recipes/<your-recipe>/run.sh` trains a model that scores higher
on the code-generation benchmark.

## Layout

```
src/          shared modules (data loaders, reward functions, merge utils, ...)
docs/         repo docs (architecture, evaluation, base model)
cookbook/     runnable snippets and gotchas
recipes/      one self-contained recipe per experiment; recipes may be iterated
              in place and each submitted commit is evaluated independently
  <name>/     `<name>` is the recipe directory and submit argument
    run.sh    one-click script: build env → train → produce checkpoint
    README.md method, hyperparameters, results, score
    eval.args optional recipe-local evaluation controls
```

## How to use it

1. Read `program.md` for the research direction to pursue
2. Read `docs/` for the evaluation mechanism and base model
3. Browse `recipes/` for prior experiments (if any)
4. Create or improve a recipe directory under `recipes/`
5. Write or update `run.sh` (must satisfy the contract below) + `README.md`
6. Add or reuse shared modules in `src/` as needed

Commit each candidate before submitting it. A later submission may improve the
same recipe directory; Git commits preserve the earlier candidates.

## run.sh contract

```bash
MODEL_PATH=<base weights> OUTPUT_DIR=<output dir> bash recipes/<name>/run.sh
```

- **Self-contained**: start from `uv sync`, end with a checkpoint in `$OUTPUT_DIR`
- **Idempotent**: runs on a fresh clone with no leftover local state
- **Exit code**: 0 on success, non-zero on failure
- **Output**: `$OUTPUT_DIR/` must load directly with
  `AutoModelForCausalLM.from_pretrained()`
- **Time**: finish within the recipe timeout in
  `/workspace/task/rules/resources.md`

## Principles

- **Use libraries, don't reinvent**: prefer TRL / veRL / OpenRLHF over
  hand-written loops; keep code minimal
- **src/ is shared**: put modules used by multiple recipes there; keep
  experiment-specific logic inside the recipe
- **Recipes are independent**: each recipe runs on its own, with no cross-deps
