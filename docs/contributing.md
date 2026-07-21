# How to add a recipe

1. Create or improve a directory under `recipes/` (use a meaningful name, not a
   cycle number):
   ```
   recipes/grpo_exec_reward/
     run.sh
     README.md
   ```

2. `run.sh` must satisfy the [contract in the README](../README.md#runsh-contract).
   The same recipe directory may be submitted again after an improvement; each
   committed snapshot is evaluated independently.

3. `README.md` is the experiment record — write it in enough detail that someone
   with only this repo could re-run your recipe and land within eval noise of the
   score you report. Use **real numbers, not paraphrases** (`lr=5e-6`, not "a
   small lr"; `pass@1 0.20 → 0.28`, not "it improved"). Record it as you go, not
   from memory at the end. Include:

   - **Method**: the exact trainer (e.g. TRL `SFTTrainer` / `GRPOTrainer`, veRL,
     OpenRLHF), and any deviation from its defaults and why.
   - **Frameworks / libraries**: names and versions (as declared in
     `pyproject.toml`).
   - **Hyperparameters**: the real values — lr, batch size, epochs/steps, KL
     coefficient, LoRA rank, seed, etc. Enough to reproduce.
   - **Data provenance** (required — no training data ships with the repo):
     exactly where every training example came from — HF dataset id + revision +
     split, or the path to your synthesis/distillation script. Enough that the
     data can be rebuilt. **Confirm you did NOT train on HumanEval or the eval
     set.**
   - **Prompt / response template**: the exact string you trained on (verbatim),
     and how you aligned it to how the model is evaluated.
   - **Results**: baseline vs candidate `pass@1` (real numbers), the job_id, and
     the delta. Note eval-run variance if you saw it.
   - **What mattered / what failed**: the 1–2 things that moved the score, and
     failed runs recorded honestly (a negative result is still a result).

4. Put shared code in `src/` and import it from the recipe. Don't reinvent the
   wheel inside a recipe.

5. CI checks automatically:
   - `run.sh` exists and is syntactically valid (shellcheck)
   - `README.md` exists and is non-empty
   - modules in `src/` import cleanly
   - code passes ruff lint + format
