# math_blind_self_sft

## Method

This recipe tests verifier-filtered, answer-blind self-rationale SFT while keeping the final learner close to the strongest public worked-solution anchor.

1. Load Hendrycks MATH train problems and gold train labels.
2. Ask the base Qwen2.5-1.5B model to solve easier-level train prompts serialized only as `Problem:\n...\n\nSolution:\n`; no answer or public solution is present during generation.
3. Keep only sampled completions whose extracted final answer matches the train label under the shared verifier.
4. Canonicalize kept traces into the Minerva-visible final-answer sentence and mix them with public worked solutions from a disjoint shuffled anchor slice.
5. Train one LoRA SFT pass and merge the adapter into a complete Hugging Face causal-LM checkpoint in `$OUTPUT_DIR`.

The recipe intentionally uses the verifier only as a filter over answer-blind generations, not as evaluator-output tuning and not with MATH-500 examples.

## Hyperparameters

- Seed: `20260727`
- Data: `384` lower-level answer-blind generation prompts, `4` samples per prompt, up to `640` verified self-rationales, plus `3600` worked-solution anchors
- Generation: `max_new_tokens=448`, `temperature=0.85`, `top_p=0.95`, `repetition_penalty=1.05`
- SFT sequence length: `1792`, packing disabled
- SFT epochs: `0.85`
- Learning rate: `1.8e-4`, cosine scheduler, warmup ratio `0.03`, weight decay `0.01`
- Batch size: `4`, gradient accumulation: `8`
- Precision: `bfloat16`
- LoRA: rank `32`, alpha `64`, dropout `0.05`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Data Provenance

Training uses only public training data downloaded at runtime:

- `the-jb/hendrycks-math`, revision `af6b99a181a909b1aec1424451f10e875fd97377`, split `train`

Generated self-rationales are produced from train problems selected by public train difficulty level without showing the model the train answer or public solution. Gold train answers are used only after generation for filtering and for the final evaluator-visible answer sentence. The recipe does not use MATH-500 evaluation examples, HumanEval examples, GSM8K, or evaluator outputs.

## Prompt / Response Template

Answer-blind generation prompts are serialized as:

```text
Problem:
{problem}

Solution:
```

SFT examples are serialized as:

```text
Problem:
{problem}

Solution:
{reasoning}
Final Answer: The final answer is {answer}. I hope it is correct.
```

No chat template is added, so canonical evaluation defaults should be used.

## Expected Behavior

If the base model solves a subset of train problems unaided, the retained rationales should lie closer to its own inference manifold than polished public worked solutions. The worked anchor limits drift if the verified answer-blind pool is small or stylistically noisy.
