# Recipes

Each recipe is a task with the settings we measured on NexCloud, the command that runs it, and the results. The
scripts use the `nextron-cloud` SDK only, so they are short enough to copy into your own code.

NexCloud has four ways to train:

- **Managed SFT**: you send a dataset and settings; NexCloud runs the whole supervised fine-tuning job and keeps the
  checkpoint. Billed per token.
- **DPO** (preference training): you send prompts, each with a reply you prefer and one you reject; NexCloud trains the
  model to make the preferred one more likely and keeps the checkpoint. Billed per token.
- **RFT** (reinforcement fine-tuning): you send prompts with their answers; NexCloud samples replies, scores them with a
  built-in reward (`boxed` or `exact`) and trains with GRPO, round after round, and keeps the checkpoint. Billed per token.
- **Serverless RL**: an RL run is a model loaded on NexCloud's GPUs; your own loop samples, scores in your own code and
  trains, one step at a time. Billed by the time the run is ready.

After a job (Managed SFT, DPO or RFT), an **inference run** serves the trained model on the same GPUs (`nextron-cloud
inference start --job job_...`), so you can try it before downloading it, on the models that list `inference` in
`nextron-cloud models`. The model of a Serverless RL run is downloaded instead: an inference run takes a job's id.

## Getting started

- [`sft_basic.py`](sft_basic.py): the smallest managed fine-tune: a sample dataset already on the service, 20 steps,
  validation loss (Gemma-4-26B: 0.6448 -> 0.535, 5,500 tokens/s/GPU, $3.52).
- [`rl_basic.py`](rl_basic.py): the smallest Serverless RL loop: sample a batch, score it, train, five times.
- [`tinker_loop.py`](tinker_loop.py): the same kind of loop written for the `tinker` SDK, which an RL run also serves.

## Recipes

| Recipe | Mode | Task | Result |
|---|---|---|---|
| [`chat_sft`](chat_sft/) | managed SFT | NoRobots: 9,500 human-written conversations | Qwen3.8-27B, Gemma-4-26B, Muse-Glimmer-30B: $1.43-$8.47 a run |
| [`chat_dpo`](chat_dpo/) | DPO | UltraFeedback: 4,000 preference pairs | Gemma-4-26B: loss 0.693 -> 0.63 in 14 steps, $1.73; Qwen3.8-27B does not learn from other models' replies |
| [`math_rft`](math_rft/) | RFT | MATH with the built-in `boxed` reward | Qwen3.8-27B: reward 0.75 -> 0.87 in 10 rounds, replies half as long |
| [`math_rl`](math_rl/) | Serverless RL | MATH with your own grader | MATH-500 35.0% -> 78.0% in 10 steps (Qwen3.8-27B) |
| [`guess_number`](guess_number/) | Serverless RL | a multi-turn game: guess a hidden number from "too low" / "too high" | Qwen3.8-27B: 75.8% of evaluation games solved before, training reward 0.99 from step 2 |
| [`code_rl`](code_rl/) | Serverless RL | competitive programming: your code runs each solution's tests in a sandbox, a failed solution gets one more attempt | 687 Codeforces and LiveCodeBench problems, all tests pass: 44.3% -> 54.1% in 6 steps (Qwen3.8-27B) |
| [`code`](../docs/recipes.md#code-multi-turn-with-a-tool) | Serverless RL, `tinker` SDK | competitive programming: two turns with a tool, the generated code runs in a sandbox | 687 Codeforces and LiveCodeBench problems, all tests pass: 20.7% -> 37.8% in 10 steps (Qwen3.8-27B) |

The tinker-cookbook recipes (`math`, `code`, `gsm8k`, `sl`) run unchanged on an RL run: `nextron-cookbook run <name>`,
see [docs/recipes.md](../docs/recipes.md).

## What the settings are tuned for

- **Managed SFT** packs a step by tokens, not samples. `batch_size` sets the samples a step; the recipes size it so
  each GPU packs about 16K tokens, which gives enough steps on a small dataset at full speed (2,700 tokens/s/GPU on
  Qwen3.8-27B). Sweep the learning rate first and watch `val_loss`: on NoRobots it was lowest after about one epoch.
- **Serverless RL** trains a step in packed rows of a fixed length per GPU (`train` returns `row_tokens` and
  `row_fill`). A row costs about the same however full it is, so the RL recipes size each step to fill the rows
  (`--rows`), aiming at 90%.
- Sampling is one request per batch (`live.sample(prompts, n=...)`); only your rewards travel back.
- Gradient clipping is on by default; `nextron-cloud rl start --no-grad-clip` makes a step about 2 s shorter.
