# nextron-cookbook

Reinforcement learning loops on [NexTron Cloud](https://nextron.physionlabs.ai)'s Serverless RL. An *RL run* is a model
loaded on a node of GPUs: your loop decides what to sample and what to train on, NexTron runs both with its own trainer.
There are two ways to drive one, and this repository has tested examples of both:

- **NexTron's own calls** (`nextron-cloud`): `sample` a batch of chat prompts in one request, score the replies in
  your code, `train` on the rewards. Nothing that was sampled travels back. Start here for a new loop.
- **Tinker's API**: [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) and other code written
  for the `tinker` SDK run unchanged.

NexTron Cloud has two modes:

| Mode | You send | NexTron runs | Client |
|---|---|---|---|
| Managed training | a dataset and settings | the whole training run | [`nextron-cloud jobs`](https://github.com/Physion-Labs/nextron-python) |
| Serverless RL | prompts and rewards from your own loop | every sample and every optimizer step | `nextron-cloud` (`rl.connect`), or the `tinker` SDK pointed at the run |

This repository is for the second mode: loops and recipes with the settings they were measured with, the results,
and what is supported.

## Quickstart: your loop with NexTron's own calls

```bash
git clone https://github.com/Physion-Labs/nextron-cookbook && cd nextron-cookbook
uv venv && uv pip install --prerelease=allow "nextron-cookbook[math] @ git+https://github.com/Physion-Labs/nextron-cookbook" nextron-cloud
source .venv/bin/activate
export NEXTRON_API_KEY="nk_live_..."
nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes     # prints the run's id; billed from ready
nextron-cloud rl wait rl_...                                       # until the model is loaded, 6-8 minutes
python examples/math_loop.py rl_... --steps 10 --eval-every 10     # MATH: sample, score here, train; stops the run
nextron-cloud rl stop rl_... --wait                                # always, in case the loop stopped early
```

The loop itself, from [examples/math_loop.py](examples/math_loop.py):

```python
with NexTron().rl.connect(run_id) as live:
    for step in range(steps):
        batch = live.sample([prompt(p) for p, _ in problems], n=8, max_tokens=512)     # one request, every prompt
        rewards = [[score(reply, answer) for reply in group] for (_, answer), group in zip(problems, batch)]
        result = live.train(batch, rewards, learning_rate=2e-6)                         # one optimizer step
    live.finish()
```

On the MATH task this reaches the same accuracy as the cookbook's recipe through Tinker's API (MATH-500 35.0% -> 78.0%
in 10 steps against 34.8% -> 76.4%) with shorter steps: 34-45 s against about 60 s, because a step is a handful of
requests instead of two per prompt and only the rewards are uploaded.

## Quickstart: tinker-cookbook recipes

```bash
uv venv && uv pip install --prerelease=allow "nextron-cookbook[math] @ git+https://github.com/Physion-Labs/nextron-cookbook" nextron-cloud
source .venv/bin/activate                                           # or prefix each command with `uv run`
export NEXTRON_API_KEY="nk_live_..."

nextron-cloud rl start --model qwen3.8-27b --max-hours 1           # the price only (US cents): exits 3, starts nothing
nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes     # start it; prints the run's id
nextron-cloud rl wait rl_...                                       # until the model is loaded, 6-8 minutes
eval "$(nextron-cloud rl env rl_...)"                              # sets TINKER_BASE_URL and TINKER_API_KEY

nextron-cookbook check                       # the RL run answers: model, context length, one sampled reply
nextron-cookbook run gsm8k                   # 3 short steps of the cookbook's math_rl recipe, about 2 minutes
nextron-cloud rl stop rl_... --wait          # always: billing stops here at the latest
```

An RL run holds **one training loop**. When the loop's process ends normally the RL run stops by itself, but stop
it anyway: after a crash or a kill it stays up and billed for another 3 minutes or more. For the next recipe start
a new RL run and apply its `env` again, e.g. for held-out accuracy before and after:

```bash
nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes
nextron-cloud rl wait rl_... && eval "$(nextron-cloud rl env rl_...)"
nextron-cookbook run math                    # MATH with a 512-token budget, MATH-500 at steps 0, 10 and 20; about 25 minutes
nextron-cloud rl stop rl_... --wait
```

A new RL run for the same model may wait in `queued` for about a minute while the previous one releases its GPUs.

`run` starts the cookbook's own module with tested arguments and prints the command. Add any of the cookbook's
`key=value` arguments to replace ours:

```bash
nextron-cookbook run math max_steps=40 learning_rate=1e-6 groups_per_batch=32 wandb_project=my-project
nextron-cookbook show math      # the full command, to copy into your own script
```

Your own Tinker code needs no `nextron-cookbook` at all: set the two environment variables and run it.
[examples/minimal_loop.py](examples/minimal_loop.py) is the smallest loop: sample, score, train one step, sample again
(it borrows tinker-cookbook's renderer for the chat template).

## Results

One node of 8 H100, Qwen3.8-27B, all weights trained, thinking off; tinker 0.39.0 and tinker-cookbook 0.6.1 unmodified.
Commands and per-step numbers are in [docs/results.md](docs/results.md).

| Recipe | Held-out metric | Before | After | Steps |
|---|---|---|---|---|
| `math` (MATH-500, 512-token replies) | correct | 34.8% | 76.4% | 10 |
| | | | 84.4% | 30 |
| `code` (687 Codeforces and LiveCodeBench problems) | all tests pass | 20.7% | 37.8% | 10 |
| | | 22.6% | 41.0% | 6 (a second run) |

## What NexTron's trainer brings

- **All weights, not an adapter.** An RL run trains the full 27B model on one 8-GPU node. `lora_config` is ignored.
- **One optimizer step is one NexTron step.** `forward_backward` + `optim_step` run as a single packed step:
  2,100-2,450 trained tokens per second per GPU inside the step in the runs above.
- **Sampling and training share the GPUs.** vLLM sleeps while a step runs and wakes with the new weights in 3-4
  seconds. Steps that follow each other directly (`num_substeps`) run back to back without waking it.
- **The sampler and the trainer agree.** The trainer's probability of a sampled token over the sampler's was
  0.9998-1.0006; the cookbook's `kl_sample_train` 1e-4 to 7e-4.
- **You see what a step cost.** `nextron/train_seconds`, `nextron/train_tokens` and
  `nextron/train_tokens_per_s_per_gpu` appear among the cookbook's own metrics.

## Documentation

- [docs/serverless-rl.md](docs/serverless-rl.md): what an RL run is, what each Tinker call does on NexTron, lifecycle
- [docs/parameters.md](docs/parameters.md): what you can change, and where
- [docs/compatibility.md](docs/compatibility.md): Tinker API and cookbook recipes, supported or not
- [docs/recipes.md](docs/recipes.md): each recipe: setup, command, what to expect
- [docs/results.md](docs/results.md): measurements
- [AGENTS.md](AGENTS.md): rules for coding agents
