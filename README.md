# nextron-cookbook

[![tests](https://github.com/Physion-Labs/nextron-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/Physion-Labs/nextron-cookbook/actions/workflows/ci.yml)
[![nextron-cloud on PyPI](https://img.shields.io/pypi/v/nextron-cloud?include_prereleases&label=nextron-cloud)](https://pypi.org/project/nextron-cloud/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Tested recipes for [NexTron Cloud](https://nextron.physionlabs.ai). Each one is a real task with the settings we
measured, the command that runs it and the results it got on the live service.

```bash
git clone https://github.com/Physion-Labs/nextron-cookbook && cd nextron-cookbook
uv venv && uv pip install --prerelease=allow -e ".[math]" nextron-cloud
source .venv/bin/activate
export NEXTRON_API_KEY="nk_live_..."       # create one in the console under API keys
```

## Recipes

| Recipe | How it trains | Task |
|---|---|---|
| [`sft_basic.py`](recipes/sft_basic.py) | Managed SFT | 20 steps on a sample dataset already on the service |
| [`rl_basic.py`](recipes/rl_basic.py) | Serverless RL | arithmetic with a 0/1 reward, 5 steps |
| [`tinker_loop.py`](recipes/tinker_loop.py) | Serverless RL, `tinker` SDK | sample, score, train one step |
| [`chat_sft`](recipes/chat_sft/) | Managed SFT | [NoRobots](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 9,500 human-written conversations |
| [`math_rft`](recipes/math_rft/) | RFT | MATH with the built-in `boxed` reward: NexTron runs the whole loop |
| [`math_rl`](recipes/math_rl/) | Serverless RL | MATH with your own grader |
| [`guess_number`](recipes/guess_number/) | Serverless RL | a multi-turn game: find a hidden number from "too low" / "too high" |
| [`code`](docs/recipes.md#code-multi-turn-with-a-tool) | Serverless RL, `tinker` SDK | competitive programming: two turns with a tool, the generated code runs in a sandbox (tinker-cookbook's `code_rl`, unchanged) |
| tinker-cookbook `math`, `gsm8k`, `sl` | Serverless RL, `tinker` SDK | [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)'s recipes, unchanged |

Each recipe's settings and measured results are in its own README; an overview is in [recipes/](recipes/README.md).

## Three ways to train

| | You send | NexTron runs | Billed |
|---|---|---|---|
| **Managed SFT** | a dataset and settings | the whole job; you get a checkpoint | per million tokens trained |
| **RFT** (reinforcement fine-tuning) | prompts with answers and a built-in reward | the whole GRPO loop; you get a checkpoint | per million tokens |
| **Serverless RL** | prompts and rewards from your own loop | every sample and every optimizer step, on GPUs kept for you | by the second the RL run is ready |

After any of them, an **inference run** serves the trained model with an OpenAI-compatible API, so you can try it before
downloading it.

## Quickstart: managed SFT

```bash
python recipes/chat_sft/data.py                       # NoRobots as chat JSONL
python recipes/chat_sft/train.py                      # uploads it, prints the steps and the price; starts nothing
python recipes/chat_sft/train.py --yes                # train; prints the validation loss as it goes
```

## Quickstart: Serverless RL

```bash
nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes     # prints the run's id; billed from ready
nextron-cloud rl wait rl_...                                       # until the model is loaded, 6-8 minutes
python recipes/math_rl/train.py rl_... --steps 10 --eval-every 10   # sample, score here, train; stops the run
nextron-cloud rl stop rl_... --wait                                # always, in case the loop stopped early
```

The loop, from [recipes/math_rl/train.py](recipes/math_rl/train.py):

```python
with NexTron().rl.connect(run_id) as live:
    for step in range(steps):
        batch = live.sample([prompt(p) for p, _ in problems], n=8, max_tokens=512)     # one request, every prompt
        rewards = [[score(reply, answer)[0] for reply in group] for (_, answer), group in zip(problems, batch)]
        result = live.train(batch, rewards, learning_rate=2e-6)                         # one optimizer step
    live.finish()
```

An RL run holds one training loop: start a new one for the next recipe.

## Try the trained model

```bash
nextron-cloud inference start --job job_... --max-hours 1 --yes   # the job's checkpoint on our GPUs, no download
nextron-cloud inference wait inf_...
nextron-cloud inference url inf_...                               # the base URL for an OpenAI client
```

## tinker-cookbook on Serverless RL

Code written for the `tinker` SDK runs unchanged against an RL run: point it at the run and go.

```bash
nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes
nextron-cloud rl wait rl_... && eval "$(nextron-cloud rl env rl_...)"   # sets TINKER_BASE_URL and TINKER_API_KEY
nextron-cookbook check                       # the RL run answers: model, context length, one sampled reply
nextron-cookbook run gsm8k                   # 3 short steps of the cookbook's math_rl recipe, about 2 minutes
nextron-cloud rl stop rl_... --wait
```

`nextron-cookbook run <name> key=value ...` runs a cookbook recipe with our tested arguments and your changes;
`nextron-cookbook show <name>` prints the command. Measured on one node of 8 H100, Qwen3.8-27B, all weights, tinker
0.39.0 and tinker-cookbook 0.6.1 unmodified ([docs/results.md](docs/results.md)):

| Recipe | Held-out metric | Before | After | Steps |
|---|---|---|---|---|
| `math` (MATH-500, 512-token replies) | correct | 34.8% | 76.4% / 84.4% | 10 / 30 |
| `code` (687 Codeforces and LiveCodeBench problems) | all tests pass | 20.7% | 37.8% | 10 |

## What NexTron's trainer brings

- **All weights, not an adapter.** An RL run trains the full model; `lora_config` is ignored.
- **One optimizer step is one packed step.** 2,100-2,450 trained tokens per second per GPU on Qwen3.8-27B.
- **Sampling and training share the GPUs.** vLLM sleeps while a step trains and wakes with the new weights in 3-4 s.
- **The sampler and the trainer agree.** Trainer over sampler probability of a sampled token: 0.9998-1.0006.
- **You see what a step cost.** `nextron/train_seconds`, `nextron/train_tokens` and `nextron/train_tokens_per_s_per_gpu`.

## Documentation

- [recipes/](recipes/README.md): every recipe, its settings and results
- [docs/serverless-rl.md](docs/serverless-rl.md): what an RL run is, what each Tinker call does on NexTron, lifecycle
- [docs/parameters.md](docs/parameters.md): what you can change, and where
- [docs/compatibility.md](docs/compatibility.md): Tinker API and cookbook recipes, supported or not
- [docs/recipes.md](docs/recipes.md): the tinker-cookbook recipes: setup, command, what to expect
- [docs/results.md](docs/results.md): measurements
- [AGENTS.md](AGENTS.md): rules for coding agents
