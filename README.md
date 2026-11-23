# nextron-cookbook

[![tests](https://github.com/Physion-Labs/nextron-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/Physion-Labs/nextron-cookbook/actions/workflows/ci.yml)
[![nextron-cloud on PyPI](https://img.shields.io/pypi/v/nextron-cloud?include_prereleases&label=nextron-cloud)](https://pypi.org/project/nextron-cloud/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Tested recipes for [NexCloud](https://nextron.physionlabs.ai). Each one is a real task with the settings we
measured, the command that runs it and the results it got on the live service.

```bash
git clone https://github.com/Physion-Labs/nextron-cookbook && cd nextron-cookbook
uv venv && uv pip install --prerelease=allow -e ".[math]"
source .venv/bin/activate
export NEXTRON_API_KEY="nk_live_..."       # create one in the console under API keys
```

## Recipes

| Recipe | How it trains | Task |
|---|---|---|
| [`sft_basic.py`](recipes/sft_basic.py) | Managed SFT | 20 steps on a sample dataset already on the service |
| [`rl_basic.py`](recipes/rl_basic.py) | Serverless RL | arithmetic with a 0/1 reward, 5 steps |
| [`rl_images.py`](recipes/rl_images.py) | Serverless RL | images in the prompts: name a picture's colour, 3 steps |
| [`tinker_loop.py`](recipes/tinker_loop.py) | Serverless RL, `tinker` SDK | sample, score, train one step |
| [`chat_sft`](recipes/chat_sft/) | Managed SFT | [NoRobots](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 9,500 human-written conversations |
| [`video_sft`](recipes/video_sft/) | Managed SFT on video | count a square's bounces in 800 short videos, and name the walls |
| [`chat_dpo`](recipes/chat_dpo/) | DPO | [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized): 4,000 prompts, each with a preferred and a rejected reply |
| [`math_rft`](recipes/math_rft/) | RFT | MATH with the built-in `boxed` reward: NexCloud runs the whole loop |
| [`math_rl`](recipes/math_rl/) | Serverless RL | MATH with your own grader |
| [`guess_number`](recipes/guess_number/) | Serverless RL | a multi-turn game: find a hidden number from "too low" / "too high" |
| [`code_rl`](recipes/code_rl/) | Serverless RL | competitive programming: your code runs each solution's tests in a sandbox, a failed solution gets one more attempt |
| [`code`](docs/recipes.md#code-multi-turn-with-a-tool) | Serverless RL, `tinker` SDK | competitive programming: two turns with a tool, the generated code runs in a sandbox (tinker-cookbook's `code_rl`, unchanged) |
| tinker-cookbook `math`, `gsm8k`, `sl` | Serverless RL, `tinker` SDK | [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)'s recipes, unchanged |

Each recipe's settings and measured results are in its own README; an overview is in [recipes/](recipes/README.md).

## Four ways to train

| | You send | NexCloud runs | Billed |
|---|---|---|---|
| **Managed SFT** | a dataset and settings | the whole job; you get a checkpoint | per million tokens trained |
| **DPO** (preference training) | prompts, each with a chosen and a rejected reply | the whole job; you get a checkpoint | per million tokens trained |
| **RFT** (reinforcement fine-tuning) | prompts with answers and a built-in reward | the whole GRPO loop; you get a checkpoint | per million tokens |
| **Serverless RL** | prompts and rewards from your own loop | every sample and every optimizer step, on GPUs kept for you | by the second the RL run is ready |

After a job (Managed SFT, DPO or RFT), an **inference run** serves the trained model with an OpenAI-compatible API, so
you can try it before downloading it, on the models that list `inference` in `nextron-cloud models`. The model of a
Serverless RL run is downloaded instead: an inference run takes a job's id, not a run's.

## Models

What each model trains and serves, with the longest sample (prompt and reply, in tokens) each method takes. The list
the service has now, with prices: `nextron-cloud models`, or the console's Models page.

| Model | Size (active) | GPUs | SFT | LoRA | DPO | RFT | Serverless RL | Inference | Images, videos | Longer samples |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3.8-27b` | 27B | 8 | 64K | 64K | 32K | 16K | 16K | 16K | both | `qwen3.8-27b-256k`: SFT, 256K |
| `gemma-4-26b-a4b` | 26B (4B) | 8 | 64K | 64K | 32K | | 16K | 16K | both | `gemma-4-26b-a4b-256k`: SFT, 256K |
| `muse-glimmer-30b` | 30B | 8 | 128K | 64K | 32K | | 16K | 16K | images | |
| `qwen3.8-next-180b` | 180B (6B) | 16 | 64K | 64K | 32K | | 16K | | both | `qwen3.8-next-180b-256k`: SFT, 256K |
| `nemotron-3-super-120b` | 120B (12B) | 16 | 64K | 64K | 32K | | | | | `nemotron-3-super-120b-256k`: SFT, 256K |
| `mimo-311b` | 309B (15B) | 32 | 32K | 32K | 32K | | | | | |
| `glm-5.3-flash` | 321B (17B) | 32 | 32K | 32K | 32K | | | | | |
| `minimax-m3.1-flash` | 428B (26B) | 32 | 32K | 32K | 32K | | | | | |
| `deepseek-v4.1-flash` | 763B (16B) | 64 | 32K | 32K | 32K | | | | | |
| `kimi-k3.1-pro` | 2.8T (104B) | 224 | 256K | | | | | | | `kimi-k3.1-pro-1m`: SFT, 1M |
| `olmo-3-7b` | 7.3B | 8 | 64K | | | | | 16K | | |
| `olmo-3-7b-instruct-sft`, `olmo-3-7b-instruct`, `olmo-3-7b-think` | 7.3B | 8 | 64K | | | | | 16K | | |
| `qwen3.5-4b` | 4B | 8 | 64K | 64K | | | | 16K | both | |
| `qwen3.5-9b`, `qwen3.5-9b-base` | 9B | 8 | 64K | 64K | | | | 16K | both | |
| `qwen3.6-35b-a3b`, `qwen3.5-35b-a3b-base` | 35B (3B) | 8 | 64K | 64K | | | | 16K | both | |

GPUs: the H100s of the smallest setup a full-parameter job of that model runs on (some models take larger setups too).
LoRA runs on fewer for the larger models: 8 for `nemotron-3-super-120b` and `qwen3.8-next-180b`, 16 for `mimo-311b`,
`glm-5.3-flash` and `minimax-m3.1-flash`, 32 for `deepseek-v4.1-flash`. SFT and DPO train all weights; LoRA trains an adapter. RFT is managed GRPO with a built-in reward; Serverless RL runs your own loop.
Images and videos go in user messages (SFT, RFT, Serverless RL and the long-context ids; not DPO or validation sets) and
are opened per organisation: `nextron-cloud models` lists them in a model's `input` when they are open to yours. The
format, video settings and their cost: the SDK guide's
[media.md](https://github.com/Physion-Labs/nextron-python/blob/main/skills/nextron-cloud/references/media.md).

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
nextron-cloud inference stop inf_... --wait                       # always, when you are done: it is billed by time
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
- [docs/serverless-rl.md](docs/serverless-rl.md): what an RL run is, what each Tinker call does on NexCloud, lifecycle
- [docs/parameters.md](docs/parameters.md): what you can change, and where
- [docs/compatibility.md](docs/compatibility.md): Tinker API and cookbook recipes, supported or not
- [docs/recipes.md](docs/recipes.md): the tinker-cookbook recipes: setup, command, what to expect
- [docs/results.md](docs/results.md): measurements
- [AGENTS.md](AGENTS.md): rules for coding agents
