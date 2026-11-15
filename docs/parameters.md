# What you can change, and where

## 1. Recipe arguments: your side

Everything the cookbook exposes is yours to set, because the cookbook's code runs on your machine:

```bash
nextron-cookbook run math group_size=16 groups_per_batch=32 max_tokens=1024 temperature=0.8 \
    num_substeps=2 max_steps=40 eval_every=5 seed=1 wandb_project=my-project
```

`nextron-cookbook show <recipe>` prints the full command. Batch shape (`group_size`, `groups_per_batch`), reply
length, temperature, the number of optimizer steps per batch, evaluation and logging are all here.

## 2. Optimizer: per step

`optim_step` carries Adam parameters and the run uses them for that step:

| `tinker.AdamParams` field | Used |
|---|---|
| `learning_rate` | yes, per step (so the cookbook's schedules work) |
| `beta1`, `beta2`, `eps` | yes |
| `weight_decay` | yes |
| `grad_clip_norm` | no: clipping is the run's `--max-grad-norm`, or `--no-grad-clip` |

Cookbook recipes take `learning_rate=...`. In your own loop:

```python
training_client.optim_step(tinker.AdamParams(learning_rate=2e-6, beta1=0.9, beta2=0.95, eps=1e-8))
```

## 3. Loss: per step

`forward_backward(data, loss_fn=..., loss_fn_config=...)`:

| `loss_fn` | Thresholds (`loss_fn_config`) |
|---|---|
| `importance_sampling` | none |
| `ppo` | `clip_low_threshold` 0.8, `clip_high_threshold` 1.2 |
| `cispo` | `clip_low_threshold` 0.0, `clip_high_threshold` 4.0 |
| `cross_entropy` | none; weights must be 0 or one value per sequence |

```bash
nextron-cookbook run math loss_fn=ppo 'loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}'
```

## 4. RL run settings: fixed when the run is started

| `nextron-cloud rl start` | Default | Meaning |
|---|---|---|
| `--model` | | the model to load |
| `--max-hours` | required | the run stops itself after this long; credits for it are held when it starts |
| `--max-context` | 16384 | longest prompt + reply, and longest training sequence, in tokens |
| `--max-grad-norm` | 1.0 | gradient clipping for every step |
| `--no-grad-clip` | clipping on | no gradient clipping. A step is 1.5-2.7 s shorter (the optimizer then runs beside the backward pass): 2,280-2,350 -> 2,600-2,650 trained tokens/s/GPU on `muse-glimmer-30b`. Clipping guards against a rare very large gradient; keep it on unless your runs are stable |
| `--idle-timeout` | 1800 | seconds without a sample or a training step before the run stops |

A shorter context leaves more GPU memory for sampling in parallel; use the smallest one your task needs.
