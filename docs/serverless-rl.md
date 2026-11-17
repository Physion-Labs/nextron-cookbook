# Serverless RL

An RL run is a model loaded on one node of 8 GPUs (two nodes for `qwen3.8-next-180b`), serving Tinker's HTTP API. The `tinker` SDK talks to it when
`TINKER_BASE_URL` points at it. Your process (the training loop, the environments, the reward code) runs on your
machine; only tokens, sampler logprobs and advantages travel.

## Lifecycle

```bash
nextron-cloud rl start --model qwen3.8-27b --max-hours 2 --max-context 16384 --name math-try --yes
nextron-cloud rl wait rl_...           # until the model is loaded, 6-8 minutes; progress on stderr
eval "$(nextron-cloud rl env rl_...)"
python my_tinker_loop.py                      # or: nextron-cookbook run math
nextron-cloud rl stop rl_... --wait   # always; returns when the GPUs are released
```

In a script, stop it in a `trap` so a failed loop does not leave the RL run up:

```bash
trap 'nextron-cloud rl stop "$RUN"' EXIT
```

States: `queued` -> `starting` (the model loads; not billed) -> `ready` -> `closing` (not billed) -> `closed`, or `failed`.

Price: per GPU-hour, by the kind of GPU: $3.50 per H100-hour. An RL run has one node of 8 GPUs (two nodes, 16 GPUs, for `qwen3.8-next-180b`), and you pay for all of them for the seconds the
RL run is `ready` (`billable_seconds`). `rl start` without `--yes` shows the price and what is held for
`--max-hours`; a failed RL run costs nothing.

After a crash, or whenever you are not sure, look and close. `billable_seconds` stops growing at `closing`:

```bash
nextron-cloud rl list                    # anything "ready" is being billed
nextron-cloud rl stop rl_... --wait
nextron-cloud rl get rl_...             # "closed", and the billable_seconds you pay for
```

A `trap` does not run when the shell itself is killed with SIGKILL; then the missing heartbeats stop the run
after 3 minutes, and `--max-hours` and `--idle-timeout` remain as limits.

An RL run holds **one training loop**. It begins to stop when:

- you close it;
- your training client ends its session: a `with tinker.ServiceClient() as service:` block ends, or you call
  `service.close("success").result()`. The state becomes `closing` within about 15 seconds. tinker-cookbook's
  `rl.train` and `supervised.train` loops, which the recipes here run, do this;
- your training client goes away without ending its session: a script that made a bare `tinker.ServiceClient()` and
  returned (the SDK does not end the session by itself when the process exits), a killed process, a lost network. The
  SDK's heartbeats stop, and after 3 minutes without one the run stops as if the client had finished. Those minutes
  are billed: stop it yourself to stop billing sooner;
- `--max-hours` have passed;
- `--idle-timeout` seconds (30 minutes unless you set it) passed without a sample or a training step.

Billing is by the RL run's time from `ready` until it begins to stop. The engine then saves the model if it trained any
step and releases the GPUs about a minute later (`closed`); `qwen3.8-next-180b` takes about 20 minutes to load and
about 15 to save, neither billed. The saved model is fetched with the run id:
`nextron-cloud jobs wait-checkpoint rl_...`, `jobs download rl_... DEST`.

## What each call does

| Tinker call | On NexTron |
|---|---|
| `create_lora_training_client(base_model=...)` | attaches to the run's model. All weights are trained; rank and the other LoRA settings are ignored. A second training client is refused. |
| `save_weights_and_get_sampling_client()` | vLLM wakes with the current weights (3-4 s after a step; instant if already awake). |
| `sample(...)` | joins vLLM's running batch on all 8 GPUs. Equal prompts with equal settings are merged into one request with a larger n. The stop token is part of the reply, with its logprob. |
| `forward(data, loss_fn)` | no update: vLLM scores the sequences under the current weights (under a second for a few sequences). |
| `forward_backward(data, loss_fn)` | queued. Nothing runs yet, and its future resolves with the step: do not wait for it before `optim_step`. |
| `optim_step(adam_params)` | vLLM sleeps; everything queued since the last step trains as one NexTron step with these Adam parameters; the `forward_backward` futures resolve with the trained model's logprobs. |
| `save_state()` / `save_weights...(name)` | returns a path, but named checkpoints during a run are not kept yet. |
| `create_sampling_client(base_model=...)` | samples from the run's current weights. Refused once training has started, because the base weights are gone from the GPUs. |

## Timing of one step

From the `math` recipe (512 replies of up to 512 tokens): sampling 20-37 s, training 12-18 s, wake 3-4 s. From
the `code` recipe (256 two-turn episodes of up to 4,096 tokens a turn): sampling 140-190 s, training 55-60 s.
Sampling time is set by the longest replies, so `max_tokens` is the first thing to look at.

After a step the engine waits 0.3 s for another `optim_step`. If one comes (the cookbook's `num_substeps`), it trains
again right away; if not, vLLM wakes so the next `sample` is served without a wait.

## What you will see in the cookbook's output

- `Pricing for Qwen/Qwen3.8-27B: TokenCostRates(...)` and the `cost/*` metrics are the cookbook's own estimate at
  Tinker's per-token prices. They are not your NexTron bill: an RL run is billed by time (`billable_seconds`).
- The run name contains `32rank` and the end of a run prints `tinker://nx-.../weights/final`. Both come from the
  cookbook: all weights are trained, and those paths are not something you can load. The trained model is the
  run's (`jobs wait-checkpoint rl_...`). `rl_...` is the only id the `nextron-cloud` commands take.
- `Retrying request ... because of try_again` is the SDK waiting for a result that takes longer than one poll (35 s):
  a long sampling round, the first training step, an evaluation. It is normal and needs no action.
- The first step of an RL run is slower than the later ones (warm-up), whatever the loss: its training takes 20 s
  more on a small batch and about a minute more on the `math` recipe's 320K tokens, and its sampling can take 30 s more.
- A recipe run with `eval_every=0` still downloads its evaluation dataset's metadata at start; nothing is evaluated.
- `404 Not Found` lines from huggingface.co are the tokenizer library probing optional files. The first run also
  downloads the tokenizer and the recipe's dataset to your machine (seconds to a minute, while the run is billed).

## Differences from Tinker

- Full-parameter training, so learning rates are about 10x lower than the cookbook's LoRA defaults (1e-6 to 2e-6 for
  RL on the 27B model).
- One model per RL run. A recipe that needs a second model (a grader, an opponent, a teacher) has to point that
  client at another endpoint.
- The losses and calls in [compatibility.md](compatibility.md) marked "not yet".
