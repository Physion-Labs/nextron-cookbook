# Compatibility

Tested with `tinker` 0.39.0 and `tinker-cookbook` 0.6.1.

## Tinker API

| | Status |
|---|---|
| Sampling: `num_samples`, `max_tokens`, stop strings and token ids, `temperature`, `top_p`, `top_k`, logprobs | yes |
| `forward_backward` + `optim_step`, `loss_fn` `importance_sampling`, `ppo`, `cispo` | yes |
| `loss_fn` `cross_entropy` (weights 0, or one value per sequence) | yes |
| Single-turn and multi-turn sequences (environment tokens between the model's turns) | yes |
| Adam parameters per step | yes, except `grad_clip_norm` |
| `forward` (no update) and `compute_logprobs`: the current weights' logprobs, scored by the sampler | yes |
| `get_server_capabilities`, `get_info`, tokenizer lookup | yes |
| `loss_fn` `stable_reinforce` | not yet (its definition is not public). Pass `loss_fn=importance_sampling`. |
| Per-token advantages (the cookbook's `kl_penalty_coef` > 0) | not yet |
| Custom losses (`forward_backward_custom`) | not yet |
| Sampling from earlier weights or a saved checkpoint | not yet |
| Named checkpoints during a run, `load_state` | not yet. The model is saved when the run ends. |
| LoRA | no: all weights are trained |
| Images in prompts | not yet |
| More than one training client, or a second model, in one RL run | no |

Unsupported calls fail with a message that says so; they do not train something else silently.

One difference in timing: `forward_backward` only queues data, and its future resolves when the `optim_step` after it has
run. Submit `optim_step` before you wait for the `forward_backward` result, as the cookbook's loops do:

```python
fwd_bwd = training_client.forward_backward(data, "importance_sampling")
optim = training_client.optim_step(tinker.AdamParams(learning_rate=2e-6))
result, _ = fwd_bwd.result(), optim.result()
```

[examples/minimal_loop.py](../examples/minimal_loop.py) is a complete loop of this kind in 60 lines.

One `optim_step` takes one `loss_fn`: every `forward_backward` since the last step must use the same loss and thresholds.

## Cookbook recipes

| Recipe | Status | Notes |
|---|---|---|
| `math_rl` (arithmetic, gsm8k, math, polaris, deepmath) | tested: gsm8k, math | `loss_fn=importance_sampling` |
| `code_rl` | tested | needs SandboxFusion; 16K context |
| `rl_basic` | tested (2 steps) | as shipped: 2,048 replies a step; pass `learning_rate=1e-6` |
| `sl_basic` | tested, with its test-set loss (`compute_logprobs`) | |
| `chat_sl` | same loop as `sl_basic`; untested | |
| `rl_loop` | tested (3 steps) | `loss_fn=importance_sampling learning_rate=1e-6` |
| `verifiers_rl` | cookbook bug at 0.6.1 (`custom_do_group_rollout` lacks `termination`) | not a NexTron issue |
| `rubric`, `preference`, `multiplayer_rl` | not in one RL run | they need a second model (grader or opponent) |
| `distillation`, `prompt_distillation` | not in one RL run | they need a teacher model |
| `search_tool`, `harbor_rl` | untested | |
| `vlm_classifier`, `audio` | no | text only |
