# For coding agents

This repository runs tinker-cookbook recipes on NexCloud Serverless RL. Read `README.md`, then
`docs/serverless-rl.md`, `docs/parameters.md` and `docs/compatibility.md` before running anything.

Rules:

- An RL run bills by time from the moment its model is loaded. Tell the user before you start one, and stop it
  (`nextron-cloud rl stop rl_... --wait`) after every recipe, whether it succeeded or failed. Check
  `nextron-cloud rl list` before you finish: nothing may be left `ready`.
- Never print the API key or write it to a file. `nextron-cloud rl env` prints shell lines that refer to
  `$NEXTRON_API_KEY`; use `eval` on them.
- One training loop per RL run: a new RL run and a new `rl env` for each recipe. Run `nextron-cookbook check`
  first. Before a long run, do a short one (2-3 steps, e.g. the `gsm8k` recipe as it is).
- The `cost/*` metrics and "Pricing for" lines in the cookbook's output are Tinker's token prices, not this bill.
- Pass `loss_fn=importance_sampling` (or `ppo`, `cispo`) to cookbook recipes whose default is `stable_reinforce`.
  `nextron-cookbook run` already does.
- Learning rates are for full-parameter training: start at 1e-6 to 2e-6 for RL, not the cookbook's LoRA defaults.
- Code written by the model runs only in a sandbox (docs/recipes.md), never directly on the user's machine.
- When something is refused, read the error: unsupported calls say what to use instead. Do not edit the installed
  `tinker` or `tinker_cookbook` packages.
- Report results from `runs/<name>-<time>/metrics.jsonl`: the held-out metric before and after, step time, and the
  `nextron/*` training metrics.
