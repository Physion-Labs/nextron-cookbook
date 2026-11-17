# Recipes

Each recipe is the cookbook's own module with arguments that were measured on NexTron. `nextron-cookbook show <name>`
prints the command; `nextron-cookbook run <name> key=value ...` runs it with your changes. Logs go to
`runs/<name>-<time>/` (`metrics.jsonl`, the cookbook's HTML traces).

An RL run holds one training loop, so each `nextron-cookbook run` needs its own RL run: start, wait,
`eval "$(nextron-cloud rl env rl_...)"` again, run, stop.

The recipes are set up for `qwen3.8-27b` (its tokenizer and chat template): `nextron-cookbook run` refuses an RL run
of another model.

## Reading the results

`metrics.jsonl` has one row per optimizer step, numbered from 0, and a last row with totals. Held-out evaluations are
in the rows that have `test/...` keys; step N there means "after N optimizer steps" (so `max_steps=4 eval_every=2`
evaluates at 0, 2 and 4). Do not take the last row:

```bash
python - <<'PY'
import glob, json
for line in open(sorted(glob.glob("runs/math-*/metrics.jsonl"))[-1]):
    row = json.loads(line)
    if "test/env/all/correct" in row:
        print(row["step"], row["test/env/all/correct"])
PY
```

An evaluation runs before the step it is numbered with and takes sampling time of its own (MATH-500: one to two
minutes; the code recipe's 687 problems: about 20 minutes).

## gsm8k: is the RL run working?

```bash
nextron-cookbook run gsm8k
```

3 steps of 16 problems x 4 replies, 23-45 s a step. The base model already gets 98% of GSM8K, so the reward does not
move; this only shows that sampling, training and the next round's sampling work. Look for `Training completed
successfully`, `env/all/correct` near 0.98 and the `nextron/*` metrics.

## math: held-out accuracy

```bash
nextron-cookbook run math                 # 20 steps, MATH-500 at steps 0, 10 and 20; about 25 minutes
nextron-cookbook run math max_steps=60    # the run in docs/results.md
```

64 problems x 8 replies per step, replies capped at 512 tokens. With thinking off the base model runs out of room on
most problems (34.8% correct); training teaches it to answer within the budget (76.4% after 10 steps). Read
`test/env/all/correct` in `metrics.jsonl`.

## code: multi-turn with a tool

Generated code must run in a sandbox, never on your machine directly. Start SandboxFusion next to your loop:

```bash
docker run -d -p 127.0.0.1:8080:8080 \
  -v "$(python -c 'import tinker_cookbook, os; print(os.path.dirname(tinker_cookbook.__file__))')/recipes/code_rl/sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml" \
  volcengine/sandbox-fusion:server-20250609
export SANDBOX_URL=http://127.0.0.1:8080/run_code

nextron-cloud rl start --model qwen3.8-27b --max-hours 3 --max-context 16384 --yes
nextron-cloud rl wait rl_... && eval "$(nextron-cloud rl env rl_...)"
nextron-cookbook run code
nextron-cloud rl stop rl_... --wait
```

32 problems x 8 episodes per step, up to two turns of 4,096 tokens. A step takes 3-4 minutes (sampling 140-190 s,
training 45-60 s) and the evaluation on 687 held-out problems about 20 minutes, so `eval_every` matters. Read
`test/env/all/by_group/frac_all_good` (all tests pass).

## sl: supervised fine-tuning through an RL run

```bash
nextron-cookbook run sl
```

The cookbook's `sl_basic` on no_robots: `loss_fn=cross_entropy`, one step per batch of 128 conversations, test-set
loss off (`eval_every=0`; `eval_every=8` turns it on: 500 test conversations scored with `compute_logprobs`, test NLL
2.29 -> 1.71 after two steps in our run). Train NLL fell
from 2.17 to 1.46 in five steps; each step trains in 8.5 s, and the cookbook renders the next batch on your CPU for
about a minute. For plain supervised fine-tuning a
managed job (`nextron-cloud jobs create`) is simpler and cheaper; this recipe is for loops that mix supervised and RL
steps.

## Your own task

Start from the cookbook's `rl_basic.py` or `math_rl/train.py`, keep `loss_fn=importance_sampling`, and begin with
`learning_rate` 1e-6 to 2e-6. Before a long run do 2-3 steps (`max_steps=3`) and check three things in the metrics: rewards are not all
equal (`env/all/by_group/frac_mixed` above 0), `optim/kl_sample_train_v2` is around 1e-3 or less, and the step time
is what you expect.
