# MATH with your own grader (Serverless RL)

Hendrycks MATH problems with tinker-cookbook's prompt and grader, driven by NexCloud's own calls: `sample` a batch,
grade every reply on your machine, `train` on the rewards. The reward is 1 for a correct `\boxed{}` answer, minus 0.1
when the reply is cut at the token limit.

```bash
pip install nextron-cloud "nextron-cookbook[math] @ git+https://github.com/Physion-Labs/nextron-cookbook"
export NEXTRON_API_KEY=nk_live_...
nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes
nextron-cloud rl wait rl_...
python recipes/math_rl/train.py rl_... --steps 10 --eval-every 10           # MATH-500 at step 0 and 10
nextron-cloud rl stop rl_... --wait
```

The script ends the RL run, and a run holds one training loop: start a new run for another try, for example the
variant whose steps are sized to fill the trainer's rows, `--steps 8 --rows 1 --eval-every 4`.

## Settings

- 8 replies a problem, 512 new tokens, learning rate 2e-6, temperature 1.
- `--rows N` sizes every step to fill N of the trainer's packed rows on each GPU (aiming at 90%): more tokens a
  second, and cheaper per trained token.

## Results

| Model | Step sizing | MATH-500 | Trained tokens/s/GPU |
|---|---|---|---|
| `qwen3.8-27b` | 64 problems | 35.0% -> 78.0% in 10 steps | 2,245 |
| `qwen3.8-27b` | `--rows 1` | 35.2% -> 74.6% in 10 steps | 2,305-2,396 |
| `muse-glimmer-30b` | `--rows 1` | 57.8% -> 79.4% in 8 steps | 1,860-2,210 |
| `gemma-4-26b-a4b` | `--rows 1` | 51.2% -> 59.4% in 8 steps | 6,660-7,260 |
| `qwen3.8-next-180b` (16 GPUs) | `--rows 1` | reward 0.28 -> 0.35 in 3 steps | 2,436-2,443 |

A step of 64 problems takes 34-45 s on Qwen3.8-27B (about half sampling); the tinker-cookbook recipe on the same task
takes about 60 s. More in [docs/results.md](../../docs/results.md).
