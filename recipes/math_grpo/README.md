# MATH with managed GRPO

NexTron runs the whole loop: each round it samples 8 replies to 64 prompts, scores them with the built-in `boxed`
reward (1 when the last `\boxed{}` equals the answer, compared as numbers), and trains one step. You send a dataset of
prompts with answers and get a checkpoint. No code runs on your side during training.

```bash
pip install nextron-cloud datasets
export NEXTRON_API_KEY=nk_live_...
python recipes/math_grpo/data.py                  # math_numeric.jsonl: 4,846 MATH problems with a numeric answer
python recipes/math_grpo/train.py                 # uploads it, prints the most it can cost; starts nothing
python recipes/math_grpo/train.py --yes           # train; prints each round
```

## Settings

- The prompt asks for brief reasoning and `--max-completion-length 1024`. A reply cut at the limit scores 0, and
  every generated token is billed: the default limit of 8,192 costs several times as much for the same reward here.
- 10 rounds of 64 prompts x 8 replies, learning rate 2e-6.
- The `boxed` reward compares numbers, so `data.py` keeps the problems whose answer is one number. For fractions,
  expressions or your own grader, use Serverless RL ([../math_rl](../math_rl)).

## Results

Qwen3.8-27B, one node of 8 H100:

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| reward | 0.746 | 0.623 | 0.814 | 0.750 | 0.754 | 0.814 | 0.793 | 0.789 | 0.828 | 0.865 |
| reply tokens | 614 | 654 | 485 | 510 | 456 | 388 | 377 | 374 | 371 | 309 |

The model learns to answer in half as many tokens while the reward rises. A round takes 38-59 s (sampling 16-25 s),
training at 2,100-2,350 tokens/s/GPU. 2.82M tokens were billed: $8.48 (the estimate held $35.89, the most the job
could have generated). Two thirds of the prompts gave every reply the same score (`zero_variance_fraction` 0.66-0.81):
Qwen3.8-27B already solves most of MATH, so a harder dataset teaches more per round.
