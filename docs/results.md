# Results

Setup for every run: one node of 8 H100, Qwen3.8-27B, all weights trained, thinking off
(`renderer_name=qwen3_8_disable_thinking`), `tinker` 0.39.0 and `tinker-cookbook` 0.6.1 unmodified, loss
`importance_sampling`.

## math

`nextron-cookbook run math max_steps=60` (64 problems x 8 replies, 512 tokens, learning rate 2e-6). 48 minutes for
the job: model load, 60 steps, 7 evaluations of 500 problems.

| Step | MATH-500 correct | Answer in the right format | Reply tokens |
|---|---|---|---|
| 0 | 34.8% | 35.4% | 451 |
| 10 | 76.4% | 94.2% | 174 |
| 20 | 82.2% | 92.2% | 196 |
| 30 | 84.4% | 92.0% | 209 |
| 60 | 84.4% | 91.0% | 199 |

Per step: sampling 34-37 s and training 17-18 s (about 2,240 trained tokens/s/GPU) in the first steps, when replies
use the whole 512 tokens; 20-30 s and about 12 s once replies are shorter; wake 3-4 s. The first training step of a
RL run took 75 s (warm-up). Most of the gain is the model learning to finish inside 512 tokens: at step 0 it runs out
of room on 65% of the problems.

A first-time user's short run through the whole service path (`nextron-cookbook run math max_steps=4 eval_every=2
learning_rate=1e-6`, three evaluations): 478 s, MATH-500 34.8% -> 37.0% -> 40.6% at steps 0, 2, 4. The lower learning
rate moves more slowly than the 2e-6 above.

## code

`nextron-cookbook run code` (32 problems x 8 episodes, up to two turns of 4,096 tokens, learning rate 2e-6), 687
held-out Codeforces and LiveCodeBench problems, one episode each.

Two runs:

| Run | Step | All tests pass | Reward |
|---|---|---|---|
| first | 0 | 20.7% | 0.130 |
| | 10 | 37.8% | 0.345 |
| second (`max_steps=6 eval_every=6`, 64 minutes in all) | 0 | 22.6% | 0.150 |
| | 6 | 41.0% | 0.370 |

In the second run the training reward went 0.25, 0.40, 0.38, 0.50, 0.65, 0.49 over the six steps and the model used
more second turns (1.6 -> 1.8-1.9 turns an episode). Per step: sampling 137-156 s, training 44-59 s at 2,390-2,450
trained tokens/s/GPU, 192-225 s in total; 0.9-1.2M tokens trained per step. The first training step took 110 s
(warm-up) and an evaluation of the 687 problems about 16 minutes.

## gsm8k

`nextron-cookbook run gsm8k` from a laptop: 3 steps in 93 s straight to the run's gateway (45 s for the first,
23-24 s after) and 44.5 / 24.7 / 24.7 s a step through the service path (API, relay, cluster agent): the relay adds
about a second a step. `env/all/correct` 0.98, 0.94; base accuracy is already 98%, so there is nothing to gain. A step here
trains only about 31K tokens, so the fixed cost of a step (about 7 s) dominates and tokens/s/GPU reads low (510-540).

## sl

`nextron-cookbook run sl` (the cookbook's `sl_basic`, `loss_fn=cross_entropy`, 128 conversations of no_robots a step,
learning rate 1e-5): train NLL 2.17, 1.81, 1.65, 1.70, 1.46 over the first five steps; with `eval_every=2`, test NLL
2.29 at step 0 and 1.71 at step 2 (500 conversations, `compute_logprobs`). A step trains in 8.5 s; the
cookbook's own rendering of a batch takes over a minute on one CPU core, which is what you wait for. These numbers
were taken with the data rendered by `sl_basic`'s default for the model (`qwen3_8_xhigh_reasoning`); the recipe now
renders it with thinking off like the others, and has not been measured again.

## rl_basic and rl_loop

The cookbook's two tutorial loops as shipped, a few steps each, with `learning_rate=1e-6` (and
`loss_fn=importance_sampling` for `rl_loop`):

| Recipe | Reward by step | Training per step | Tokens trained per step | Trained tokens/s/GPU |
|---|---|---|---|---|
| `rl_basic` (2,048 replies a step) | 0.640, 0.712 | 41 s (the first: 125 s) | 820-850K | 2,480 |
| `rl_loop` | 0.758, 0.750, 0.802 | 17-22 s (the first: 92 s) | 290-390K | 2,200-2,290 |

## The MATH loop with NexCloud's own calls

`python recipes/math_rl/train.py rl_... --steps 10 --eval-every 10`: the `math` recipe's problems, prompt and reward, with
`rl.connect` (`sample`, score here, `train`) instead of the tinker SDK. 64 problems x 8 replies, 512 tokens,
learning rate 2e-6.

| | NexCloud's own calls | The cookbook recipe through Tinker's API |
|---|---|---|
| MATH-500 at step 0 / 10 | 35.0% / 78.0% | 34.8% / 76.4% |
| A step while replies are ~410-450 tokens | 43-48 s | about 60 s |
| of it sampling / training | 25-29 s / 18-19 s | 34-37 s / 17-18 s plus the upload |

Measured through the public API from a laptop (an RL run on each of two clusters). A step is a handful of requests
instead of two per prompt, and only rewards are uploaded. The run's own record of a step says where the time went:
for a step of ~1,950 replies, `generation_s` 71-80, `train_s` 62, `client_s` 4 (scoring and the network).

### Sizing a step to the trainer's rows

`--rows N` sizes every step to fill N of the trainer's packed rows on each GPU (a row holds 57,344 tokens in a
`qwen3.8-27b` RL run; `train` returns `row_tokens` and `row_fill`).

| Step sizing | Problems a step | Row fill | Trained tokens/s/GPU | MATH-500 |
|---|---|---|---|---|
| 64 problems (default) | 64 | 0.67-0.71 | 2,245 | 35.0% -> 78.0% in 10 steps |
| `--rows 1` | 85-128 | 0.89-0.92 | 2,305-2,396 | 35.2% -> 74.6% in 10 steps |
| `--rows 3` | 243-266 | 0.80-0.95 | 2,522-2,583 | 34.4% -> 69.6% in 4 steps |

Fuller rows and more rows a step train more tokens per second (15% at three rows), and sampling time is proportional
to the replies asked for (the run feeds its sampler 512 sequences at a time, about 12,500-13,000 generated tokens
a second at any batch size):

| Replies in one `sample` call | Sampling |
|---|---|
| 512 | 17 s |
| 2,048 | 72 s |
| 4,096 | 150 s |

So a large step costs less per trained token: at three rows a GPU a step of ~1,950 replies takes 143-161 s and trains
1.1-1.2M tokens, about $0.99-1.03 per 1M trained tokens at $28 an hour, against about $1.13 for the 64-problem step.
It does not learn more per sampled problem: a larger step is fewer updates for the same samples. Use `--rows` when the
reward is noisy and you want large batches, or to spend less per token; the default suits a first run. A step that is
a few tokens over a multiple of the capacity opens one more row on every GPU, each half empty, which is why the example
aims at 90%.

### Other models

The same loop (`recipes/math_rl/train.py --rows 1 --steps 8 --eval-every 4`, learning rate 2e-6, 8 replies a problem,
512 new tokens) on the two other models that have RL runs:

| Model | Row (tokens a GPU) | Problems a step | MATH-500 at step 0 / 4 / 8 | Trained tokens/s/GPU |
|---|---|---|---|---|
| `muse-glimmer-30b` | 49,152 | 72-94 | 57.8% / 75.2% / 79.4% | 1,860-2,210 |
| `gemma-4-26b-a4b` | 131,072 | 188-193 | 51.2% / 53.8% / 59.4% | 6,660-7,260 |

A `gemma-4-26b-a4b` RL run pads every row to a fixed length (131,072 tokens, or one of a few shorter lengths for a
small step), because a mixture-of-experts model tunes its kernels for each row length it meets. Two things follow:
size steps with `--rows` so the padding is small, and expect the first step at a row length to take 20-50 s longer
than the ones after it. Gemma trains three times as many tokens a second as the dense models and gains less per step
at this learning rate.

## Step size and training throughput

The `math` recipe with more replies a step (`groups_per_batch=256` and `512`, `learning_rate=4e-6`):

| Replies a step | Tokens trained a step | Training | Trained tokens/s/GPU |
|---|---|---|---|
| 64 (`gsm8k`) | 31K | 6 s | 510-700 |
| 512 (`math` as shipped) | about 310K | 17-18 s | 2,240 |
| 2,048 | 0.7-1.2M | 34-59 s | 2,400-2,525 |
| 4,096 | 1.8-2.4M | 87-114 s | 2,600 |

A training step has a fixed cost of a few seconds, so tokens/s rises with the step's size and levels off above about
1M tokens a step. Sampling time grows in proportion to the replies. With 2,048 replies a step MATH-500 went 34.6% ->
59.0% -> 82.0% at steps 0, 3 and 6; the shipped 512 replies reach 82.2% at step 20 with about the same number of
trained tokens. A larger step is about 10% cheaper per trained token and learns the same per token, so the recipes
keep their sizes; above 300K tokens a step you are within 15% of the best rate.

## forward, and a killed client

`forward` on 4 sequences of about 90 sampled tokens: 0.6 s. Its logprobs differ from the sampler's own by at most
0.07-0.12 on a token and from the trainer's by at most 0.11-0.16 (bf16 arithmetic in different batch shapes).

A client killed with `kill -9` at 04:03:42: the run was `closing` by 04:07:04 with `billable_seconds` frozen (the
gateway finishes after 180 s without a heartbeat).

## Sampler and trainer agreement

The engine reports the trainer's probability of each sampled token over the sampler's: 0.9998-1.0006 across these runs.
The cookbook's `optim/kl_sample_train_v2`: 1e-4 to 7e-4.
