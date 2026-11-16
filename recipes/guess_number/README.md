# Guess the number: a multi-turn task (Serverless RL)

The model has 10 guesses at a hidden integer between 0 and 1024. Your code answers each guess with "Too low." or
"Too high."; a game scores 1 when the model finds the number, -1 when a reply is not in the format `Guess: <n>`, and 0
otherwise. Ten guesses are just enough for a binary search, so the model has to search well.

```bash
pip install nextron-cloud
export NEXTRON_API_KEY=nk_live_...
nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes
nextron-cloud rl wait rl_...
python recipes/guess_number/train.py rl_... --eval-every 5     # 5 steps; the same 128 games before and after
nextron-cloud rl stop rl_... --wait
```

## How it maps to the API

- Every turn is one `sample` call for all the games still open: their conversations so far, one reply each.
- A game is one episode: `Episode(samples=[its replies in turn order], reward, group=...)`, compared with the other
  games on the same hidden number (8 by default).
- `--rows 1` sizes the next step from the tokens this one packed: a game's turns each hold the turns before them, so a
  game is many more training tokens than its replies.

## Results

Qwen3.8-27B, one node of 8 H100, learning rate 2e-6:

| | |
|---|---|
| evaluation games solved before training | 75.8% (8.9 guesses when solved; 18.8% end on a malformed reply) |
| mean reward of the training games, step 1 / 2 / 3 / 4 | 0.680 / 0.976 / 0.994 / 0.994 |
| a step at full rows (~1,550 games, ~430K tokens) | 55 s sampling (10 turns), 22 s training at 2,500 tokens/s/GPU, 57 s in the loop's own turns |

The task is learned in one step; after that almost every group scores the same and teaches nothing, so 5 steps is
enough. The four steps cost $6.42 (826 s at $28 an hour, the model's load not billed). For a harder version raise the
upper bound or lower the number of guesses.
