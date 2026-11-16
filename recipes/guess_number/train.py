"""Guess the number: a multi-turn task. The model has 10 guesses at a hidden integer between 0 and 1024; your code
answers each guess with "Too low" or "Too high"; an episode scores 1 when it finds the number.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/guess_number/train.py rl_... --eval-every 5     # a ready RL run
    nextron-cloud rl stop rl_... --wait                                       # if the loop did not reach its end

Every turn is one `sample` call for all the games still open, and a game is one episode: its replies in turn order
with one reward, compared with the other games on the same hidden number.
"""
import argparse
import json
import random
import re

from nextron_cloud import Episode, NexTron

UPPER, TURNS = 1024, 10
SYSTEM = (f"Your job is to guess an integer between 0 and {UPPER}. Output your guess in the format Guess: <guess> "
          "(without the angle brackets) and say nothing else. After each guess you are told whether it is too low or "
          f"too high. You have {TURNS} guesses.")


def play(live, numbers, games):
    """`games` games on each hidden number: [(number index, the game's replies in turn order, its reward, guesses)]."""
    open_games = [{"group": index, "number": number, "samples": [], "reward": 0.0,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Make your first guess."}]}
                  for index, number in enumerate(numbers) for _ in range(games)]
    finished = []
    for _ in range(TURNS):
        batch = live.sample([game["messages"] for game in open_games], max_tokens=16)
        still_open = []
        for game, (reply,) in zip(open_games, batch):
            game["samples"].append(reply)
            guess = re.fullmatch(r"Guess: (\d+)", (reply.text or "").strip())
            if guess is None:
                game["reward"] = -1.0          # not in the format: the game ends
            elif int(guess.group(1)) == game["number"]:
                game["reward"] = 1.0
            else:
                hint = "Too low." if int(guess.group(1)) < game["number"] else "Too high."
                game["messages"] = [*game["messages"], {"role": "assistant", "content": reply.text},
                                    {"role": "user", "content": hint}]
                still_open.append(game)
                continue
            finished.append(game)
        open_games = still_open
        if not open_games:
            break
    return finished + open_games


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="the RL run's id")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--numbers", type=int, default=32, help="hidden numbers per step (the first step's, with --rows)")
    parser.add_argument("--games", type=int, default=8, help="games per hidden number")
    parser.add_argument("--rows", type=int, default=1, help="size every step to fill this many trainer rows on each GPU; 0: keep --numbers")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--eval-every", type=int, default=0, help="128 fixed numbers, one game each, every this many steps and at the end")
    args = parser.parse_args()
    fixed = random.Random(0).sample(range(UPPER + 1), 128)
    draw = random.Random(1)
    with NexTron().rl.connect(args.run) as live:
        def evaluate(step):
            games = play(live, fixed, 1)
            solved = [game for game in games if game["reward"] == 1.0]
            print(json.dumps({"eval_step": step, "solved": round(len(solved) / len(games), 3),
                              "guesses_when_solved": round(sum(len(g["samples"]) for g in solved) / max(1, len(solved)), 2),
                              "bad_format": round(sum(g["reward"] < 0 for g in games) / len(games), 3)}), flush=True)

        count = args.numbers
        for step in range(args.steps):
            if args.eval_every and step % args.eval_every == 0:
                evaluate(step)
            games = play(live, [draw.randint(0, UPPER) for _ in range(count)], args.games)
            episodes = [Episode(samples=game["samples"], reward=game["reward"], group=f"{step}-{game['group']}") for game in games]
            result = live.train(episodes=episodes, learning_rate=args.learning_rate)
            print(json.dumps({"step": step, "numbers": count, "games": len(games), "reward": round(result.reward_mean, 3),
                              "flat_groups": round(result.zero_variance_fraction, 2),
                              "generation_s": round(result.generation_s, 1), "train_s": round(result.train_s, 1),
                              "client_s": round(result.client_s, 1), "tokens": result.tokens,
                              "tokens_per_s_per_gpu": round(result.tokens_per_s_per_gpu or 0),
                              "rows": result.rows, "row_fill": round(result.row_fill or 0, 2)}), flush=True)
            if args.rows and result.row_tokens:
                # A game's turns are separate training sequences that each hold the turns before them, so a game is
                # far more tokens than its replies: size the next step from what this one really packed, aiming at 90%.
                per_number = result.tokens / count
                count = max(8, int(0.9 * result.gpus * args.rows * result.row_tokens / per_number))
        if args.eval_every:
            evaluate(args.steps)
        live.finish()   # saves the model and stops the run


if __name__ == "__main__":
    main()
