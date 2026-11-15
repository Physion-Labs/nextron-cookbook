"""The `math` recipe's task with NexTron's own calls instead of the tinker SDK: sample a batch, score it here, train.

Same problems (Hendrycks MATH), prompt and reward as `nextron-cookbook run math`, taken from tinker-cookbook's
environment, so the two can be compared. Needs `nextron-cloud` and `nextron-cookbook[math]`.

    export NEXTRON_API_KEY=nk_live_...
    python examples/math_loop.py rl_... --steps 10 --eval-every 10       # a ready RL run
    python examples/math_loop.py rl_... --steps 6 --rows 2 --eval-every 6  # steps sized to fill 2 trainer rows a GPU
    nextron-cloud rl stop rl_... --wait                                  # if the loop did not reach its end

Everything in this file runs on your machine. The RL run samples, keeps the sampled tokens, and trains.
"""
import argparse
import json
import time

from nextron_cloud import NexTron
from tinker_cookbook.recipes.math_rl.math_env import (MathEnv, _get_hendrycks_math_test, _get_hendrycks_math_train,
                                                      extract_boxed, safe_grade)

PREFIX = MathEnv.standard_fewshot_prefix()


def prompt(problem):
    return [*PREFIX, {"role": "user", "content": problem + MathEnv.question_suffix()}]


def score(reply, answer):
    """(reward, correct, format): the cookbook's ProblemEnv reward, 0.1 * (format - 1) + correct."""
    try:
        given = extract_boxed(reply.text)
    except ValueError:
        return -0.1, 0.0, 0.0
    fmt = 1.0 if reply.finish_reason == "stop" else 0.0
    correct = 1.0 if safe_grade(given, answer) else 0.0
    return 0.1 * (fmt - 1) + correct, correct, fmt


def with_answers(rows):
    out = []
    for row in rows:
        try:
            out.append((row["problem"], extract_boxed(row["solution"])))
        except ValueError:
            pass
    return out


def mean(values):
    return round(sum(values) / len(values), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="the RL run's id")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--problems", type=int, default=64, help="problems per step (the first step's, with --rows)")
    parser.add_argument("--rows", type=int, default=0, help="size every step to fill this many trainer rows on each GPU")
    parser.add_argument("--replies", type=int, default=8, help="replies per problem")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--eval-every", type=int, default=0, help="MATH-500 every this many steps, and at the end")
    parser.add_argument("--log-dir", help="keep metrics.jsonl and rollouts.jsonl (a few groups of every step) here")
    args = parser.parse_args()
    train = _get_hendrycks_math_train().shuffle(seed=0)
    test = with_answers(_get_hendrycks_math_test())
    with NexTron().rl.connect(args.run, log_dir=args.log_dir) as live:
        def evaluate(step):
            batch = live.sample([prompt(p) for p, _ in test], max_tokens=args.max_tokens)
            scored = [score(group[0], answer) for (_, answer), group in zip(test, batch)]
            print(json.dumps({"eval_step": step, "test_correct": mean([s[1] for s in scored]),
                              "test_format": mean([s[2] for s in scored]), "seconds": batch.seconds}), flush=True)

        count, taken = args.problems, 0
        for step in range(args.steps):
            if args.eval_every and step % args.eval_every == 0:
                evaluate(step)
            started = time.time()
            problems = with_answers(train.select(range(taken, taken + count)))
            taken += count
            batch = live.sample([prompt(p) for p, _ in problems], n=args.replies, max_tokens=args.max_tokens)
            scored = [[score(reply, answer) for reply in group] for (_, answer), group in zip(problems, batch)]
            result = live.train(batch, [[s[0] for s in group] for group in scored], learning_rate=args.learning_rate)
            flat = [s for group in scored for s in group]
            print(json.dumps({"step": step, "problems": len(problems), "reward": mean([s[0] for s in flat]),
                              "correct": mean([s[1] for s in flat]), "flat_groups": round(result.zero_variance_fraction, 2),
                              "truncated": round(result.truncated_fraction or 0, 2),
                              "generation_s": round(result.generation_s, 1), "train_s": round(result.train_s, 1),
                              "client_s": round(result.client_s, 1), "tokens": result.tokens,
                              "tokens_per_s_per_gpu": round(result.tokens_per_s_per_gpu or 0),
                              "rows": result.rows, "row_fill": round(result.row_fill or 0, 2),
                              "kl_sample_train": round(result.kl_sample_train, 5),
                              "step_seconds": round(time.time() - started, 1)}), flush=True)
            if args.rows and result.row_tokens:
                # The trainer packs a step into rows of row_tokens tokens on each GPU, and a row costs a few seconds
                # however full it is: ask for as many problems as fill the rows. Reply lengths move from step to
                # step and one token too many opens another row on every GPU, so aim at 90%.
                per_problem = result.tokens / len(problems)
                count = max(args.replies, int(0.9 * result.gpus * args.rows * result.row_tokens / per_problem))
        if args.eval_every:
            evaluate(args.steps)
        live.finish()   # saves the model and stops the run


if __name__ == "__main__":
    main()
