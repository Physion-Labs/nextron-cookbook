"""Competitive programming with tests as the reward: the model writes a Python solution, your code runs it against
the problem's tests in a sandbox, tells the model what failed, and it gets one more attempt.

Same problems (DeepCoder: Codeforces, TACO, PrimeIntellect, LiveCodeBench), tests and grader as `nextron-cookbook run
code`, taken from tinker-cookbook; the loop is our own. Needs `nextron-cloud`, `nextron-cookbook` and a SandboxFusion
container next to this loop (README.md): model-written code runs there, never in this process.

    export NEXTRON_API_KEY=nk_live_... SANDBOX_URL=http://127.0.0.1:8080/run_code
    python recipes/code_rl/train.py rl_... --eval-every 6     # a ready RL run: 6 steps
    nextron-cloud rl stop rl_... --wait                                 # if the loop did not reach its end

A problem's episodes are compared with each other; an episode is its one or two replies with one reward.
"""
import argparse
import ast
import asyncio
import json
import time

from nextron_cloud import Episode, NexTron
from tinker_cookbook.recipes.code_rl.code_env import load_deepcoder_tasks
from tinker_cookbook.recipes.code_rl.code_grading import extract_code_from_model, sandbox_check_correctness

ATTEMPTS = 2
LOOP = asyncio.new_event_loop()   # the sandbox client keeps one HTTP session, so every check runs on this loop


def verdict(stdout):
    """What the model is told about a failed run: the verdict, the failing input and its own output or error. Not the
    expected output: most problems have few tests, and a program that prints them would pass."""
    try:
        failed = ast.literal_eval(stdout.strip().splitlines()[-1])
    except (ValueError, SyntaxError, IndexError):
        failed = None
    if not isinstance(failed, dict):
        return "The program did not run to the end."
    return "\n".join(f"{key}: {str(failed[key])[:1000]}" for key in ("error_message", "error", "inputs", "output") if key in failed)


def run_tests(tests_and_code):
    """[(passed, what to tell the model)] for each (tests, code); code None is a reply without a code block."""
    async def one(tests, code):
        if code is None:
            return False, "No Python code block found in your reply."
        passed, details = await sandbox_check_correctness(tests, code)
        return passed, verdict((details.get("run_result") or {}).get("stdout") or "")

    async def every():
        return await asyncio.gather(*[one(tests, code) for tests, code in tests_and_code])

    return LOOP.run_until_complete(every())


def play(live, tasks, episodes, max_tokens):
    """`episodes` episodes on each task: [{"group", "samples": its replies in order, "passed", "reward"}]."""
    games = [{"group": index, "task": task, "samples": [], "messages": [{"role": "user", "content": task.problem}]}
             for index, task in enumerate(tasks) for _ in range(episodes)]
    open_games = games
    for attempt in range(ATTEMPTS):
        batch = live.sample([game["messages"] for game in open_games], max_tokens=max_tokens)
        codes = [extract_code_from_model(reply.text or "") for (reply,) in batch]
        results = run_tests([(game["task"].tests, code) for game, code in zip(open_games, codes)])
        still_open = []
        for game, (reply,), code, (passed, told) in zip(open_games, batch, codes, results):
            game["samples"].append(reply)
            game["passed"] = passed
            game["reward"] = 0.1 * ((code is not None) - 1) + passed     # the cookbook's: format and all tests pass
            if not passed and attempt + 1 < ATTEMPTS:
                game["messages"] = [*game["messages"], {"role": "assistant", "content": reply.text or ""},
                                    {"role": "user", "content": f"Your solution failed a test:\n{told}\n"
                                                                "Fix it and reply with the full program in one code block."}]
                still_open.append(game)
        open_games = still_open
    return games


def mean(values):
    return round(sum(values) / len(values), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="the RL run's id")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--problems", type=int, default=32, help="problems per step")
    parser.add_argument("--episodes", type=int, default=8, help="episodes per problem")
    parser.add_argument("--max-tokens", type=int, default=4096, help="new tokens per reply")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--eval-every", type=int, default=0, help="the 687 test problems every this many steps, and at the end")
    parser.add_argument("--log-dir", help="keep metrics.jsonl and rollouts.jsonl (a few groups of every step) here")
    args = parser.parse_args()
    echo = [{"input": "7\n", "output": "7\n", "testtype": "stdin_stdout", "metadata": {}}]
    if not run_tests([(echo, "print(input())")])[0][0]:
        raise SystemExit("the sandbox did not run a one-line program: start SandboxFusion and set SANDBOX_URL (README.md)")
    train = load_deepcoder_tasks("train")
    test = load_deepcoder_tasks("test")
    with NexTron().rl.connect(args.run, log_dir=args.log_dir) as live:
        def evaluate(step):
            started = time.time()
            games = play(live, test, 1, args.max_tokens)
            print(json.dumps({"eval_step": step, "problems": len(games), "all_tests_pass": mean([g["passed"] for g in games]),
                              "reward": mean([g["reward"] for g in games]),
                              "attempts": mean([len(g["samples"]) for g in games]),
                              "seconds": round(time.time() - started)}), flush=True)

        for step in range(args.steps):
            if args.eval_every and step % args.eval_every == 0:
                evaluate(step)
            started = time.time()
            games = play(live, train[step * args.problems:(step + 1) * args.problems], args.episodes, args.max_tokens)
            episodes = [Episode(samples=game["samples"], reward=game["reward"], group=f"{step}-{game['group']}") for game in games]
            result = live.train(episodes=episodes, learning_rate=args.learning_rate)
            print(json.dumps({"step": step, "episodes": len(games), "reward": round(result.reward_mean, 3),
                              "all_tests_pass": mean([g["passed"] for g in games]),
                              "attempts": mean([len(g["samples"]) for g in games]),
                              "flat_groups": round(result.zero_variance_fraction, 2),
                              "truncated": round(result.truncated_fraction or 0, 2),
                              "generation_s": round(result.generation_s, 1), "train_s": round(result.train_s, 1),
                              "client_s": round(result.client_s, 1), "tokens": result.tokens,
                              "tokens_per_s_per_gpu": round(result.tokens_per_s_per_gpu or 0),
                              "rows": result.rows, "row_fill": round(result.row_fill or 0, 2),
                              "kl_sample_train": round(result.kl_sample_train, 5),
                              "step_seconds": round(time.time() - started, 1)}), flush=True)
        if args.eval_every:
            evaluate(args.steps)
        live.finish()   # saves the model and stops the run


if __name__ == "__main__":
    main()
