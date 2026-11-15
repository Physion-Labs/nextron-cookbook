"""The smallest Serverless RL loop: sample a batch, score it in your own code, train one step, repeat.

    export NEXTRON_API_KEY=nk_live_...
    nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --yes     # prints the run's id; billed from ready
    nextron-cloud rl wait rl_...
    python recipes/rl_basic.py rl_...                                  # 5 steps, about 4 minutes; stops the run
    nextron-cloud rl stop rl_... --wait                                # always, in case the loop stopped early

The task is arithmetic with a 0/1 reward. Everything in this file runs on your machine; the RL run samples and trains.
"""
import random
import re
import sys

from nextron_cloud import NexTron

PROMPT = "What is {a} * {b} + {c}? Think briefly, then give the result as \\boxed{{number}}."


def reward(reply, answer):
    found = re.findall(r"\\boxed\{(-?\d+)\}", reply)
    return 1.0 if found and int(found[-1]) == answer else 0.0


def main():
    with NexTron().rl.connect(sys.argv[1]) as live:
        for step in range(5):
            problems = [(random.randint(11, 99), random.randint(11, 99), random.randint(1, 999)) for _ in range(64)]
            # One request for the whole batch: 64 prompts, 8 replies each. The run keeps the sampled tokens.
            batch = live.sample([PROMPT.format(a=a, b=b, c=c) for a, b, c in problems], n=8, max_tokens=256)
            rewards = [[reward(reply.text or "", a * b + c) for reply in group] for (a, b, c), group in zip(problems, batch)]
            # A reply's advantage is its reward minus the mean of the replies to the same prompt.
            result = live.train(batch, rewards, learning_rate=2e-6)
            print(f"step {step}: reward {result.reward_mean:.3f}, {result.zero_variance_fraction:.0%} of the groups flat; "
                  f"sampling {result.generation_s:.0f} s, training {result.train_s:.0f} s "
                  f"({result.tokens_per_s_per_gpu or 0:.0f} tokens/s/GPU), this script {result.client_s:.0f} s")
        live.finish()   # saves the model and stops the run


if __name__ == "__main__":
    main()
