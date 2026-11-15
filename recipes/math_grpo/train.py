"""Managed GRPO on MATH: NexTron runs the whole loop (sampling, the `boxed` reward, training); you only send prompts.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/math_grpo/data.py                    # math_numeric.jsonl
    python recipes/math_grpo/train.py                   # uploads it, prints the price; starts nothing
    python recipes/math_grpo/train.py --yes             # start the job and print each round's reward

A reply is cut at `--max-completion-length` tokens and a cut reply scores 0, so the prompt asks for brief reasoning and
the limit is 1,024: the default of 8,192 lets a round cost several times as much for the same reward.
"""
import argparse

from nextron_cloud import NexTron


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--data", default="math_numeric.jsonl")
    parser.add_argument("--rounds", type=int, default=10, help="one optimizer step each")
    parser.add_argument("--prompts", type=int, default=64, help="prompts a round")
    parser.add_argument("--replies", type=int, default=8, help="replies a prompt")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--yes", action="store_true", help="start the job (it is charged per token)")
    args = parser.parse_args()
    client = NexTron()
    name = "math numeric (grpo)"
    ready = [d for d in client.datasets.list() if d.name == name and d.state == "ready"]
    dataset = ready[0] if ready else client.datasets.wait(client.datasets.upload(args.data, name=name).id)
    job = dict(model=args.model, dataset_id=dataset.id,
               params={"train_type": "grpo", "reward": "boxed", "max_steps": args.rounds, "batch_size": args.prompts,
                       "num_generations": args.replies, "max_completion_length": args.max_completion_length,
                       "learning_rate": args.learning_rate})
    estimate = client.jobs.estimate(**job)
    print(f"{dataset.samples:,} prompts; {args.rounds} rounds of {args.prompts} x {args.replies} replies: "
          f"at most ${estimate.credits / 100:.2f} (held); the charge is for the tokens really generated")
    if not args.yes:
        return
    started = client.jobs.create(**job, name="math_grpo", tags=["math_grpo"])
    print(started.id, "queued")
    done = client.jobs.wait(started.id)
    for point in client.jobs.metrics(started.id):
        print(f"round {point.step:>3}  reward {point.reward_mean:.3f}  flat groups {point.zero_variance_fraction:.2f}  "
              f"reply {point.completion_length_mean} tokens  sampling {point.generation_s:.0f} s of {point.round_time_s:.0f} s")
    print(done.state, f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged")


if __name__ == "__main__":
    main()
