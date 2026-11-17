"""Reinforcement fine-tuning (RFT, GRPO) on MATH: NexTron runs the whole loop (sampling, the `boxed` reward, training); you only send prompts.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/math_rft/data.py                    # math_numeric.jsonl
    python recipes/math_rft/train.py                   # uploads it, prints the price; starts nothing
    python recipes/math_rft/train.py --yes             # start the job and print each round's reward

A reply is cut at `--max-completion-length` tokens and a cut reply scores 0, so the prompt asks for brief reasoning and
the limit is 1,024: the default of 8,192 lets a round cost several times as much for the same reward.
"""
import argparse
import hashlib
from pathlib import Path

from nextron_cloud import NexTron


def uploaded(client, path, name):
    """The file as a dataset on the service: uploaded once, found again by its name, which ends in the file's hash."""
    name = f"{name} {hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]}"
    for dataset in client.datasets.list():
        if dataset.name == name and dataset.state == "ready":
            return dataset
    return client.datasets.wait(client.datasets.upload(path, name=name).id)

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
    dataset = uploaded(client, args.data, "math numeric (rft)")
    job = dict(model=args.model, dataset_id=dataset.id,
               params={"train_type": "grpo", "reward": "boxed", "max_steps": args.rounds, "batch_size": args.prompts,
                       "num_generations": args.replies, "max_completion_length": args.max_completion_length,
                       "learning_rate": args.learning_rate})
    estimate = client.jobs.estimate(**job)
    print(f"{dataset.id}: {dataset.samples:,} prompts; {args.rounds} rounds of {args.prompts} x {args.replies} replies: "
          f"at most ${estimate.credits / 100:.2f} (held); the charge is for the tokens really generated")
    if not args.yes:
        return
    started = client.jobs.create(**job, name="math_rft", tags=["math_rft"])
    print(started.id, "queued")
    done = client.jobs.wait(started.id)
    if done.state != "succeeded":
        raise SystemExit(f"{started.id} {done.state}: {done.error}")
    for point in client.jobs.metrics(started.id):
        whole = f" of {point.round_time_s:.0f} s" if point.round_time_s is not None else ""   # the first round has none
        print(f"round {point.step:>3}  reward {point.reward_mean:.3f}  flat groups {point.zero_variance_fraction:.2f}  "
              f"reply {point.completion_length_mean} tokens  sampling {point.generation_s:.0f} s{whole}")
    print(done.state, f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged")


if __name__ == "__main__":
    main()
