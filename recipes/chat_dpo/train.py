"""DPO on UltraFeedback preference pairs: the model learns to prefer the chosen reply over the rejected one.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/chat_dpo/data.py                  # ultrafeedback_pairs.jsonl
    python recipes/chat_dpo/train.py                 # uploads it, prints the steps and the price; starts nothing
    python recipes/chat_dpo/train.py --yes           # start the job, follow it, print the loss
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
    parser.add_argument("--model", default="gemma-4-26b-a4b")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, help="the model's default if not given")
    parser.add_argument("--yes", action="store_true", help="start the job (it is charged per trained token)")
    args = parser.parse_args()
    client = NexTron()
    pairs = uploaded(client, "ultrafeedback_pairs.jsonl", "ultrafeedback pairs")
    params = {"train_type": "dpo", "num_train_epochs": args.epochs, "beta": args.beta}
    if args.learning_rate:
        params["learning_rate"] = args.learning_rate
    estimate = client.jobs.estimate(model=args.model, dataset_id=pairs.id, params=params)
    print(f"{pairs.id}: {pairs.samples:,} pairs, {pairs.tokens[args.model]['total']:,} tokens: "
          f"{estimate.steps} steps, ${estimate.credits / 100:.2f} held")
    if not args.yes:
        return
    started = client.jobs.create(model=args.model, dataset_id=pairs.id, params=params, name="chat_dpo", tags=["chat_dpo"])
    print(started.id, "queued")
    done = client.jobs.wait(started.id)
    if done.state != "succeeded":
        raise SystemExit(f"{started.id} {done.state}: {done.error}")
    for point in client.jobs.metrics(started.id):
        print(f"step {point.step:>3} of {point.total_steps}  loss {point.loss:.4f}")
    print(done.state, f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged")


if __name__ == "__main__":
    main()
