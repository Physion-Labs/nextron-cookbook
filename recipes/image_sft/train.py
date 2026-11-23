"""Fine-tune on images: read analog clocks (full or LoRA).

    export NEXTRON_API_KEY=nk_live_...
    python recipes/image_sft/data.py                          # clocks/train.jsonl, clocks/test.jsonl and their PNGs
    python recipes/image_sft/train.py                         # uploads train.jsonl with its images, prints the steps and the price
    python recipes/image_sft/train.py --yes                   # start the job, follow it, print the loss of every step
    python recipes/image_sft/train.py --model qwen3.5-9b --lora --yes

`datasets.upload` packs the JSONL and every image it names into one upload; the server checks each image. Images are
opened organisation by organisation: `nextron-cloud models` lists `image` in a model's `input` when they are open to
yours. Then evaluate.py serves the job's model and scores it on test.jsonl.
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
    parser.add_argument("--data", default="clocks/train.jsonl")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--lora", action="store_true", help="train a LoRA adapter (merged into the checkpoint)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, help="default 2e-5 full, 2e-4 LoRA")
    # 6,000 short samples: the default step packs ~512K tokens (3 steps an epoch); 264 is the smallest batch the
    # service takes for this data on 8 GPUs (8K tokens a GPU), 23 steps an epoch
    parser.add_argument("--batch-size", type=int, default=264)
    parser.add_argument("--name", default="image_sft")
    parser.add_argument("--yes", action="store_true", help="start the job (it is charged per trained token)")
    args = parser.parse_args()
    client = NexTron()
    learning_rate = args.learning_rate or (2e-4 if args.lora else 2e-5)
    params = {"train_type": "lora" if args.lora else "full", "num_train_epochs": args.epochs,
              "learning_rate": learning_rate, "batch_size": args.batch_size}
    train = uploaded(client, args.data, "clocks train")
    job = dict(model=args.model, dataset_id=train.id, params=params)
    estimate = client.jobs.estimate(**job)
    print(f"{train.id}: {train.samples:,} samples, {estimate.tokens / args.epochs:,.0f} tokens an epoch on {args.model}; "
          f"{params['train_type']} lr {learning_rate:g}, batch {args.batch_size}: {estimate.steps} steps, "
          f"${estimate.credits / 100:.2f} held")
    for warning in estimate.warnings:
        print("warning:", warning["text"])
    if not args.yes:
        return
    started = client.jobs.create(**job, name=f"{args.name} {args.model} {params['train_type']}", tags=["image_sft"])
    print(started.id, "queued")
    for event in client.jobs.events(started.id, follow=True):   # ends when training is over
        if event.type == "training.step":
            print(f"step {event.data['step']:>3} of {event.data['total_steps']}  loss {event.data['loss']:.4f}", flush=True)
    done = client.jobs.wait(started.id)
    if done.state != "succeeded":
        raise SystemExit(f"{started.id} {done.state}: {done.error}")
    print(done.state, f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged;",
          f"next: python recipes/image_sft/evaluate.py {started.id}")


if __name__ == "__main__":
    main()
