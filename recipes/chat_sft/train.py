"""Chat fine-tuning on NoRobots (9,500 human-written conversations): every weight trained, steps sized for the trainer.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/chat_sft/data.py                  # no_robots_train.jsonl, no_robots_val.jsonl
    python recipes/chat_sft/train.py                 # uploads them, prints the steps and the price; starts nothing
    python recipes/chat_sft/train.py --yes           # start the job, follow it, print the validation loss

NexTron packs a step by tokens, not by samples: without `batch_size` each GPU takes about 64K tokens, which is 5 steps
an epoch on this data. This script asks for the batch that gives each GPU `--tokens-per-gpu` tokens instead.
"""
import argparse
import hashlib
import math
from pathlib import Path

from nextron_cloud import NexTron


def uploaded(client, path, name):
    """The file as a dataset on the service: uploaded once, found again by its name, which ends in the file's hash."""
    name = f"{name} {hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]}"
    for dataset in client.datasets.list():
        if dataset.name == name and dataset.state == "ready":
            return dataset
    return client.datasets.wait(client.datasets.upload(path, name=name).id)


def batch_for(dataset, model, gpus, tokens_per_gpu):
    """Samples a step so that each GPU packs about `tokens_per_gpu` tokens; a multiple of 8 and of the GPU count."""
    average = dataset.tokens[model]["total"] / dataset.samples
    multiple = math.lcm(8, gpus)
    return max(multiple, round(tokens_per_gpu * gpus / average / multiple) * multiple)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--layout", default="1x8-h100")
    parser.add_argument("--train-type", default="full", choices=["full", "lora"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--tokens-per-gpu", type=int, default=16384, help="tokens each GPU packs in a step")
    parser.add_argument("--name", default="chat_sft")
    parser.add_argument("--yes", action="store_true", help="start the job (it is charged per trained token)")
    args = parser.parse_args()
    client = NexTron()
    train = uploaded(client, "no_robots_train.jsonl", "no_robots train")
    validation = uploaded(client, "no_robots_val.jsonl", "no_robots val")
    layouts = next(m for m in client.models() if m.id == args.model).layouts
    if args.layout not in layouts:   # the batch is sized for the layout's GPUs, so it is named, not "auto"
        raise SystemExit(f"{args.model} trains on {', '.join(layouts)}: pass one with --layout")
    gpus = layouts[args.layout].gpus
    batch = batch_for(train, args.model, gpus, args.tokens_per_gpu)
    job = dict(model=args.model, dataset_id=train.id, layout=args.layout,
               params={"train_type": args.train_type, "num_train_epochs": args.epochs, "batch_size": batch,
                       "learning_rate": args.learning_rate, "val_dataset_id": validation.id})
    estimate = client.jobs.estimate(**job)
    print(f"{train.id}: {train.samples:,} samples, {train.tokens[args.model]['total']:,} tokens; batch_size {batch}: "
          f"{estimate.steps} steps, ${estimate.credits / 100:.2f} held")
    for warning in estimate.warnings:
        print("warning:", warning["text"])
    if not args.yes:
        return
    started = client.jobs.create(**job, name=args.name, tags=["chat_sft"])
    print(started.id, "queued")
    for event in client.jobs.events(started.id, follow=True):   # ends when training is over
        if event.type == "training.validation":
            total = f" of {estimate.steps}" if estimate.steps else ""   # a model with a tuned pack plans no step count
            print(f"step {event.data['step']:>4}{total}  val_loss {event.data['val_loss']:.4f}", flush=True)
    done = client.jobs.wait(started.id)
    if done.state != "succeeded":
        raise SystemExit(f"{started.id} {done.state}: {done.error}")
    points = client.jobs.metrics(started.id)
    speeds = sorted(point.tokens_per_s_per_gpu for point in points[1:])   # the first step compiles kernels
    speed = f"{speeds[len(speeds) // 2]:,} tokens/s/GPU (median)," if speeds else "one step: no steady speed,"
    print(done.state, speed, f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged")


if __name__ == "__main__":
    main()
