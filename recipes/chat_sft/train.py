"""Chat fine-tuning on NoRobots (9,500 human-written conversations): every weight trained, steps sized for the trainer.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/chat_sft/data.py                  # no_robots_train.jsonl, no_robots_val.jsonl
    python recipes/chat_sft/train.py                 # uploads them, prints the steps and the price; starts nothing
    python recipes/chat_sft/train.py --yes           # start the job, follow it, print the validation loss

NexTron packs a step by tokens, not by samples: without `batch_size` each GPU takes about 64K tokens, which is 5 steps
an epoch on this data. This script asks for the batch that gives each GPU `--tokens-per-gpu` tokens instead.
"""
import argparse
import math

from nextron_cloud import NexTron

def uploaded(client, path, name):
    """The file as a dataset on the service: uploaded once, found by name afterwards."""
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
    gpus = next(m for m in client.models() if m.id == args.model).layouts[args.layout].gpus
    batch = batch_for(train, args.model, gpus, args.tokens_per_gpu)
    job = dict(model=args.model, dataset_id=train.id, layout=args.layout,
               params={"train_type": args.train_type, "num_train_epochs": args.epochs, "batch_size": batch,
                       "learning_rate": args.learning_rate, "val_dataset_id": validation.id})
    estimate = client.jobs.estimate(**job)
    print(f"{train.samples:,} samples, {train.tokens[args.model]['total']:,} tokens; batch_size {batch}: "
          f"{estimate.steps} steps, ${estimate.credits / 100:.2f} held")
    for warning in estimate.warnings:
        print("warning:", warning["text"])
    if not args.yes:
        return
    started = client.jobs.create(**job, name=args.name, tags=["chat_sft"])
    print(started.id, "queued")
    done = client.jobs.wait(started.id)
    points = client.jobs.metrics(started.id)
    for point in points:
        if point.val_loss is not None:
            print(f"step {point.step:>4} of {point.total_steps}  loss {point.loss:.4f}  val_loss {point.val_loss:.4f}")
    speeds = sorted(point.tokens_per_s_per_gpu for point in points[1:])   # the first step compiles kernels
    print(done.state, f"{speeds[len(speeds) // 2]:,} tokens/s/GPU (median),",
          f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged")


if __name__ == "__main__":
    main()
