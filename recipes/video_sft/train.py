"""Fine-tune on video: count a square's bounces and name the walls, every weight trained.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/video_sft/data.py                  # bounce/train.jsonl, bounce/test.jsonl and their MP4s
    python recipes/video_sft/train.py                 # uploads train.jsonl with its videos, prints the steps and the price
    python recipes/video_sft/train.py --yes           # start the job, follow it, print the loss of every step
    python recipes/video_sft/train.py --model gemma-4-26b-a4b --lora --learning-rate 1e-4 --video-frame-tokens 140 --yes

`datasets.upload` packs the JSONL and every video it names into one upload; the server checks and re-encodes the
videos. `--video-fps` (Qwen), `--video-max-frames` and `--video-frame-tokens` choose how training samples each video
(the estimate shows the dataset's tokens at those settings). Then evaluate.py serves the job's model and scores it on
test.jsonl. Videos are opened organisation by organisation: `nextron-cloud models`
lists `video` in a model's `input` when they are open to yours.
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


def batch_for(tokens, samples, gpus, tokens_per_gpu):
    """Samples a step so that each GPU packs about `tokens_per_gpu` tokens; a multiple of 8 and of the GPU count."""
    multiple = math.lcm(8, gpus)
    return max(multiple, round(tokens_per_gpu * gpus * samples / tokens / multiple) * multiple)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="bounce/train.jsonl")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--layout", default="1x8-h100")
    parser.add_argument("--lora", action="store_true", help="train a LoRA adapter (merged into the checkpoint)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--tokens-per-gpu", type=int, default=16384, help="tokens each GPU packs in a step")
    parser.add_argument("--video-fps", type=float, help="frames a second training reads (Qwen: 1, 2, 4, 8, 16; default 4)")
    parser.add_argument("--video-max-frames", type=int, help="frames a video at most (Qwen default 40, Gemma 32)")
    parser.add_argument("--video-frame-tokens", type=int, help="tokens a frame (Qwen 256/512/900, default 512; Gemma 70/140/280, default 70)")
    parser.add_argument("--name", default="video_sft")
    parser.add_argument("--yes", action="store_true", help="start the job (it is charged per trained token)")
    args = parser.parse_args()
    client = NexTron()
    params = {"train_type": "lora" if args.lora else "full", "num_train_epochs": args.epochs, "learning_rate": args.learning_rate}
    if args.video_fps:
        params["video_fps"] = args.video_fps
    if args.video_max_frames:
        params["video_max_frames"] = args.video_max_frames
    if args.video_frame_tokens:
        params["video_frame_tokens"] = args.video_frame_tokens
    train = uploaded(client, args.data, "bounce train")
    layouts = next(m for m in client.models() if m.id == args.model).layouts
    if args.layout not in layouts:   # the batch is sized for the layout's GPUs, so it is named, not "auto"
        raise SystemExit(f"{args.model} trains on {', '.join(layouts)}: pass one with --layout")
    job = dict(model=args.model, dataset_id=train.id, layout=args.layout, params=params)
    # a video's tokens depend on the settings, so the batch is sized from the estimate's count at them
    tokens = client.jobs.estimate(**job).tokens / args.epochs
    params["batch_size"] = batch_for(tokens, train.samples, layouts[args.layout].gpus, args.tokens_per_gpu)
    estimate = client.jobs.estimate(**job)
    print(f"{train.id}: {train.samples:,} samples with videos, {tokens:,.0f} tokens an epoch at {estimate.video}; "
          f"batch_size {params['batch_size']}: {estimate.steps} steps, ${estimate.credits / 100:.2f} held")
    for warning in estimate.warnings:
        print("warning:", warning["text"])
    if not args.yes:
        return
    started = client.jobs.create(**job, name=args.name, tags=["video_sft"])
    print(started.id, "queued")
    for event in client.jobs.events(started.id, follow=True):   # ends when training is over
        if event.type == "training.step":
            print(f"step {event.data['step']:>3} of {event.data['total_steps']}  loss {event.data['loss']:.4f}", flush=True)
    done = client.jobs.wait(started.id)
    if done.state != "succeeded":
        raise SystemExit(f"{started.id} {done.state}: {done.error}")
    points = client.jobs.metrics(started.id)
    speeds = sorted(point.tokens_per_s_per_gpu for point in points[1:])   # the first step compiles kernels
    speed = f"{speeds[len(speeds) // 2]:,} tokens/s/GPU (median)," if speeds else "one step: no steady speed,"
    print(done.state, speed, f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged;",
          f"next: python recipes/video_sft/evaluate.py {started.id}")


if __name__ == "__main__":
    main()
