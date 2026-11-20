"""The smallest managed fine-tune: a sample dataset that is already on the service, 20 short steps, validation loss.

    export NEXTRON_API_KEY=nk_live_...
    python recipes/sft_basic.py            # the price and the steps; starts nothing
    python recipes/sft_basic.py --yes      # start the job and follow it

You send a dataset id and a few settings; NexCloud runs the whole training job and keeps the checkpoint for you.
"""
import argparse

from nextron_cloud import NexTron

TRAIN, VALIDATION = "OpenThoughts Math (subset)", "OpenThoughts Math (validation subset)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="gemma-4-26b-a4b")
    parser.add_argument("--yes", action="store_true", help="start the job (it is charged per trained token)")
    args = parser.parse_args()
    client = NexTron()
    samples = {d.name: d.id for d in client.datasets.list() if d.sample}
    # batch_size 16 is the smallest step this data allows (each GPU needs 8K tokens): many cheap steps for a first look.
    job = dict(model=args.model, dataset_id=samples[TRAIN], layout="1x8-h100",
               params={"max_steps": 20, "batch_size": 16, "learning_rate": 1e-5,
                       "val_dataset_id": samples[VALIDATION], "eval_steps": 5})
    estimate = client.jobs.estimate(**job)
    print(f"{estimate.steps} steps, about {estimate.tokens:,} tokens, ${estimate.credits / 100:.2f} held")
    if not args.yes:
        return
    started = client.jobs.create(**job, name="sft_basic")
    print(started.id, "queued; training starts about 2 minutes after a GPU setup is free")
    done = client.jobs.wait(started.id)
    if done.state != "succeeded":
        raise SystemExit(f"{started.id} {done.state}: {done.error}")
    for point in client.jobs.metrics(started.id):
        val = f"  val_loss {point.val_loss:.4f}" if point.val_loss is not None else ""
        print(f"step {point.step:>3}  loss {point.loss:.4f}  {point.tokens_per_s_per_gpu:,} tokens/s/GPU{val}")
    print(done.state, f"${client.jobs.billing(started.id).charged_credits / 100:.2f} charged;",
          f"the checkpoint uploads now: nextron-cloud jobs checkpoint {started.id}")


if __name__ == "__main__":
    main()
