"""Serve a trained job's model on NexCloud and score it on the held-out bounce videos.

    python recipes/video_sft/evaluate.py job_...          # inference run, 100 prompts, scores; stops the run

The run reads each video with the frame settings the job trained with. An inference run is billed by the hour from
ready to stopped (a minute or two here). The test videos are checked first, as one dataset of your organisation,
before the run starts, so the run does not wait for them while billed.
"""
import argparse
import json
import re
from pathlib import Path

from nextron_cloud import NexTron

COUNTS = {"once": 1, "twice": 2}


def bounces(reply: str) -> tuple[int | None, list[str]]:
    """The count and the walls, in order, that a reply names (the training replies' wording)."""
    walls = re.findall(r"\b(left|right)\b", reply)
    if "does not bounce" in reply or "never reaches" in reply:
        return 0, walls
    word = re.search(r"bounces (once|twice|(\d+) times)", reply)
    if not word:
        return None, walls
    return COUNTS.get(word.group(1)) or int(word.group(2)), walls


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("job")
    parser.add_argument("--test", default="bounce/test.jsonl")
    args = parser.parse_args()
    client = NexTron()
    rows = [json.loads(line) for line in open(args.test)]
    root = Path(args.test).parent
    prompts = []
    for row in rows:
        video, question = row["messages"][0]["content"]
        prompts.append([{"role": "user", "content": [{"type": "video", "video": str(root / video["video"])}, question]}])
    paths = [prompt[0]["content"][0]["video"] for prompt in prompts]
    media = client.datasets.upload_images(paths, name="bounce test videos")
    run = client.inference.start(job_id=args.job, max_hours=1, images=True)
    print(run.id, "loading the model")
    try:
        client.inference.wait(run.id)
        with client.inference.connect(run.id) as live:
            live.add_images(paths, dataset_id=media.id)
            batch = live.sample(prompts, n=1, max_tokens=120, temperature=0)
    finally:
        client.inference.stop(run.id)
        client.datasets.delete(media.id)
    count = sequence = 0
    for row, group in zip(rows, batch):
        got, walls = bounces(group[0].text)
        want, want_walls = bounces(row["messages"][1]["content"])
        count += got == want
        sequence += got == want and walls == want_walls
    print(f"bounce count right: {count}/{len(rows)}; count and walls in order: {sequence}/{len(rows)}")


if __name__ == "__main__":
    main()
