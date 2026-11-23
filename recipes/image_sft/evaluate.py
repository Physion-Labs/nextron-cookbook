"""Serve a trained job's model on NexCloud and score it on the held-out clocks.

    python recipes/image_sft/evaluate.py job_...              # inference run, 300 prompts, scores; stops the run
    python recipes/image_sft/evaluate.py job_... --base       # for a near-untrained job: room to reason, last answer scored

An inference run is billed by the hour from ready to stopped (a minute or two here). The test images are checked first,
as one dataset of your organisation, before the run starts, so the run does not wait for them while billed.
"""
import argparse
import json
import re
from pathlib import Path

from nextron_cloud import NexTron


def prompt(row: dict, root: Path) -> list:
    content = []
    for part in row["messages"][0]["content"]:
        content.append({"type": "image", "image": str(root / part["image"])} if part["type"] == "image" else part)
    return [{"role": "user", "content": content}]


def score(row: dict, reply: str) -> dict:
    """Read: the last H:MM in the reply against the answer, exact and within a minute. Compare: the last A or B."""
    gold = row["messages"][1]["content"]
    if row["kind"] == "compare":
        letters = re.findall(r"\b([AB])\b", reply)
        return {"kind": "compare", "exact": bool(letters) and letters[-1] == gold}
    times = re.findall(r"(\d{1,2}):(\d{2})", reply)
    hours, minutes = map(int, gold.split(":"))
    if not times:
        return {"kind": "read", "exact": False, "within1": False}
    off = abs((int(times[-1][0]) % 12 * 60 + int(times[-1][1])) - (hours % 12 * 60 + minutes))
    off = min(off, 720 - off)
    return {"kind": "read", "exact": off == 0, "within1": off <= 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("job")
    parser.add_argument("--test", default="clocks/test.jsonl")
    parser.add_argument("--base", action="store_true", help="512 reply tokens instead of 16: a base model reasons first")
    args = parser.parse_args()
    client = NexTron()
    rows = [json.loads(line) for line in open(args.test)]
    prompts = [prompt(row, Path(args.test).parent) for row in rows]
    paths = sorted({part["image"] for p in prompts for part in p[0]["content"] if part["type"] == "image"})
    media = client.datasets.upload_images(paths, name="clocks test images")
    run = client.inference.start(job_id=args.job, max_hours=1, images=True)
    print(run.id, "loading the model")
    try:
        client.inference.wait(run.id)
        with client.inference.connect(run.id) as live:
            live.add_images(paths, dataset_id=media.id)
            batch = live.sample(prompts, n=1, max_tokens=512 if args.base else 16, temperature=0)
    finally:
        client.inference.stop(run.id)
        client.datasets.delete(media.id)
    scores = [score(row, group[0].text) for row, group in zip(rows, batch)]
    read = [s for s in scores if s["kind"] == "read"]
    compare = [s for s in scores if s["kind"] == "compare"]
    print(f"read the time: {sum(s['exact'] for s in read)}/{len(read)} exact, "
          f"{sum(s['within1'] for s in read)}/{len(read)} within a minute; "
          f"which is later: {sum(s['exact'] for s in compare)}/{len(compare)}")


if __name__ == "__main__":
    main()
