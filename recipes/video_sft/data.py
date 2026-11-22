"""Short videos of a coloured square bouncing between two walls, for a managed fine-tune on video.

    brew install ffmpeg                         # or apt install ffmpeg
    python recipes/video_sft/data.py            # writes bounce/: train.jsonl, test.jsonl and the MP4s they name

Each video is 4 to 10 seconds of a square sliding left and right at its own speed; the reply says how many times it
bounces and off which walls, in order. The answer is in the motion, not in any single frame. test.jsonl is held out:
it is never uploaded (a dataset with videos takes no validation set yet), and the README evaluates on it.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

WIDTH, HEIGHT, SIDE, FPS = 320, 240, 28, 8
TRAVEL = WIDTH - SIDE                     # pixels between the two walls, for the square's left edge
COLOURS = ["red", "green", "blue", "yellow", "white", "orange"]
QUESTION = "How many times does the square bounce off a wall, and off which walls, in order?"


def clip(rng):
    """One video's settings: the square's colour, start, speed and direction, and how long the video is. Each count
    of bounces from 0 to 5 is equally likely, so always guessing one answer scores 1 in 6."""
    bounces = rng.randint(0, 5)
    while True:
        seconds = rng.randint(4, 10)
        start, speed = rng.uniform(0, TRAVEL), rng.uniform(40, 160)
        end = start + speed * (seconds - 1 / FPS)   # where the square is in the last frame
        # every bounce at least half a second from the first and the last frame, so the frames show each one
        margin = speed / 2
        if margin < start < TRAVEL - margin and margin < end % TRAVEL < TRAVEL - margin and end // TRAVEL == bounces:
            return {"colour": rng.choice(COLOURS), "start": start, "speed": speed, "seconds": seconds,
                    "leftward": rng.random() < 0.5, "bounces": bounces}


def render(c, path):
    """The square's left edge folds start + speed * t back and forth between 0 and TRAVEL; leftward mirrors it."""
    x = f"{TRAVEL}-abs(mod({c['start']:.2f}+{c['speed']:.2f}*t,{2 * TRAVEL})-{TRAVEL})"
    if c["leftward"]:
        x = f"{TRAVEL}-({x})"
    scene = (f"color=c=0x202020:s={WIDTH}x{HEIGHT}:r={FPS}:d={c['seconds']}[bg];"
             f"color=c={c['colour']}:s={SIDE}x{SIDE}:r={FPS}[sq];"
             f"[bg][sq]overlay=x='{x}':y={(HEIGHT - SIDE) // 2}:shortest=1")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-filter_complex", scene, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)


def reply(c):
    if not c["bounces"]:
        return f"The {c['colour']} square never reaches a wall: it does not bounce."
    walls = ["left" if (i % 2 == 0) == c["leftward"] else "right" for i in range(c["bounces"])]
    times = "once" if c["bounces"] == 1 else "twice" if c["bounces"] == 2 else f"{c['bounces']} times"
    return f"The {c['colour']} square bounces {times}: off the {', then the '.join(walls)} wall."


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default="bounce")
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--test", type=int, default=100)
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for split, count in (("train", args.train), ("test", args.test)):
        with open(out / f"{split}.jsonl", "w") as f:
            for i in range(count):
                c, name = clip(rng), f"clips/{split}_{i:04d}.mp4"
                render(c, out / name)
                user = [{"type": "video", "video": name}, {"type": "text", "text": QUESTION}]
                f.write(json.dumps({"messages": [{"role": "user", "content": user},
                                                 {"role": "assistant", "content": reply(c)}],
                                    "bounces": c["bounces"]}) + "\n")
        print(count, "videos ->", out / f"{split}.jsonl")


if __name__ == "__main__":
    main()
