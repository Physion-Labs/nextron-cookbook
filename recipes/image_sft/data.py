"""Analog clocks to read, for a managed fine-tune on images.

    python recipes/image_sft/data.py            # writes clocks/: train.jsonl (6,000), test.jsonl (300) and the PNGs

Each sample shows one clock and asks for the time as H:MM; one in five shows two clocks and asks which is later (two
images in one message). Faces, colours, numerals, hands and a small tilt vary. test.jsonl is held out: it is never
uploaded (a dataset with images takes no validation set yet), and evaluate.py scores the trained model on it.
The README's results were measured on data drawn with Arial; another font changes the pictures a little.
"""
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

READ = "What time does this clock show? Answer with the time only, as H:MM (12-hour clock)."
COMPARE = ("Clock A is the first image and clock B the second. Which shows the later time within the same 12 hours? "
           "Answer A or B only.")
FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size: int) -> ImageFont.ImageFont:
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def draw_clock(h: int, m: int, rng: random.Random) -> Image.Image:
    size = rng.choice([320, 384, 448, 512])
    bg = tuple(rng.randint(150, 255) for _ in range(3))
    face = tuple(rng.randint(200, 255) for _ in range(3))
    ink = tuple(rng.randint(0, 70) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    c, r = size / 2, size * rng.uniform(0.38, 0.46)
    d.ellipse([c - r, c - r, c + r, c + r], fill=face, outline=ink, width=max(2, int(size / 80)))
    style = rng.choice(["arabic", "ticks", "both"])
    numerals = font(int(r * 0.18))
    for i in range(60):
        a = math.radians(i * 6)
        major = i % 5 == 0
        if style != "arabic" or major:
            r0 = r * (0.85 if major else 0.92)
            d.line([c + r0 * math.sin(a), c - r0 * math.cos(a), c + r * 0.97 * math.sin(a), c - r * 0.97 * math.cos(a)],
                   fill=ink, width=3 if major else 1)
        if major and style != "ticks":
            x, y = c + r * 0.72 * math.sin(a), c - r * 0.72 * math.cos(a)
            d.text((x, y), str(i // 5 or 12), fill=ink, font=numerals, anchor="mm")
    hour_angle = math.radians((h % 12) * 30 + m * 0.5)
    minute_angle = math.radians(m * 6)
    hand = tuple(rng.randint(0, 90) for _ in range(3))
    d.line([c, c, c + r * 0.5 * math.sin(hour_angle), c - r * 0.5 * math.cos(hour_angle)], fill=hand, width=int(size / 40))
    d.line([c, c, c + r * 0.8 * math.sin(minute_angle), c - r * 0.8 * math.cos(minute_angle)], fill=hand, width=int(size / 70))
    d.ellipse([c - size / 60, c - size / 60, c + size / 60, c + size / 60], fill=hand)
    return img.rotate(rng.uniform(-8, 8), fillcolor=bg)


def sample(i: int, rng: random.Random, out: Path, split: str) -> dict:
    """One sample; "kind" (read or compare) is kept in test.jsonl for scoring."""
    times = [(rng.randint(1, 12), rng.randint(0, 59)) for _ in range(2)]
    while times[0] == times[1]:
        times[1] = (rng.randint(1, 12), rng.randint(0, 59))
    compare = rng.random() < 0.2
    files = []
    for k, (h, m) in enumerate(times[:2 if compare else 1]):
        path = f"{split}/img/{i:05d}_{k}.png"
        draw_clock(h, m, rng).save(out / path)
        files.append(path)
    if compare:
        later = "A" if times[0][0] % 12 * 60 + times[0][1] > times[1][0] % 12 * 60 + times[1][1] else "B"
        content = [{"type": "image", "image": files[0]}, {"type": "image", "image": files[1]}, {"type": "text", "text": COMPARE}]
        return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": later}], "kind": "compare"}
    h, m = times[0]
    content = [{"type": "image", "image": files[0]}, {"type": "text", "text": READ}]
    return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": f"{h}:{m:02d}"}], "kind": "read"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="clocks")
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=300)
    args = parser.parse_args()
    out = Path(args.out)
    for split, count, seed in [("train", args.train, 1), ("test", args.test, 2)]:
        (out / split / "img").mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        with open(out / f"{split}.jsonl", "w") as f:
            for i in range(count):
                row = sample(i, rng, out, split)
                if split == "train":
                    row.pop("kind")
                f.write(json.dumps(row) + "\n")
    print(f"{out}/train.jsonl: {args.train:,} samples, {out}/test.jsonl: {args.test:,}")


if __name__ == "__main__":
    main()
