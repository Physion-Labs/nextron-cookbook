"""Serverless RL with images in the prompts: name the colour that fills a picture, 0/1 reward, three steps.

    export NEXTRON_API_KEY=nk_live_...
    nextron-cloud rl start --model qwen3.8-27b --max-hours 1 --images --yes   # prints the run's id; billed from ready
    nextron-cloud rl wait rl_...
    python recipes/rl_images.py rl_...                                         # about 5 minutes; stops the run
    nextron-cloud rl stop rl_... --wait                                        # always, in case the loop stopped early

The pictures are made here (plain PNGs, one colour each). `add_images` sends them to the run once, before the loop; a
prompt then names a picture by its path. Videos work the same way on the models whose `input` lists `video`
(`{"type": "video", "video": "clip.mp4"}`). Images and videos are opened per organisation: `nextron-cloud models`
lists them in a model's `input` when they are open to yours.
"""
import random
import re
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from nextron_cloud import NexTron

COLOURS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 220, 30),
           "purple": (130, 40, 160), "orange": (245, 140, 20), "black": (10, 10, 10), "white": (250, 250, 250)}
QUESTION = "What colour fills this image? Put the one-word answer in \\boxed{}."


def png(rgb, size=96):
    """A PNG file of one colour."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + bytes(rgb) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def reward(reply, colour):
    found = re.findall(r"\\boxed\{([^}]*)\}", reply)
    return 1.0 if found and found[-1].strip().lower() == colour else 0.0


def main():
    folder = Path(tempfile.mkdtemp())
    pictures = {}
    for colour, rgb in COLOURS.items():
        pictures[colour] = folder / f"{colour}.png"
        pictures[colour].write_bytes(png(rgb))
    with NexTron().rl.connect(sys.argv[1]) as live:
        live.add_images([str(path) for path in pictures.values()])   # once: uploaded, checked and cleaned
        for step in range(3):
            colours = random.choices(list(COLOURS), k=64)
            prompts = [[{"role": "user", "content": [{"type": "image", "image": str(pictures[c])},
                                                     {"type": "text", "text": QUESTION}]}] for c in colours]
            batch = live.sample(prompts, n=8, max_tokens=256)
            rewards = [[reward(reply.text or "", c) for reply in group] for c, group in zip(colours, batch)]
            result = live.train(batch, rewards, learning_rate=1e-6)
            print(f"step {step}: reward {result.reward_mean:.3f}; sampling {result.generation_s:.0f} s, "
                  f"training {result.train_s:.0f} s ({result.tokens_per_s_per_gpu or 0:.0f} tokens/s/GPU)")
        live.finish()   # saves the model and stops the run


if __name__ == "__main__":
    main()
