# Video fine-tuning: count the bounces (managed SFT)

800 short videos (4 to 10 seconds, 320 x 240) of a coloured square sliding between two walls at its own speed. The
question asks how many times it bounces and off which walls, in order; the reply says so in one sentence. The answer
is in the motion, not in any single frame, and each count from 0 to 5 is equally likely, so a model that always gives
the same count gets 1 in 6. Every weight is trained.

```bash
brew install ffmpeg                                   # or apt install ffmpeg: the videos are drawn locally
pip install nextron-cloud
export NEXTRON_API_KEY=nk_live_...
python recipes/video_sft/data.py                      # bounce/: train.jsonl (800), test.jsonl (100) and the MP4s
python recipes/video_sft/train.py                     # uploads train.jsonl with its videos, prints the steps and the price
python recipes/video_sft/train.py --yes               # train; prints the loss of every step
```

Videos are opened organisation by organisation: `nextron-cloud models` lists `video` in a model's `input` when they
are open to yours. The format, the limits and what each video setting costs are in the SDK guide's
[media.md](https://github.com/Physion-Labs/nextron-python/blob/main/skills/nextron-cloud/references/media.md).

## What the scripts do

- `data.py` draws each video with ffmpeg and writes a sample per video:
  `{"messages": [{"role": "user", "content": [{"type": "video", "video": "clips/train_0000.mp4"}, {"type": "text", "text": "How many times ..."}]}, {"role": "assistant", "content": "The red square bounces twice: off the left, then the right wall."}]}`.
  Every bounce is at least half a second from the start and the end of its video, so the frames training reads
  show each one.
- `datasets.upload` packs the JSONL and the 800 MP4s (8 MB) into one upload; the server checks and re-encodes them.
- `train.py` sizes the batch from the dataset's tokens at the job's video settings (the estimate counts them), so
  each GPU packs about 16K tokens a step. `--video-fps` and `--video-max-frames` change the settings.
- `test.jsonl` is held out: a dataset with videos takes no validation set yet, so the job reports training loss only,
  and the held-out check below runs on the downloaded checkpoint.

## Results

On the 100 held-out clips, the fine-tuned Qwen3.8-27B checkpoint got the bounce count and wall sequence correct on 100/100. Base Qwen3.8-27B got the count correct on 11/100 and the full sequence on 8/100. Both used greedy decoding. The base run received a one-sentence answer-format example and up to 400 output tokens. The checkpoint had up to 120 output tokens. These scores measure this synthetic task and its answer parser.
