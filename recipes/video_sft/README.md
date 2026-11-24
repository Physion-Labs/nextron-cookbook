# Video fine-tuning: count the bounces (managed SFT)

800 short videos (4 to 10 seconds, 320 x 240) of a coloured square sliding between two walls at its own speed. The
question asks how many times it bounces and off which walls, in order; the reply says so in one sentence. The answer
is in the motion, not in any single frame, and each count from 0 to 5 is equally likely, so a model that always gives
the same count gets 1 in 6. Qwen3.8-27B trains every weight; Gemma-4-26B-A4B trains a LoRA adapter.

```bash
brew install ffmpeg                                   # or apt install ffmpeg: the videos are drawn locally
pip install nextron-cloud
export NEXTRON_API_KEY=nk_live_...
python recipes/video_sft/data.py                      # bounce/: train.jsonl (800), test.jsonl (100) and the MP4s
python recipes/video_sft/train.py                     # uploads train.jsonl with its videos, prints the steps and the price
python recipes/video_sft/train.py --yes               # train; prints the loss of every step and the job id
python recipes/video_sft/evaluate.py job_...          # serve the trained model, score it on test.jsonl, stop the run
# Gemma: a LoRA adapter, 140 tokens a frame
python recipes/video_sft/train.py --model gemma-4-26b-a4b --lora --learning-rate 1e-4 --video-frame-tokens 140 --yes
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
  each GPU packs about 16K tokens a step. `--video-fps` (Qwen), `--video-max-frames` and `--video-frame-tokens`
  change the settings.
- `test.jsonl` is held out: a dataset with videos takes no validation set yet, so the job reports training loss only.
- `evaluate.py` uploads the 100 test videos as one dataset, starts an inference run of the job's model (it reads
  each video at the settings the job trained with), asks every question with greedy decoding and up to 120 tokens,
  then stops the run and deletes the videos. The run is billed by the hour while ready, a few minutes here.

## Results

Measured with `evaluate.py` on the 100 held-out clips, 3 epochs on one 8-GPU node each (2026-11-23):

| Model | Training | Video settings | Steps | Charged | Count right | Count and walls |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | full, lr 1e-5 | 4 fps, 40 frames, 512 tokens a frame (defaults) | 45 | $4.88 | 100/100 | 100/100 |
| Gemma-4-26B-A4B | LoRA, lr 1e-4 | 32 frames, 140 tokens a frame | 75 | $5.41 | 93/100 | 93/100 |

Base Qwen3.8-27B got the count right on 11/100 and the full sequence on 8/100 (greedy, a one-sentence answer-format
example and up to 400 tokens). These scores measure this synthetic task and its answer parser.
