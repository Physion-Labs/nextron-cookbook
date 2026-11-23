# Image fine-tuning: read analog clocks (managed SFT)

6,000 pictures of analog clocks with different faces, colours, numerals, hands and a small tilt. Most samples ask
for the time as `H:MM`; one in five shows two clocks in one message and asks which is later. A base model reads the
hour and roughly the minute, but rarely the exact minute; 3 epochs of SFT teach it. Runs on any model whose `input`
lists `image` in `nextron-cloud models`.

```bash
pip install nextron-cloud pillow
export NEXTRON_API_KEY=nk_live_...
python recipes/image_sft/data.py                                  # clocks/: train.jsonl (6,000), test.jsonl (300), PNGs
python recipes/image_sft/train.py                                 # uploads train.jsonl with its images; prints steps and price
python recipes/image_sft/train.py --yes                           # qwen3.5-4b, full, 3 epochs; prints the loss of each step
python recipes/image_sft/train.py --model qwen3.5-9b --lora --yes
python recipes/image_sft/evaluate.py job_...                      # serves the job's model, scores test.jsonl, stops the run
```

Images are opened organisation by organisation: `nextron-cloud models` lists `image` in a model's `input` when they
are open to yours. The format and limits are in the SDK guide's
[media.md](https://github.com/Physion-Labs/nextron-python/blob/main/skills/nextron-cloud/references/media.md).

## What the scripts do

- `data.py` draws every clock with Pillow and writes one sample per picture:
  `{"messages": [{"role": "user", "content": [{"type": "image", "image": "train/img/00000_0.png"}, {"type": "text", "text": "What time does this clock show? ..."}]}, {"role": "assistant", "content": "1:05"}]}`.
  The two-clock samples hold two image parts before the question.
- `datasets.upload` packs the JSONL and its 7,155 PNGs (82 MB) into one upload; the server checks every image.
  The dataset is about 1.5M tokens on the Qwen models (an image of 512 x 512 is 258 tokens).
- `train.py` sets `batch_size` 264: at the default step (each GPU packs 64K tokens) these short samples make 3 steps
  an epoch; 264 is the smallest batch the service takes for them on 8 GPUs, 23 steps an epoch.
- `evaluate.py` checks the 362 test images as one dataset (`datasets.upload_images`) before it starts an inference
  run with `images=True`, gives them to the run, sends the 300 held-out prompts in one `sample` call
  (temperature 0, 16 tokens; `--base` gives a base model 512 to reason and scores its last answer), and stops the run.
  `test.jsonl` is never uploaded for training: a dataset with images takes no validation set yet.

## Results

Measured 2026-11-23 on NexCloud, one node of 8 H100, 3 epochs, batch 264 (69 steps), full at lr 2e-5, LoRA (rank 32)
at lr 2e-4. Held-out: 238 clocks to read, 62 pairs to compare. "Base" is the model before training, given 512 tokens
to reason (`qwen3.5-4b`: 16, it answers at once).

| Model | Base: exact / within a minute / later | Full: exact / later | LoRA: exact / later |
|---|---|---|---|
| `qwen3.5-4b` | 38.7% / 91.2% / 41.9% | 91.2% / 62.9% | 96.6% / 90.3% |
| `qwen3.5-9b` | 39.9% / 95.8% / 82.3% | 95.0% / 77.4% | 97.9% / 79.0% |
| `qwen3.5-9b-base` | 36.1% / 93.3% / 90.3% | 97.1% / 91.9% | 95.4% / 82.3% |
| `qwen3.6-35b-a3b` | 37.4% / 100% / 83.9% | 98.3% / 91.9% | 95.8% / 83.9% |
| `qwen3.5-35b-a3b-base` | 35.7% / 89.9% / 48.4% | 98.7% / 88.7% | 96.6% / 80.6% |

"Exact": the time read to the minute; "later": which of two clocks is later. After training every model reads within a
minute on all 238 clocks.

- Every model reads the exact minute after training (36-40% before, 91-99% after); before, most answers were one
  minute off.
- Asked which of two clocks is later, the base models that reason first (9B, 9B-Base, 35B-A3B) already get 82-90%;
  a fine-tuned model answers with one letter and gets 63-92%.
- LoRA beat full training on `qwen3.5-4b` and `qwen3.5-9b` at these learning rates; full was ahead on `qwen3.5-9b-base`
  and both 35B-A3B models. Full training of the 4B model breaks down at lr 1e-4.

Settings on `qwen3.5-4b`: each full row changes the learning rate of the first, each LoRA row one setting of the
rank-32 row.

| Setting | Exact | Later | Training |
|---|---|---|---|
| full, lr 2e-5 | 91.2% | 62.9% | $0.92 |
| full, lr 5e-5 | 71.0% | 85.5% | $0.92 |
| full, lr 1e-4 | 0% (a few fixed answers) | 43.5% | $0.92 |
| LoRA rank 32, lr 2e-4 | 96.6% | 90.3% | $0.92 |
| LoRA rank 8 | 92.9% | 85.5% | $0.92 |
| LoRA rank 64 | 96.2% | 87.1% | $0.92 |
| LoRA rank 32, 5 epochs | **98.3%** | **91.9%** | $1.53 |

A run cost $0.92-$1.60 to train (1.5M tokens x 3 epochs, 69 steps) and $0.63-$1.19 of inference run time to score.
