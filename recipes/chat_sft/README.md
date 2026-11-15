# Chat fine-tuning on NoRobots (managed SFT)

[NoRobots](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 9,500 conversations written by people
(brainstorming, open questions, rewriting, coding, chat), and 500 more as the validation set. Every weight is trained.

```bash
pip install nextron-cloud datasets
export NEXTRON_API_KEY=nk_live_...
python recipes/chat_sft/data.py                       # no_robots_train.jsonl, no_robots_val.jsonl
python recipes/chat_sft/train.py                      # uploads them, prints the steps and the price; starts nothing
python recipes/chat_sft/train.py --yes                # train; prints the validation loss as it goes
python recipes/chat_sft/train.py --model gemma-4-26b-a4b --yes
```

Then try the model without downloading it:

```bash
nextron-cloud inference start --job job_... --max-hours 1 --yes
nextron-cloud inference wait inf_...
nextron-cloud inference url inf_...                   # the base URL for an OpenAI client
```

## Settings

- `--epochs 1`, `--learning-rate 1e-5`: from the sweep below.
- `--tokens-per-gpu 16384`: the batch size is set so each GPU packs about 16K tokens a step (448 conversations on 8
  GPUs, 63 steps for 3 epochs). Without it a step packs 64K tokens a GPU, which is 5 steps an epoch here.

## Results

Qwen3.8-27B, one node of 8 H100, all weights, 448 conversations a step, 3 epochs (63 steps), validation every 6 steps:

| learning rate | lowest val_loss (step) | val_loss at the end | tokens/s/GPU | cost |
|---|---|---|---|---|
| 1e-5 | 1.688 (24) | 1.718 | 2,709 | $8.47 |
| 2e-5 | 1.698 (18) | 1.783 | 2,714 | $8.47 |
| 4e-5 | 1.722 (18) | 1.931 | 2,698 | $8.47 |

The validation loss is lowest after about one epoch (21 steps) and rises after: train one epoch. A one-epoch job costs
about a third of the above.

One epoch with the defaults on the other models (validation every 2 steps; losses are not comparable between models,
their tokenizers differ):

| model | steps | val_loss start -> end | tokens/s/GPU | cost |
|---|---|---|---|---|
| `gemma-4-26b-a4b` | 21 | 2.233 -> 1.832 | 2,583 | $1.43 |
| `muse-glimmer-30b` | 21 | 1.775 -> 1.703 | 2,704 | $3.16 |
