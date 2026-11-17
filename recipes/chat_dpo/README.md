# Preference training on UltraFeedback (DPO)

[UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized): prompts, each with a reply
rated better (chosen) and one rated worse (rejected). DPO trains the model to make the chosen reply more likely than
the rejected one, measured against the model it started from. Every weight is trained.

```bash
pip install nextron-cloud datasets
export NEXTRON_API_KEY=nk_live_...
python recipes/chat_dpo/data.py                       # ultrafeedback_pairs.jsonl: 4,000 pairs
python recipes/chat_dpo/train.py                      # uploads it, prints the steps and the price; starts nothing
python recipes/chat_dpo/train.py --yes                # train; prints the loss of every step
python recipes/chat_dpo/train.py --model qwen3.8-27b --yes
```

Your own pairs work the same way: any JSONL of `{"chosen": [user, assistant], "rejected": [user, assistant]}`, or the
other forms in `nextron-cloud guide dpo`.

## Settings

- `--beta 0.1`: how far the model may move from where it started; lower lets it move further.
- `--learning-rate`: the model's default when not given (Gemma-4-26B 2e-6, Qwen3.8-27B 5e-7; `nextron-cloud models`).
- `--epochs 1`. A step packs whole pairs into each GPU's row, so the step count comes from the data (14 here).

## Results

The loss is exactly 0.693 (ln 2) at step 1 and falls as the model prefers the chosen replies. Each step trains on
pairs it has not seen yet, so a falling loss means the preference carries over to new pairs.

| model | steps | loss at steps 1 / 7 / 14 | tokens/s/GPU | cost |
|---|---|---|---|---|
| `gemma-4-26b-a4b` | 14 | 0.693 / 0.618 / 0.633 (lowest 0.557) | 4,293 | $1.73 |
| `qwen3.8-27b` | 14 | 0.693 / 0.743 / 0.860 | 2,350 | $3.46 |

Gemma-4-26B learns the preference. Qwen3.8-27B does not on this data. Its replies here were written by other models,
and it finds them unlikely, so its scores of the two replies are noisy, and the noise outweighs what one step learns.
A lower learning rate (1e-7) gives the same picture, a higher one (2e-6) makes it worse, and `--beta 0.02` keeps the
loss near 0.693.

For Qwen models, build the pairs from the model's own replies: two replies per prompt from the model, ranked by a judge
or your own scoring.
