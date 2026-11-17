"""UltraFeedback preference pairs as one file for a DPO job: each row a prompt with a preferred and a rejected reply.

    pip install datasets
    python recipes/chat_dpo/data.py                 # writes ultrafeedback_pairs.jsonl (4,000 pairs)
"""
import argparse

from datasets import load_dataset
from nextron_cloud import write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", type=int, default=4000)
    args = parser.parse_args()
    rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=0)
    rows = rows.filter(lambda row: row["chosen"][-1]["content"].strip() and row["rejected"][-1]["content"].strip())
    rows = rows.select(range(args.pairs))
    count = write_jsonl(({"chosen": row["chosen"], "rejected": row["rejected"]} for row in rows),
                        "ultrafeedback_pairs.jsonl")
    print(count, "pairs -> ultrafeedback_pairs.jsonl")


if __name__ == "__main__":
    main()
