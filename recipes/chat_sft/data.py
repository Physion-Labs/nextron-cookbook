"""NoRobots as two files for a managed fine-tune: train.jsonl (9,500 conversations) and val.jsonl (its 500 test ones).

    pip install datasets
    python recipes/chat_sft/data.py            # writes no_robots_train.jsonl and no_robots_val.jsonl
"""
from datasets import load_dataset
from nextron_cloud import write_jsonl


def main():
    for split, path in (("train", "no_robots_train.jsonl"), ("test", "no_robots_val.jsonl")):
        rows = load_dataset("HuggingFaceH4/no_robots", split=split)
        count = write_jsonl(({"messages": row["messages"]} for row in rows), path)
        print(count, "conversations ->", path)


if __name__ == "__main__":
    main()
