"""MATH problems with a numeric answer as an RFT dataset: one {"prompt", "answer"} row a problem.

    pip install datasets
    python recipes/math_rft/data.py            # writes math_numeric.jsonl (4,846 problems)

The built-in `boxed` reward compares numbers as numbers, so this keeps the problems whose answer is one number and
leaves out fractions, expressions and intervals, which need a grader of your own (see ../math_rl).
"""
import json
import re

from datasets import load_dataset

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus"]
INSTRUCTION = "\n\nReason briefly, then give the final answer as \\boxed{number}."


def boxed(solution):
    """The contents of the last \\boxed{...} in a solution, or None."""
    start = solution.rfind("\\boxed{")
    if start < 0:
        return None
    depth, begin = 0, start + len("\\boxed{")
    for index in range(begin, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            if depth == 0:
                return solution[begin:index]
            depth -= 1
    return None


def main():
    rows = []
    for subject in SUBJECTS:
        for problem in load_dataset("EleutherAI/hendrycks_math", subject, split="train"):
            answer = boxed(problem["solution"])
            if answer and re.fullmatch(r"-?\d[\d,]*(\.\d+)?", answer):
                rows.append({"prompt": problem["problem"] + INSTRUCTION, "answer": answer})
    with open("math_numeric.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(len(rows), "problems -> math_numeric.jsonl")


if __name__ == "__main__":
    main()
