"""Convert a MixATIS or MixSNIPS parallel-file train split to STP JSONL."""

import argparse
import random
from pathlib import Path

from stp import write_jsonl


def prepare(split, output, seed_size, random_state):
    texts = (split / "seq.in").read_text(encoding="utf-8").splitlines()
    labels = (split / "label").read_text(encoding="utf-8").splitlines()
    if len(texts) != len(labels) or not 0 < seed_size < len(texts):
        raise ValueError("the parallel files must match and seed size must leave unlabeled rows")
    data = []
    for index, (text, label_line) in enumerate(zip(texts, labels)):
        intents = [label.strip() for label in label_line.split("#")]
        if not text.strip() or not all(intents):
            raise ValueError(f"empty text or intent in row {index + 1}")
        data.append({"id": f"train-{index + 1}", "text": text, "labels": intents})

    seed_rows = set(random.Random(random_state).sample(range(len(data)), seed_size))
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "seeds.jsonl", (row for n, row in enumerate(data) if n in seed_rows))
    write_jsonl(output / "unlabeled.jsonl", (
        {"id": row["id"], "text": row["text"]} for n, row in enumerate(data) if n not in seed_rows
    ))
    inventory = sorted({label for row in data for label in row["labels"]})
    (output / "labels.txt").write_text("\n".join(inventory) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.train, args.output, args.seed_size, args.seed)


if __name__ == "__main__":
    main()
