"""Generate an invented dataset and matching pseudo-labels for smoke tests."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sentences = {
        "weather": ["Will it rain later?", "Is it snowing?", "Will tomorrow be sunny?", "How windy is it?"],
        "calendar": ["Add an event.", "Cancel my meeting.", "Move my appointment.", "When is my next meeting?"],
    }
    examples = [{"input": text, "label": label} for label, texts in sentences.items() for text in texts]
    (args.output / "examples.jsonl").write_text(
        "".join(json.dumps(example) + "\n" for example in examples), encoding="utf-8"
    )
    np.savez(args.output / "pseudo_labels.npz", pseudo_labels=[example["label"] for example in examples])


if __name__ == "__main__":
    main()
