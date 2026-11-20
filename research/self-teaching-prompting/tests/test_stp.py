import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_data import prepare
from stp import Example, TEMPERATURES, parse_labels, read_examples, run_stp


class SelfTeachingTests(unittest.TestCase):
    def test_agreement_updates_hints_between_epochs_without_self_retrieval(self):
        seeds = [Example("seed", "clean example", ("a",))]
        inputs = [Example("x", "new example"), Example("y", "ambiguous example")]
        calls = []

        def embed(_text):
            return np.ones(2)

        def predict(row, hints, _labels, temperature):
            ids = [hint.id for hint in hints]
            calls.append((row.id, ids, temperature))
            if row.id == "y" and temperature == TEMPERATURES[-1]:
                return ["b"]
            return ["b", "a"] if row.id == "x" and temperature == TEMPERATURES[-1] else (
                ["a", "b"] if row.id == "x" else ["a"]
            )

        rounds = list(run_stp(seeds, inputs, {"a", "b"}, embed, predict, later_hints=2))
        self.assertEqual([i for i, _ in rounds], [1, 2, 3])
        self.assertEqual(len(calls), 3 * len(inputs) * len(TEMPERATURES))
        for _, rows in rounds:
            self.assertEqual(rows[0]["consistent_labels"], ("a", "b"))
            self.assertIsNone(rows[1]["consistent_labels"])
            self.assertEqual(rows[0]["hint_ids"], ["seed"])
        self.assertEqual(rounds[0][1][1]["hint_ids"], ["seed"])
        self.assertEqual(rounds[1][1][1]["hint_ids"], ["seed", "x"])

    def test_invalid_response_is_rejected(self):
        for response in (None, [], ["known", "unknown"], [{}], "known"):
            self.assertIsNone(parse_labels(response, {"known"}))

    def test_converter_keeps_gold_out_of_unlabeled_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seq.in").write_text("book a flight\nplay jazz and set an alarm\n", encoding="utf-8")
            (root / "label").write_text("flight\nmusic#alarm\n", encoding="utf-8")
            output = root / "converted"
            prepare(root, output, seed_size=1, random_state=42)
            seeds = read_examples(output / "seeds.jsonl", labeled=True)
            pool = read_examples(output / "unlabeled.jsonl", labeled=False)
            self.assertNotEqual(seeds[0].id, pool[0].id)
            self.assertEqual(pool[0].labels, ())
            self.assertEqual(set((output / "labels.txt").read_text().split()), {"alarm", "flight", "music"})
