# Self-Teaching Prompting for Multi-Intent Learning

Code accompanying [Self-Teaching Prompting for Multi-Intent Learning with Limited Supervision](https://openreview.net/notes/edits/attachment?id=FZR4qpTuUS&name=pdf), by Cheng Chen and Ivor W. Tsang, ICLR 2024 Tiny Papers.

This release is a clean implementation of the method in Section 3 and Appendix A.0.1. It supports **MixATIS** and **MixSNIPS** through their parallel-file dataset format, with a synthetic example for trying the procedure locally.

## Method

1. Start with a small pool of labeled seed examples and use cosine similarity over Word2Vec sentence embeddings to retrieve a hint.
2. Predict the intent set at temperatures **0.1, 0.3, 0.5, and 0.7**.
3. Keep a predicted set only when all four predictions contain exactly the same intents.
4. Use the seed examples and the previous epoch’s agreed predictions as hints in the next epoch. Run three epochs.

Epoch one retrieves one hint. Use `--later-hints` to set the maximum in later epochs. The default is one as in the main algorithm. Retrieval excludes the current utterance and identical normalized text. The implementation uses mean-pooled Word2Vec embeddings and a JSON output prompt.

## Run the synthetic example

From this directory, with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python stp.py --demo --seeds examples/seeds.jsonl \
  --unlabeled examples/unlabeled.jsonl --labels examples/labels.txt \
  --output runs/demo
python -m unittest discover -s tests
```

The demo uses a deterministic responder to illustrate agreement and disagreement without an API key. Each epoch writes a JSONL file with the selected hint IDs, the prediction at each temperature, and the agreed intent set or null. Choose a fresh output directory for each run.

## MixATIS and MixSNIPS

Get the cleaned datasets from the [GL-GIN data release](https://github.com/qinlibo-hit/GL-GIN/tree/master/data). The converter reads `seq.in` and `label` in the training split. Multiple intents are separated by `#`.

```bash
python prepare_data.py --train /path/to/MixATIS_clean/train \
  --seed-size 70 --output data/mixatis
python prepare_data.py --train /path/to/MixSNIPS_clean/train \
  --seed-size 400 --output data/mixsnips
```

The converter creates seeded random splits of the requested sizes. For a particular experiment, supply its original seed set and unlabeled pool if available. Both files use an `id` and `text` per row, and only the labeled seed file includes `labels`. The standalone `labels.txt` file specifies the closed-set intent inventory.

For a live run, set `OPENAI_API_KEY` in the environment and select a Chat Completions model that supports the requested temperature values. Each utterance and its retrieved hints are sent to the model API.

```bash
python stp.py --model MODEL_ID --seeds data/mixatis/seeds.jsonl \
  --unlabeled data/mixatis/unlabeled.jsonl --labels data/mixatis/labels.txt \
  --output runs/mixatis
```

Use the corresponding `data/mixsnips` files to run MixSNIPS. Each full run makes 12 model requests per unlabeled utterance. The code and input formats have been tested locally. Reproducing the paper’s reported scores requires the historical seed selection, prompts, model version, and evaluation configuration.

## Citation

```bibtex
@inproceedings{chen2024selfteaching,
  title     = {Self-Teaching Prompting for Multi-Intent Learning with Limited Supervision},
  author    = {Chen, Cheng and Tsang, Ivor W.},
  booktitle = {ICLR 2024 Tiny Papers},
  year      = {2024},
  url       = {https://openreview.net/notes/edits/attachment?id=FZR4qpTuUS&name=pdf}
}
```
