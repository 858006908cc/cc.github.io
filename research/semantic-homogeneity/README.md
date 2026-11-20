# Semantic Homogeneity As Demonstration (BIDS-ICL)

Research code accompanying [Semantic Homogeneity As Demonstration: Batch-Structured Semi-Supervised In-Context Learning for Natural Language Understanding](https://openreview.net/forum?id=y5bJXgBx6a), by Cheng Chen, Yuan Gang Pan, and Ivor W. Tsang (CPAL 2026).

The paper calls its method **Batch-Structured Implicit Demonstration-Free Semi-supervised ICL (BIDS-ICL)**. It uses semantically homogeneous batches and distinguishes **No-Hint** (unlabeled BS-ICL), **With-Hint** (pseudo-labeled BS-ICL), and the **Random Batch** baseline.

This package contains the attention mechanism experiments and a notebook for summarizing recorded aggregate results. Reproducing the paper’s end-to-end accuracy results requires the original dataset splits, student prediction pipeline, and LLM annotation runs. This package starts from supplied examples and predictions.

## Setup

Use Python 3.10 or later. The default model is Meta Llama 3 8B Instruct. For the full-size checkpoint, obtain access on Hugging Face, authenticate with `hf auth login` or set `HF_TOKEN` in your environment, and use a supported CUDA GPU with sufficient memory. Use `--device cpu` with a small local Llama checkpoint for development.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Inputs

Prepare a UTF-8 JSONL file with an `input` string and a `label` string or integer per row. Create an NPZ with a one-dimensional `pseudo_labels` array containing one encoder-predicted class per JSONL row, in the same order. The legacy `assigned_labels` key is also accepted. Use a string or numeric array. Legacy object arrays require `--allow_pickle` and should only be loaded from a trusted source.

Generate example inputs locally with `python examples/make_bids_icl_demo_inputs.py data/synthetic`. They are intended for smoke tests.

## Attention experiments

```bash
python experiments/bids_icl_homogeneity_diagnostics.py \
  --dataset-name synthetic \
  --examples data/synthetic/examples.jsonl \
  --pseudo-labels data/synthetic/pseudo_labels.npz \
  --batch_size 3 --min_label_frequency 3 --num_batches 4 \
  --layer_idx 0 --head_idx 0 --save_dir results
```

The extended script compares attention entropy, alignment, and residual-stream projections. It writes a JSON report per layer and head. Pass comma-separated zero-based indices to `--test_layers` to generate a summary plot. By default, homogeneous batches are grouped by encoder-predicted pseudo-labels. The original extended script grouped by gold labels. Use `--group-by gold-label` to reproduce that oracle diagnostic. Both scripts use input-only prompts, as in the No-Hint setting. Pseudo-labels can determine the grouping without appearing in the prompt. The notebook summarizes recorded No-Hint and With-Hint results. The simpler `experiments/bids_icl_attention_comparison.py` compares predicted-label batches with randomly drawn batches and prints its results.

Repeatedly drawn batches can overlap. Treat the reported Welch tests as exploratory diagnostics when interpreting these comparisons.

## Results notebook

Run `jupyter lab notebooks/bids_icl_backbone_ablation.ipynb` to regenerate tables and plots from the aggregate values recorded in the notebook.

## Citation

```bibtex
@inproceedings{chen2026semantic,
  title     = {Semantic Homogeneity As Demonstration: Batch-Structured Semi-Supervised In-Context Learning for Natural Language Understanding},
  author    = {Chen, Cheng and Pan, Yuan Gang and Tsang, Ivor W.},
  booktitle = {The Third Conference on Parsimony and Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=y5bJXgBx6a}
}
```
