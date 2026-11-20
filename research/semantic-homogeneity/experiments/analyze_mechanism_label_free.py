"""Compare attention diagnostics for predicted-label homogeneous and random batches."""

import os
import json
import argparse
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from common import add_input_options, load_inputs, load_model
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# -----------------------------
# 0) Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=8)  # Smaller default for better homogeneity
parser.add_argument("--num_batches", type=int, default=30, help="how many batches to analyze per condition")
parser.add_argument("--layer_idx", type=int, default=12)
parser.add_argument("--head_idx", type=int, default=0)
parser.add_argument("--max_sent_len", type=int, default=200, help="truncate each sentence to this many chars")
parser.add_argument("--group-by", choices=("predicted", "gold"), default="predicted",
                    help="Label source for homogeneous batches. Gold is an oracle diagnostic.")
parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
parser.add_argument("--test_layers", type=str, default="", help="Comma-separated layer indices to test (e.g., '6,12,18')")
parser.add_argument("--min_label_frequency", type=int, default=5, help="Minimum examples per label for homogeneity")
add_input_options(parser)
args = parser.parse_args()
if min(args.batch_size, args.num_batches, args.max_sent_len) < 1:
    parser.error("Batch size, number of batches, and sentence length must be positive.")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Create save directory
os.makedirs(args.save_dir, exist_ok=True)

inputs_pool, true_pool, down_stream_label = load_inputs(args)
if len(inputs_pool) < args.batch_size:
    parser.error("The input file must contain at least one batch of examples.")
tokenizer, model = load_model(args)

# -----------------------------
# 5) Utility functions
# -----------------------------
def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute entropy of probability distribution."""
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(-1)

def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute cosine similarity."""
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)

def test_significance(homo_vals: np.ndarray, rand_vals: np.ndarray, metric_name: str) -> Dict:
    """Perform t-test between conditions."""
    # t_stat, p_value = stats.ttest_ind(homo_vals, rand_vals)

    t_stat, p_value = stats.ttest_ind(homo_vals, rand_vals, equal_var=False)

    pooled_std = np.sqrt((np.std(homo_vals)**2 + np.std(rand_vals)**2) / 2)
    effect_size = (np.mean(homo_vals) - np.mean(rand_vals)) / (pooled_std + 1e-12)

    return {
        "metric": metric_name,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "significant": p_value < 0.05,
        "homo_mean": float(np.mean(homo_vals)),
        "rand_mean": float(np.mean(rand_vals)),
        "homo_std": float(np.std(homo_vals)),
        "rand_std": float(np.std(rand_vals))
    }

# -----------------------------
# 6) Enhanced batch construction functions
# -----------------------------
def build_predicted_homo_batches(
    texts: List[str],
    pred_labels: List[str],
    batch_size: int,
    num_batches: int,
    seed: int = 0,
    min_frequency: int = 5
) -> List[List[str]]:
    """
    More efficient predicted-label homogeneous batches.
    Only uses labels with sufficient examples.
    """
    rng = random.Random(seed)

    # Group indices by predicted label
    idx_by_label = defaultdict(list)
    for i, label in enumerate(pred_labels):
        idx_by_label[label].append(i)

    # Filter labels with enough examples
    eligible_labels = [
        label for label, idxs in idx_by_label.items()
        if len(idxs) >= max(batch_size, min_frequency)
    ]

    if not eligible_labels:
        raise RuntimeError(
            f"No labels have >= max({batch_size}, {min_frequency}) examples. "
            f"Try smaller batch_size or min_label_frequency."
        )

    print(f"[BATCH] {len(eligible_labels)}/{len(idx_by_label)} labels have sufficient examples")

    batches = []
    for _ in range(num_batches):
        label = rng.choice(eligible_labels)
        idxs = rng.sample(idx_by_label[label], batch_size)
        batches.append([texts[i] for i in idxs])

    return batches

def build_true_label_homo_batches(
    texts: List[str],
    true_labels: List[str],
    batch_size: int,
    num_batches: int,
    seed: int = 0,
    min_frequency: int = 5
) -> List[List[str]]:
    """Returns homogeneous batches based on true labels."""
    rng = random.Random(seed)
    idx_by_label = defaultdict(list)

    for i, y in enumerate(true_labels):
        idx_by_label[y].append(i)

    # Keep only labels with enough examples
    eligible = [
        y for y, idxs in idx_by_label.items()
        if len(idxs) >= max(batch_size, min_frequency)
    ]

    if not eligible:
        raise RuntimeError(f"No labels have >= {batch_size} examples.")

    batches = []
    for _ in range(num_batches):
        y = rng.choice(eligible)
        idxs = rng.sample(idx_by_label[y], batch_size)
        batches.append([texts[i] for i in idxs])

    return batches

def build_random_batches(
    texts: List[str],
    batch_size: int,
    num_batches: int,
    seed: int = 0
) -> List[List[str]]:
    """Returns randomly sampled batches."""
    rng = random.Random(seed)
    n = len(texts)

    if n < batch_size:
        raise RuntimeError(f"Not enough examples ({n}) for batch size {batch_size}")

    batches = []
    for _ in range(num_batches):
        idxs = rng.sample(range(n), batch_size)
        batches.append([texts[i] for i in idxs])

    return batches

# -----------------------------
# 7) Mechanism analysis functions
# -----------------------------


@torch.no_grad()
def get_values_for_layer_head(hidden_states, layer_idx, head_idx):
    layer = model.model.layers[layer_idx]
    hs = hidden_states[layer_idx]                  # [bs, seq, hidden]
    hs_norm = layer.input_layernorm(hs)

    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = cfg.hidden_size // num_heads
    group_size = num_heads // num_kv_heads

    V_all = layer.self_attn.v_proj(hs_norm)         # [bs, seq, num_kv_heads*head_dim]
    V_kv = V_all.view(
        V_all.shape[0],
        V_all.shape[1],
        num_kv_heads,
        head_dim
    )

    kv_head_idx = head_idx // group_size
    Vh = V_kv[0, :, kv_head_idx, :].float()
    return Vh


# delta_r_h = get_delta_r_for_layer_head(out, layer_idx, head_idx, qpos)


@torch.no_grad()
def get_delta_r_for_layer_head(out, layer_idx: int, head_idx: int, qpos: int) -> torch.Tensor:
    """
    Returns the residual-stream write (hidden-dim vector) contributed by query head `head_idx`
    at layer `layer_idx` for query position `qpos`.

    This is the clean object to study if you claim "additive accumulation in the residual stream".
    """
    layer = model.model.layers[layer_idx]

    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = cfg.hidden_size // num_heads
    group_size = num_heads // num_kv_heads

    # Attention weights for this head at qpos: [seq]
    alpha_h = out.attentions[layer_idx][0, head_idx, qpos, :].float()  # [seq]

    # Hidden states at this layer (input to attention), normalized
    hs = out.hidden_states[layer_idx]          # [bs, seq, hidden]
    hs_norm = layer.input_layernorm(hs)        # [bs, seq, hidden]

    # Values for KV heads: [seq, num_kv_heads, head_dim]
    V_all = layer.self_attn.v_proj(hs_norm)[0].float()                 # [seq, num_kv_heads*head_dim]
    V_kv = V_all.view(V_all.shape[0], num_kv_heads, head_dim)          # [seq, kv, d]

    kv_head_idx = head_idx // group_size
    Vh = V_kv[:, kv_head_idx, :]                                        # [seq, d]

    # Head output in head-dim: o_h = Σ_j alpha_h[j] Vh[j]
    o_h = (alpha_h.unsqueeze(-1) * Vh).sum(dim=0)                       # [d]

    # Place this head output into the concatenated hidden vector slot
    o_cat = torch.zeros(cfg.hidden_size, device=o_h.device, dtype=o_h.dtype)
    start = head_idx * head_dim
    o_cat[start:start + head_dim] = o_h

    # Apply output projection -> residual write
    # (F.linear(x, W) computes x @ W^T)
    Wo = layer.self_attn.o_proj.weight.float()                          # [hidden, hidden]
    delta_r_h = F.linear(o_cat, Wo)                                     # [hidden]

    return delta_r_h





@torch.no_grad()
def analyze_batch(
    sentences: List[str],
    layer_idx: int,
    head_idx: int,
    support_positions: Optional[List[int]] = None,
) -> Dict:

    # ======================================================
    # 1) Prompt construction
    # ======================================================
    sents = [s[:args.max_sent_len] for s in sentences]

    prompt = ""
    for i, s in enumerate(sents):
        prompt += f"the {i} th sentence is: {s}\n"
    prompt += "Please identify the expressed intentions.\n"

    messages = [{"role": "system", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # ======================================================
    # 2) Forward pass
    # ======================================================
    out = model(
        input_ids=input_ids,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    qpos = input_ids.shape[-1] - 1   # query token position

    # ======================================================
    # 3) Residual-stream write (THIS IS THE KEY ADDITION)
    # ======================================================
    # Δr_h ∈ R^{hidden}
    delta_r_h = get_delta_r_for_layer_head(
        out, layer_idx, head_idx, qpos
    )

    # ----- Define u_hidden (residual-space coherence direction) -----
    H_layer = out.hidden_states[layer_idx][0].float()   # [seq, hidden]

    if support_positions is not None and len(support_positions) > 0:
        u_hidden = H_layer[support_positions].mean(dim=0)
    else:
        u_hidden = H_layer.mean(dim=0)

    u_hidden = u_hidden / (u_hidden.norm() + 1e-12)

    # ----- Residual projection (main mechanistic signal) -----
    proj_resid = float(torch.dot(delta_r_h, u_hidden).item())

    # ======================================================
    # 4) Attention pattern (softmax geometry)
    # ======================================================
    alpha = out.attentions[layer_idx][0, head_idx, qpos, :].float()
    attn_entropy = float(entropy(alpha).item())

    # ======================================================
    # 5) Value-space analysis (kept for continuity)
    # ======================================================
    Vh = get_values_for_layer_head(
        out.hidden_states, layer_idx, head_idx
    )  # [seq, head_dim]

    if support_positions is not None and len(support_positions) > 0:
        u_val = Vh[support_positions].mean(dim=0)
    else:
        u_val = Vh.mean(dim=0)

    u_val = u_val / (u_val.norm() + 1e-12)

    cos_all = cosine(Vh, u_val.unsqueeze(0))

    aligned_mass = float((alpha * (cos_all > 0).float()).sum().item())
    antialigned_mass = float((alpha * (cos_all < 0).float()).sum().item())

    # Head output in value space (diagnostic only)
    attn_out = (alpha.unsqueeze(-1) * Vh).sum(dim=0)
    proj_val = float(torch.dot(attn_out, u_val).item())

    # ======================================================
    # 6) Additional concentration / dispersion metrics
    # ======================================================
    k = min(3, len(alpha))
    topk_mass = torch.topk(alpha, k).values.sum().item()

    alignment_variance = cos_all.var().item()
    effective_positions = torch.exp(entropy(alpha)).item()

    most_aligned_idx = cos_all.argmax()
    most_antialigned_idx = cos_all.argmin()

    # ======================================================
    # 7) Return metrics
    # ======================================================
    return {
        # --- attention geometry ---
        "attn_entropy": attn_entropy,
        "topk_mass": topk_mass,
        "effective_positions": effective_positions,

        # --- value-space diagnostics ---
        "aligned_mass": aligned_mass,
        "antialigned_mass": antialigned_mass,
        "proj_attn_out_on_u": proj_val,
        "alignment_variance": alignment_variance,
        "mass_most_aligned": alpha[most_aligned_idx].item(),
        "mass_most_antialigned": alpha[most_antialigned_idx].item(),

        # --- residual-stream MECHANISM (core claim) ---
        "proj_delta_r_on_u": proj_resid,

        # --- misc ---
        "sequence_length": len(alpha),
    }


# -----------------------------
# 8) Main analysis loop
# -----------------------------
def run_analysis_for_layer_head(
    layer_idx: int,
    head_idx: int,
) -> Tuple[Dict, Dict, Dict]:
    """Run complete analysis for a specific layer and head."""

    print(f"\n{'='*60}")
    print(f"Analyzing Layer {layer_idx}, Head {head_idx}")
    print(f"{'='*60}")

    # Build batches using enhanced functions
    try:
        homo_batches = build_predicted_homo_batches(
            texts=inputs_pool,
            pred_labels=down_stream_label if args.group_by == "predicted" else true_pool,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            seed=args.seed,
            min_frequency=args.min_label_frequency
        )

        rand_batches = build_random_batches(
            texts=inputs_pool,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            seed=args.seed + 999
        )

        print(f"[BATCH] Homogeneous ({args.group_by}) batches: {len(homo_batches)}")
        print(f"[BATCH] Random batches: {len(rand_batches)}")

    except RuntimeError as error:
        raise RuntimeError(f"Batch construction failed: {error}") from error

    # Analyze batches
    homo_results = []
    rand_results = []

    print("\n[RUN] Analyzing homogeneous batches...")
    for batch in homo_batches:
        result = analyze_batch(batch, layer_idx, head_idx)
        if result is not None:
            homo_results.append(result)

    print("[RUN] Analyzing random batches...")
    for batch in rand_batches:
        result = analyze_batch(batch, layer_idx, head_idx)
        if result is not None:
            rand_results.append(result)

    if not homo_results or not rand_results:
        print("[WARN] No valid results obtained")
        return {}, {}, {}

    # Summarize results
    def summarize(results: List[Dict]) -> Dict:
        summary = {}
        for key in results[0].keys():
            vals = np.array([r[key] for r in results], dtype=float)
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(np.median(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "values": vals.tolist()
            }
        return summary

    homo_summary = summarize(homo_results)
    rand_summary = summarize(rand_results)

    # Statistical significance testing
    significance_tests = {}
    for metric in homo_summary.keys():
        homo_vals = np.array([r[metric] for r in homo_results])
        rand_vals = np.array([r[metric] for r in rand_results])
        significance_tests[metric] = test_significance(homo_vals, rand_vals, metric)

    return homo_summary, rand_summary, significance_tests

def print_comparison_table(
    homo_summary: Dict,
    rand_summary: Dict,
    significance_tests: Dict,
    layer_idx: int,
    head_idx: int
) -> None:
    """Print formatted comparison table."""

    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS - Layer {layer_idx}, Head {head_idx}")
    print(f"{'='*80}")

    headers = ["Metric", "Homo Mean ± Std", "Rand Mean ± Std", "Diff", "p-value", "Sig", "ES"]
    row_format = "{:<25} {:<20} {:<20} {:<10} {:<10} {:<5} {:<8}"

    print(row_format.format(*headers))
    print("-" * 100)

    for metric, test in significance_tests.items():
        homo_mean = homo_summary[metric]["mean"]
        homo_std = homo_summary[metric]["std"]
        rand_mean = rand_summary[metric]["mean"]
        rand_std = rand_summary[metric]["std"]

        diff = homo_mean - rand_mean
        p_val = test["p_value"]
        sig = "✓" if p_val < 0.05 else "✗"
        es = test["effect_size"]

        homo_str = f"{homo_mean:.4f} ± {homo_std:.4f}"
        rand_str = f"{rand_mean:.4f} ± {rand_std:.4f}"

        print(row_format.format(
            metric,
            homo_str,
            rand_str,
            f"{diff:.4f}",
            f"{p_val:.4f}",
            sig,
            f"{es:.3f}"
        ))

    print("\nExpected patterns (if hypothesis holds):")
    print("  ✓ attn_entropy:           Homogeneous < Random   (sharper softmax)")
    print("  ✓ aligned_mass:           Homogeneous > Random   (more aligned mass)")
    print("  ✓ antialigned_mass:       Homogeneous < Random   (less anti-aligned mass)")
    print("  ✓ proj_attn_out_on_u:     Homogeneous > Random   (constructive accumulation)")
    print("  ✓ topk_mass:              Homogeneous > Random   (more concentrated)")
    print("  ✓ effective_positions:    Homogeneous < Random   (fewer effective tokens)")

def save_results(
    homo_summary: Dict,
    rand_summary: Dict,
    significance_tests: Dict,
    layer_idx: int,
    head_idx: int,
    args: argparse.Namespace
) -> None:
    """Save results to JSON file."""

    results = {
        "args": {key: value for key, value in vars(args).items() if key not in {"examples", "predictions", "save_dir"}},
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "homo_summary": homo_summary,
        "rand_summary": rand_summary,
        "significance_tests": significance_tests,
    }

    def finite_json(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    filename = f"results_{args.dataset_name}_{args.group_by}_layer{layer_idx}_head{head_idx}_seed{args.seed}.json"
    save_path = os.path.join(args.save_dir, filename)
    with open(save_path, "w", encoding="utf-8") as stream:
        json.dump(finite_json(results), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"[SAVE] {save_path}")


def create_summary_plot(
    all_results: List[Dict],
    args: argparse.Namespace
) -> None:
    """Create summary plot across all tested layers/heads."""

    if not all_results:
        return

    # Prepare data
    layers = []
    attn_entropy_diffs = []
    aligned_mass_diffs = []
    proj_diffs = []
    effect_sizes = []

    for result in all_results:
        layer_idx = result["layer_idx"]
        head_idx = result.get("head_idx", 0)
        sig_tests = result.get("significance_tests", {})

        if "attn_entropy" in sig_tests:
            test = sig_tests["attn_entropy"]
            layers.append(f"L{layer_idx}H{head_idx}")
            attn_entropy_diffs.append(test["homo_mean"] - test["rand_mean"])
            effect_sizes.append(test["effect_size"])

        if "aligned_mass" in sig_tests:
            test = sig_tests["aligned_mass"]
            aligned_mass_diffs.append(test["homo_mean"] - test["rand_mean"])

        if "proj_attn_out_on_u" in sig_tests:
            test = sig_tests["proj_attn_out_on_u"]
            proj_diffs.append(test["homo_mean"] - test["rand_mean"])

    if not layers:
        return

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Attention entropy differences
    x = range(len(layers))
    axes[0, 0].bar(x, attn_entropy_diffs, color=['red' if d > 0 else 'green' for d in attn_entropy_diffs])
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(layers, rotation=45)
    axes[0, 0].set_ylabel("Δ Entropy (Homo - Rand)")
    axes[0, 0].set_title("Attention Entropy Difference")
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Effect sizes
    axes[0, 1].bar(x, effect_sizes, color=['blue' if es > 0 else 'orange' for es in effect_sizes])
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layers, rotation=45)
    axes[0, 1].set_ylabel("Effect Size (Cohen's d)")
    axes[0, 1].set_title("Effect Size by Layer/Head")
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 1].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
    axes[0, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
    axes[0, 1].legend(fontsize='small')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Aligned mass differences
    if aligned_mass_diffs:
        axes[1, 0].bar(x[:len(aligned_mass_diffs)], aligned_mass_diffs,
                       color=['green' if d > 0 else 'red' for d in aligned_mass_diffs])
        axes[1, 0].set_xticks(x[:len(aligned_mass_diffs)])
        axes[1, 0].set_xticklabels(layers[:len(aligned_mass_diffs)], rotation=45)
        axes[1, 0].set_ylabel("Δ Aligned Mass (Homo - Rand)")
        axes[1, 0].set_title("Aligned Mass Difference")
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Projection differences
    if proj_diffs:
        axes[1, 1].bar(x[:len(proj_diffs)], proj_diffs,
                      color=['green' if d > 0 else 'red' for d in proj_diffs])
        axes[1, 1].set_xticks(x[:len(proj_diffs)])
        axes[1, 1].set_xticklabels(layers[:len(proj_diffs)], rotation=45)
        axes[1, 1].set_ylabel("Δ Projection (Homo - Rand)")
        axes[1, 1].set_title("Output Projection Difference")
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Homogeneity Mechanism Analysis - {args.dataset_name} (seed={args.seed})", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(args.save_dir, f"summary_plot_{args.dataset_name}_{args.group_by}_seed{args.seed}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Summary plot saved to {save_path}")

# -----------------------------
# 9) Main execution
# -----------------------------
def main():
    """Main execution function."""

    print("\n" + "="*80)
    print("HOMOGENEITY MECHANISM ANALYSIS")
    print("="*80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches per condition: {args.num_batches}")
    print(f"Seed: {args.seed}")
    print(f"Save directory: {args.save_dir}")
    print("="*80)

    # Determine which layers/heads to test
    if args.test_layers:
        layers_to_test = [int(x) for x in args.test_layers.split(",")]
    else:
        layers_to_test = [args.layer_idx]

    all_results = []

    # Run analysis for each layer
    for layer_idx in layers_to_test:
        homo_summary, rand_summary, significance_tests = run_analysis_for_layer_head(
            layer_idx=layer_idx,
            head_idx=args.head_idx,
        )

        if homo_summary and rand_summary:
            print_comparison_table(
                homo_summary,
                rand_summary,
                significance_tests,
                layer_idx,
                args.head_idx
            )

            save_results(
                homo_summary,
                rand_summary,
                significance_tests,
                layer_idx,
                args.head_idx,
                args
            )

            all_results.append({
                "layer_idx": layer_idx,
                "head_idx": args.head_idx,
                "homo_summary": homo_summary,
                "rand_summary": rand_summary,
                "significance_tests": significance_tests
            })

    # Create summary plot if multiple layers were tested
    if len(all_results) > 1:
        create_summary_plot(all_results, args)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Reports written to {args.save_dir}.")
    print("="*80)

if __name__ == "__main__":
    main()
