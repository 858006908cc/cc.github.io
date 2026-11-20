"""Compare attention diagnostics for semantically homogeneous batches and the Random Batch baseline."""

import argparse
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from bids_icl_inputs import add_input_options, load_inputs, load_model

# -----------------------------
# 0) Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--num_batches", type=int, default=30, help="how many batches to analyze per condition")
parser.add_argument("--layer_idx", type=int, default=12)
parser.add_argument("--head_idx", type=int, default=0)
parser.add_argument("--max_sent_len", type=int, default=200, help="truncate each sentence to this many chars")
add_input_options(parser)
args = parser.parse_args()
if min(args.batch_size, args.num_batches, args.max_sent_len) < 1:
    parser.error("Batch size, number of batches, and sentence length must be positive.")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

inputs_pool, gold_labels, pseudo_labels = load_inputs(args)
if len(inputs_pool) < args.batch_size:
    parser.error("The input file must contain at least one batch of examples.")
tokenizer, model = load_model(args)

# -----------------------------
# 5) Build homogeneous ordering by predicted label (your idea)
# -----------------------------
labels_unique = np.unique(pseudo_labels)
label_to_idx = {lab: i for i, lab in enumerate(labels_unique)}
order = np.argsort([label_to_idx[lab] for lab in pseudo_labels])

inputs_sorted = [inputs_pool[i] for i in order]
pseudo_labels_sorted = [pseudo_labels[i] for i in order]
gold_labels_sorted = [gold_labels[i] for i in order]

# -----------------------------
# 6) Mechanism analysis (attention percentages + value alignment)
# -----------------------------
def entropy(p, eps=1e-12):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(-1)

def cosine(a, b, eps=1e-12):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


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


@torch.no_grad()
def analyze_batch(sentences, layer_idx, head_idx):
    """
    Build a batch-structured prompt consistent with your agent11 "the i th sentence is: ..."
    Then compute:
    - attention entropy at last token (query)
    - alignment mass wrt u (mean value direction)
    - projection <attn_out, u>
    """
    # Truncate to keep prompts bounded
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

    out = model(
        input_ids=input_ids,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    qpos = input_ids.shape[-1] - 1

    # Attention pattern alpha: [seq]
    alpha = out.attentions[layer_idx][0, head_idx, qpos, :].float()
    H = float(entropy(alpha).item())

    # Values: [seq, head_dim]
    Vh = get_values_for_layer_head(out.hidden_states, layer_idx, head_idx)

    # Define shared direction u_ell as mean over all value vectors in this prompt
    u = Vh.mean(dim=0)
    u = u / (u.norm() + 1e-12)

    # Alignment per position
    cos_all = cosine(Vh, u.unsqueeze(0))  # [seq]
    aligned_mass = float((alpha * (cos_all > 0).float()).sum().item())
    antialigned_mass = float((alpha * (cos_all < 0).float()).sum().item())

    # Head output (attn_out) and projection
    attn_out = (alpha.unsqueeze(-1) * Vh).sum(dim=0)  # [head_dim]
    proj = float(torch.dot(attn_out, u).item())

    return {
        "attn_entropy": H,
        "aligned_mass": aligned_mass,
        "antialigned_mass": antialigned_mass,
        "proj_attn_out_on_u": proj,
    }

def sample_homogeneous_batches(inputs_sorted, pseudo_labels_sorted, batch_size, num_batches):
    """
    Take contiguous windows where predicted labels are identical.
    """
    batches = []
    n = len(inputs_sorted)
    i = 0
    while i < n and len(batches) < num_batches:
        lab = pseudo_labels_sorted[i]
        j = i
        while j < n and pseudo_labels_sorted[j] == lab:
            j += 1
        # segment [i, j) is homogeneous in predicted label
        seg_len = j - i
        if seg_len >= batch_size:
            # take one or more windows
            start = i
            while start + batch_size <= j and len(batches) < num_batches:
                batches.append(inputs_sorted[start:start+batch_size])
                start += batch_size
        i = j
    return batches

from collections import defaultdict
import random

def build_gold_label_oracle_batches(texts, true_labels, batch_size, num_batches, seed=0):
    """
    Returns list[list[str]] of homogeneous batches where all labels are identical.
    """
    rng = random.Random(seed)
    idx_by_label = defaultdict(list)
    for i, y in enumerate(true_labels):
        idx_by_label[y].append(i)

    # keep only labels with enough examples
    eligible = [y for y, idxs in idx_by_label.items() if len(idxs) >= batch_size]
    if not eligible:
        raise RuntimeError(f"No labels have >= batch_size={batch_size} examples.")

    batches = []
    # sample labels then sample indices within each label
    for _ in range(num_batches):
        y = rng.choice(eligible)
        idxs = rng.sample(idx_by_label[y], batch_size)
        batches.append([texts[i] for i in idxs])
    return batches

def build_random_batches(texts, batch_size, num_batches, seed=0):
    rng = random.Random(seed)
    n = len(texts)
    batches = []
    for _ in range(num_batches):
        idxs = rng.sample(range(n), batch_size)
        batches.append([texts[i] for i in idxs])
    return batches


def sample_random_batches(inputs_pool, batch_size, num_batches):
    batches = []
    n = len(inputs_pool)
    for _ in range(num_batches):
        idx = np.random.choice(n, size=batch_size, replace=False)
        batches.append([inputs_pool[i] for i in idx])
    return batches

# # small_labels_true
# texts = small_inputs
# true_labels = small_labels_true

# homogeneous_batches = build_gold_label_oracle_batches(
#     texts=texts,
#     true_labels=true_labels,
#     batch_size=args.batch_size,
#     num_batches=args.num_batches,
#     seed=args.seed
# )

# random_batches = build_random_batches(
#     texts=texts,
#     batch_size=args.batch_size,
#     num_batches=args.num_batches,
#     seed=args.seed + 999
# )


# -----------------------------
# 7) Run analysis on dataset-derived batches
# -----------------------------
homogeneous_batches = sample_homogeneous_batches(inputs_sorted, pseudo_labels_sorted, args.batch_size, args.num_batches)
random_batches = sample_random_batches(inputs_pool, args.batch_size, args.num_batches)

if len(homogeneous_batches) == 0:
    raise RuntimeError(
        "Could not find any homogeneous segments of length >= batch_size "
        f"(batch_size={args.batch_size}). Try smaller batch_size or different dataset/seed."
    )

print(f"[BATCH] homogeneous batches: {len(homogeneous_batches)} | random batches: {len(random_batches)}")

def summarize(results):
    # results: list[dict]
    keys = results[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out

homo_res = []
rand_res = []

print("\n[RUN] analyzing homogeneous batches ...")
for b in homogeneous_batches:
    homo_res.append(analyze_batch(b, args.layer_idx, args.head_idx))

print("[RUN] analyzing random batches ...")
for b in random_batches:
    rand_res.append(analyze_batch(b, args.layer_idx, args.head_idx))

homo_sum = summarize(homo_res)
rand_sum = summarize(rand_res)

print("\n=== Mechanism summary (mean ± std) ===")
print(f"Layer={args.layer_idx}, Head={args.head_idx}, batch_size={args.batch_size}, num_batches={args.num_batches}\n")

print("Semantically homogeneous batches (grouped by pseudo-label):")
for k, (m, s) in homo_sum.items():
    print(f"  {k:22s}: {m:.6f} ± {s:.6f}")

print("\nRandom Batch baseline:")
for k, (m, s) in rand_sum.items():
    print(f"  {k:22s}: {m:.6f} ± {s:.6f}")

print("\n=== Expected direction if your hypothesis holds ===")
print("- attn_entropy:           Homogeneous < Random   (sharper softmax percentages)")
print("- aligned_mass:           Homogeneous > Random   (more mass on aligned values)")
print("- antialigned_mass:       Homogeneous < Random   (less mass on conflicting values)")
print("- proj_attn_out_on_u:     Homogeneous > Random   (constructive accumulation in residual)")
