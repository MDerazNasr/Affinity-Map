"""
script/embed_and_plot.py

Generates PCA, UMAP, and prototype distance heatmap figures from the
trained CNN ProtoNet encoder on the full 155-family dataset.

Outputs (results/):
  pca_embeddings.png/.pdf
  umap_embeddings.png/.pdf
  prototype_distance_heatmap.png/.pdf

Run from project root:
    python script/embed_and_plot.py
"""

import os, sys, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.encoder import ProteinEncoderCNN
from data.configs.protonet import CONF

RESULTS      = "results"
CHECKPOINT   = "checkpoints/best_protonet.pt"
PROTEINS_JSON= "data/processed/proteins.json"
PAD_LEN      = 400
FIG_DPI      = 200
MAX_PER_FAM  = 30   # sequences per family to embed (keeps it fast)
RANDOM_SEED  = 42

AA_ORDER  = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": FIG_DPI,
    "savefig.bbox": "tight", "savefig.dpi": FIG_DPI,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def encode_seq(seq):
    tokens = [AA_TO_IDX.get(aa, 0) for aa in seq[:PAD_LEN]]
    tokens += [0] * (PAD_LEN - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

def save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(RESULTS, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved {path}")
    plt.close(fig)

# ── load model ────────────────────────────────────────────────────────────────

def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg  = ckpt.get("config", CONF)
    model = ProteinEncoderCNN(proj_dim=cfg["proj_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# ── embed all families ────────────────────────────────────────────────────────

def get_embeddings(model, device):
    random.seed(RANDOM_SEED)
    with open(PROTEINS_JSON) as f:
        all_seqs = json.load(f)

    embeddings, labels = [], []
    family_names = sorted(all_seqs.keys())

    with torch.no_grad():
        for fam in family_names:
            seqs = all_seqs[fam]
            random.shuffle(seqs)
            seqs = seqs[:MAX_PER_FAM]
            batch = torch.stack([encode_seq(s) for s in seqs]).to(device)
            embs  = model(batch).cpu().numpy()
            embeddings.append(embs)
            labels.extend([fam] * len(seqs))

    X = np.vstack(embeddings)
    y = np.array(labels)
    print(f"  Embedded {len(X)} sequences from {len(family_names)} families")
    return X, y, family_names

# ── colour palette ────────────────────────────────────────────────────────────

def make_palette(n):
    cmap = plt.cm.get_cmap("tab20", 20)
    colours = [cmap(i % 20) for i in range(n)]
    return colours

# ── 1. PCA ────────────────────────────────────────────────────────────────────

def fig_pca(X, y, family_names):
    pca  = PCA(n_components=2, random_state=RANDOM_SEED)
    X2   = pca.fit_transform(X)
    le   = LabelEncoder().fit(y)
    yi   = le.transform(y)
    n    = len(family_names)
    cols = make_palette(n)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, fam in enumerate(sorted(set(y))):
        mask = y == fam
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=[cols[le.transform([fam])[0]]],
                   s=12, alpha=0.7, linewidths=0)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var.)")
    ax.set_title(f"PCA of CNN ProtoNet embeddings  ({n} families, {len(X)} sequences)")

    # legend — only if ≤ 40 families; otherwise skip for readability
    if n <= 40:
        patches = [mpatches.Patch(color=cols[i], label=fam)
                   for i, fam in enumerate(sorted(set(y)))]
        ax.legend(handles=patches, fontsize=5, ncol=2,
                  loc="upper right", frameon=False)

    fig.tight_layout()
    save(fig, "pca_embeddings")
    return X2

# ── 2. UMAP ───────────────────────────────────────────────────────────────────

def fig_umap(X, y, family_names):
    try:
        import umap as umap_mod
    except ImportError:
        print("  [skip] umap-learn not installed")
        return

    print("  Running UMAP (may take ~1 min)...")
    reducer = umap_mod.UMAP(n_neighbors=15, min_dist=0.1,
                             metric="cosine", random_state=RANDOM_SEED)
    X2  = reducer.fit_transform(X)
    le  = LabelEncoder().fit(y)
    n   = len(family_names)
    cols = make_palette(n)

    fig, ax = plt.subplots(figsize=(9, 7))
    for fam in sorted(set(y)):
        mask = y == fam
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=[cols[le.transform([fam])[0]]],
                   s=12, alpha=0.7, linewidths=0)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP of CNN ProtoNet embeddings  ({n} families, {len(X)} sequences)")

    if n <= 40:
        patches = [mpatches.Patch(color=cols[i], label=fam)
                   for i, fam in enumerate(sorted(set(y)))]
        ax.legend(handles=patches, fontsize=5, ncol=2,
                  loc="upper right", frameon=False)

    fig.tight_layout()
    save(fig, "umap_embeddings")

# ── 3. Prototype distance heatmap ─────────────────────────────────────────────

def fig_proto_heatmap(X, y, family_names):
    fams_sorted = sorted(set(y))
    n = len(fams_sorted)

    # Prototype = mean embedding per family
    protos = np.stack([X[y == fam].mean(axis=0) for fam in fams_sorted])
    # Normalise to unit vectors (cosine distance = 1 - dot product)
    norms  = np.linalg.norm(protos, axis=1, keepdims=True)
    protos = protos / (norms + 1e-8)
    # Cosine distance matrix
    sim    = protos @ protos.T
    dist   = 1 - sim
    np.fill_diagonal(dist, 0)

    fig_size = max(10, n * 0.12)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    cmap     = LinearSegmentedColormap.from_list("wb", ["white", "#2166AC"])
    im       = ax.imshow(dist, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cosine distance")

    # Short labels
    short = [f[:12] for f in fams_sorted]
    tick_fs = max(4, 8 - n // 20)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(short, fontsize=tick_fs)
    ax.set_title(f"Pairwise cosine distance between family prototypes  ({n} families)")
    ax.set_xlabel("Family"); ax.set_ylabel("Family")

    fig.tight_layout()
    save(fig, "prototype_distance_heatmap")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(device)

    print("Computing embeddings...")
    X, y, family_names = get_embeddings(model, device)

    print("\n[1/3] PCA figure...")
    fig_pca(X, y, family_names)

    print("[2/3] UMAP figure...")
    fig_umap(X, y, family_names)

    print("[3/3] Prototype distance heatmap...")
    fig_proto_heatmap(X, y, family_names)

    print("\nDone.")

if __name__ == "__main__":
    main()
