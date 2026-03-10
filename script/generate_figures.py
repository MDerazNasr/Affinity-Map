"""
script/generate_figures.py

Generates all publication-quality matplotlib figures from experiment results.

Outputs (all in results/):
  fig_episode_hist.pdf/.png      — real per-episode accuracy distribution
  fig_kshot_sweep.pdf/.png       — accuracy vs K-shot curve with baselines
  fig_named_confusion.pdf/.png   — named family confusion matrix heatmap
  fig_learning_curves.pdf/.png   — train/val accuracy & loss per epoch

Run from project root:
    python script/generate_figures.py
"""

import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

RESULTS = "results"
FIG_DPI = 200

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "savefig.dpi": FIG_DPI,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(RESULTS, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved {path}")
    plt.close(fig)

# ── 1. Per-episode accuracy histogram ────────────────────────────────────────

def fig_episode_hist():
    path = os.path.join(RESULTS, "episode_accuracies.json")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return
    with open(path) as f:
        data = json.load(f)
    accs = np.array(data["per_episode"])
    mu, sigma = accs.mean(), accs.std()
    N_ep = len(accs)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    n, bins, patches = ax.hist(accs, bins=25, color="#4C72B0", edgecolor="white",
                                linewidth=0.5, density=False, alpha=0.85)
    ax.axvline(mu, color="#DD4444", lw=1.8, linestyle="--", label=f"Mean = {mu:.3f}")
    ax.axvline(mu - sigma, color="#DD4444", lw=1.0, linestyle=":", alpha=0.7,
               label=f"±1σ = {sigma:.3f}")
    ax.axvline(mu + sigma, color="#DD4444", lw=1.0, linestyle=":", alpha=0.7)
    ax.set_xlabel("Episode accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of 5-way 5-shot episode accuracy  "
                 f"({N_ep} episodes, cosine metric)")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    fig.tight_layout()
    save(fig, "fig_episode_hist")

# ── 2. K-shot sweep ───────────────────────────────────────────────────────────

def fig_kshot_sweep():
    sweep_path = os.path.join(RESULTS, "kshot_sweep.json")
    kmer_path  = os.path.join(RESULTS, "kmer_baseline.json")
    if not os.path.exists(sweep_path):
        print(f"[skip] {sweep_path} not found"); return

    with open(sweep_path) as f:
        sweep = json.load(f)

    # CNN ProtoNet
    cnn_ks  = sorted([int(k) for k in sweep.keys()])
    cnn_mu  = [sweep[str(k)]["mean"]  for k in cnn_ks]
    cnn_sig = [sweep[str(k)]["std"]   for k in cnn_ks]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Shaded ±1σ band for CNN
    ax.fill_between(cnn_ks,
                    [m - s for m, s in zip(cnn_mu, cnn_sig)],
                    [m + s for m, s in zip(cnn_mu, cnn_sig)],
                    alpha=0.15, color="#4C72B0")
    ax.plot(cnn_ks, cnn_mu, "o-", color="#4C72B0", lw=2.0,
            markersize=6, label="CNN ProtoNet (ours)")

    # k-mer baselines
    if os.path.exists(kmer_path):
        with open(kmer_path) as f:
            kmer = json.load(f)
        for key, label, color, marker in [
            ("kmer_1nn",     "k-mer 1-NN",      "#E8763A", "s--"),
            ("kmer_protonet","k-mer ProtoNet",   "#59A14F", "^--"),
        ]:
            if key in kmer:
                bdata = kmer[key]
                ks  = sorted([int(k) for k in bdata.keys()])
                mus = [bdata[str(k)]["mean"] for k in ks]
                sigs= [bdata[str(k)]["std"]  for k in ks]
                ax.fill_between(ks,
                                [m-s for m,s in zip(mus,sigs)],
                                [m+s for m,s in zip(mus,sigs)],
                                alpha=0.10, color=color)
                ax.plot(ks, mus, marker, color=color, lw=1.5,
                        markersize=5, label=label)

    # ESM-2 ProtoNet baseline
    esm2_path = os.path.join(RESULTS, "esm2_results.json")
    if os.path.exists(esm2_path):
        with open(esm2_path) as f:
            esm2 = json.load(f)
        ks   = sorted([int(k) for k in esm2.keys()])
        mus  = [esm2[str(k)]["esm2_mean"] for k in ks]
        sigs = [esm2[str(k)]["esm2_std"]  for k in ks]
        ax.fill_between(ks, [m-s for m,s in zip(mus,sigs)],
                            [m+s for m,s in zip(mus,sigs)],
                        alpha=0.10, color="#E84393")
        ax.plot(ks, mus, "D-", color="#E84393", lw=2.0,
                markersize=6, label="ESM-2 ProtoNet (frozen)")

    # BLAST baselines
    blast_path = os.path.join(RESULTS, "blast_baseline.json")
    if os.path.exists(blast_path):
        with open(blast_path) as f:
            blast = json.load(f)
        for key, label, color, marker in [
            ("blast_1nn",     "BLAST 1-NN",      "#9B59B6", "D:"),
            ("blast_protonet","BLAST ProtoNet",   "#C0392B", "v:"),
        ]:
            if key in blast:
                bdata = blast[key]
                ks  = sorted([int(k) for k in bdata.keys()])
                mus = [bdata[str(k)]["mean"] for k in ks]
                sigs= [bdata[str(k)]["std"]  for k in ks]
                ax.fill_between(ks,
                                [m-s for m,s in zip(mus,sigs)],
                                [m+s for m,s in zip(mus,sigs)],
                                alpha=0.08, color=color)
                ax.plot(ks, mus, marker, color=color, lw=1.5,
                        markersize=5, label=label)

    # Random baseline
    random_acc = 1.0 / 5
    ax.axhline(random_acc, color="gray", lw=1.2, linestyle=":",
               label=f"Random ({random_acc:.0%})")

    ax.set_xlabel("Number of support examples per class  (K)")
    ax.set_ylabel("5-way classification accuracy")
    ax.set_title("Few-shot accuracy vs. number of shots  (5-way, cosine metric)")
    ax.set_xticks(cnn_ks)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_kshot_sweep")

# ── 3. Named confusion matrix ─────────────────────────────────────────────────

def fig_named_confusion():
    path = os.path.join(RESULTS, "named_confusion.json")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return
    with open(path) as f:
        data = json.load(f)
    cm = np.array(data["confusion_matrix"])
    names = data["family_names"]

    # Shorten names for readability
    short = [n.replace("Endopeptidase","Endopep.").replace("Phospho","Phos.")
              .replace("CarlavirusCoat","Coat").replace("Outer","Out.")
              .replace("Membrane","Memb.").replace("Finger","Fing.")
              .replace("Retroviral","Retrovir.").replace("ZincFing.AN1","ZnFingAN1")
              for n in names]

    n = len(names)
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    # Blue-white colormap
    cmap = LinearSegmentedColormap.from_list("bw", ["white", "#2166AC"])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Fraction predicted")

    # Annotate cells
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0.01:
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(5, 8 - n // 4), color=color)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=max(6, 9 - n // 5))
    ax.set_yticklabels(short, fontsize=max(6, 9 - n // 5))
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(f"Named family confusion matrix  ({data['n_episodes']} episodes, 5-way 5-shot)")

    # Highlight diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="#DD4444",
                                   linewidth=1.5, clip_on=False))
    fig.tight_layout()
    save(fig, "fig_named_confusion")

# ── 4. Learning curves ────────────────────────────────────────────────────────

def fig_learning_curves():
    path = os.path.join(RESULTS, "learning_curves.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return

    epochs, train_loss, train_acc, val_acc = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Accuracy
    ax1.plot(epochs, train_acc, "o-", color="#4C72B0", lw=2.0, markersize=5,
             label="Train accuracy")
    ax1.plot(epochs, val_acc, "s--", color="#DD4444", lw=2.0, markersize=5,
             label="Val accuracy (unseen families)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Episode accuracy")
    ax1.set_title("Training and validation accuracy")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(epochs)
    ax1.legend(frameon=False)
    ax1.grid(axis="y", alpha=0.3)

    # Loss
    ax2.plot(epochs, train_loss, "o-", color="#59A14F", lw=2.0, markersize=5,
             label="Train loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title("Training loss")
    ax2.set_xticks(epochs)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Learning curves — Prototypical Network, 5-way 5-shot episodic training",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "fig_learning_curves")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("Generating figures...\n")
    fig_episode_hist()
    fig_kshot_sweep()
    fig_named_confusion()
    fig_learning_curves()
    print("\nDone.")

if __name__ == "__main__":
    main()
