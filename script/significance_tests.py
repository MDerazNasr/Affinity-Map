"""
script/significance_tests.py

Runs matched episodic evaluation for CNN ProtoNet vs k-mer ProtoNet,
then performs paired Wilcoxon signed-rank tests across all K values.

Outputs:
  results/significance_tests.json  — p-values, effect sizes, per-episode arrays
  prints a summary table for the paper

Run from project root:
    python script/significance_tests.py
"""

import os, sys, json, random
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.encoder import ProteinEncoderCNN
from data.configs.protonet import CONF
from utils.splits import get_splits

PROTEINS_JSON = "data/processed/proteins.json"
CHECKPOINT    = "checkpoints/best_protonet.pt"
RESULTS       = "results"
N_EPISODES    = 1000   # matched episodes per K value
K_VALUES      = [1, 2, 5, 10, 20]
N_WAY         = 5
Q_QUERIES     = 10
PAD_LEN       = 400
RANDOM_SEED   = 42

AA_ORDER  = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
AA_SET    = set(AA_ORDER)

# ── k-mer utilities ───────────────────────────────────────────────────────────

def build_kmer_vocab(k=3):
    import itertools
    aas = list(AA_ORDER)
    return {"".join(t): i for i, t in enumerate(itertools.product(aas, repeat=k))}

KMER_VOCAB = build_kmer_vocab(3)
KMER_DIM   = len(KMER_VOCAB)  # 8000

def kmer_vector(seq):
    v = np.zeros(KMER_DIM, dtype=np.float32)
    for i in range(len(seq) - 2):
        km = seq[i:i+3]
        if km in KMER_VOCAB:
            v[KMER_VOCAB[km]] += 1
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)

# ── CNN utilities ─────────────────────────────────────────────────────────────

def encode_seq(seq):
    tokens = [AA_TO_IDX.get(aa, 0) for aa in seq[:PAD_LEN]]
    tokens += [0] * (PAD_LEN - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

def load_model(device):
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    cfg   = ckpt.get("config", CONF)
    model = ProteinEncoderCNN(proj_dim=cfg["proj_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg

# ── episode runner ────────────────────────────────────────────────────────────

def run_matched_episodes(fam_seqs, eligible, model, device, K, n_episodes):
    """
    Run n_episodes matched episodes; return (cnn_accs, kmer_accs) arrays.
    Each episode uses the SAME sampled families and sequences for both methods.
    """
    cnn_accs, kmer_accs = [], []

    with torch.no_grad():
        for ep in range(n_episodes):
            chosen = random.sample(eligible, N_WAY)

            sup_seqs, sup_labels = [], []
            qry_seqs, qry_labels = [], []

            for lbl, fam in enumerate(chosen):
                seqs = fam_seqs[fam]
                idxs = random.sample(range(len(seqs)), K + Q_QUERIES)
                for i in idxs[:K]:
                    sup_seqs.append(seqs[i]); sup_labels.append(lbl)
                for i in idxs[K:]:
                    qry_seqs.append(seqs[i]); qry_labels.append(lbl)

            sy = np.array(sup_labels)
            qy = np.array(qry_labels)

            # ── CNN ProtoNet ─────────────────────────────────────────────────
            sx_t = torch.stack([encode_seq(s) for s in sup_seqs]).to(device)
            qx_t = torch.stack([encode_seq(s) for s in qry_seqs]).to(device)
            z_s  = model(sx_t).cpu().numpy()   # (N*K, D)
            z_q  = model(qx_t).cpu().numpy()   # (N*Q, D)

            protos_cnn = np.stack([z_s[sy == c].mean(0) for c in range(N_WAY)])
            norms = np.linalg.norm(protos_cnn, axis=1, keepdims=True)
            protos_cnn /= (norms + 1e-8)
            sims = z_q @ protos_cnn.T          # (NQ, N)
            preds_cnn = sims.argmax(1)
            cnn_accs.append((preds_cnn == qy).mean())

            # ── k-mer ProtoNet ───────────────────────────────────────────────
            ks_s = np.stack([kmer_vector(s) for s in sup_seqs]).astype(np.float64)
            ks_q = np.stack([kmer_vector(s) for s in qry_seqs]).astype(np.float64)

            protos_km = np.stack([ks_s[sy == c].mean(0) for c in range(N_WAY)])
            nk = np.linalg.norm(protos_km, axis=1, keepdims=True)
            protos_km /= (nk + 1e-8)
            nq = np.linalg.norm(ks_q, axis=1, keepdims=True)
            ks_q_n = ks_q / (nq + 1e-8)
            sims_km = ks_q_n @ protos_km.T
            preds_km = sims_km.argmax(1)
            kmer_accs.append((preds_km == qy).mean())

    return np.array(cnn_accs), np.array(kmer_accs)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model and sequences...")
    model, cfg = load_model(device)
    with open(PROTEINS_JSON) as f:
        raw = json.load(f)

    # Clean sequences
    all_fam_seqs = {}
    for fam, seqs in raw.items():
        clean = [s for s in seqs if isinstance(s, str) and
                 all(aa in AA_SET for aa in s) and 10 <= len(s) <= 3000]
        if clean:
            all_fam_seqs[fam] = clean

    # Use test families only
    _, _, fam_seqs = get_splits(all_fam_seqs)
    print(f"  {len(all_fam_seqs)} total families → using {len(fam_seqs)} test families\n")

    results = {}

    print(f"{'K':>4} {'CNN':>8} {'k-mer':>8}  {'diff':>7}  {'p-value':>10}  {'sig':>4}")
    print("-" * 55)

    for K in K_VALUES:
        min_seqs = K + Q_QUERIES
        eligible = [f for f, s in fam_seqs.items() if len(s) >= min_seqs]
        if len(eligible) < N_WAY:
            print(f"K={K}: only {len(eligible)} eligible families, skipping")
            continue

        print(f"K={K}: running {N_EPISODES} matched episodes...", flush=True)
        cnn_a, km_a = run_matched_episodes(
            fam_seqs, eligible, model, device, K, N_EPISODES)

        # Paired Wilcoxon signed-rank test (non-parametric)
        stat, p = stats.wilcoxon(cnn_a, km_a, alternative="two-sided")

        # Effect size: rank-biserial correlation
        n = len(cnn_a)
        r_effect = 1 - (2 * stat) / (n * (n + 1) / 2)

        diff = cnn_a.mean() - km_a.mean()
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

        results[K] = {
            "cnn_mean":   float(cnn_a.mean()),
            "cnn_std":    float(cnn_a.std()),
            "kmer_mean":  float(km_a.mean()),
            "kmer_std":   float(km_a.std()),
            "mean_diff":  float(diff),
            "wilcoxon_stat": float(stat),
            "p_value":    float(p),
            "effect_r":   float(r_effect),
            "significant": sig,
            "n_episodes": N_EPISODES,
            "n_eligible": len(eligible),
        }

        print(f"{K:>4} {cnn_a.mean():>8.4f} {km_a.mean():>8.4f}  "
              f"{diff:>+7.4f}  {p:>10.2e}  {sig:>4}")

    # Save
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "significance_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # LaTeX table snippet
    print("\n── LaTeX table rows (CNN vs k-mer ProtoNet) ──")
    print(r"\midrule")
    for K, r in results.items():
        sig = r["significant"]
        print(f"  K={K} & {r['cnn_mean']*100:.1f}\\% & {r['kmer_mean']*100:.1f}\\% "
              f"& {r['mean_diff']*100:+.1f} pp & {r['p_value']:.2e} & {sig} \\\\")

if __name__ == "__main__":
    main()
