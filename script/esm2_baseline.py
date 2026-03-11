"""
script/esm2_baseline.py

Frozen ESM-2 (esm2_t6_8M_UR50D, 8M params, 320-dim) ProtoNet baseline.

Strategy: pre-cache all ESM-2 embeddings in one pass, then replay the same
matched episodes as significance_tests.py using the cached vectors.  This is
orders of magnitude faster than running ESM-2 per-episode.

Outputs:
  results/esm2_results.json   -- K-shot accuracy + Wilcoxon vs CNN/k-mer
  results/esm2_embeddings.npy -- cached embeddings (fam -> array)

Run from project root:
    python script/esm2_baseline.py
"""

import os, sys, json, random, itertools
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import esm as esm_lib
from models.encoder import ProteinEncoderCNN
from data.configs.protonet import CONF
from utils.splits import get_splits

PROTEINS_JSON  = "data/processed/proteins.json"
CNN_CHECKPOINT = "checkpoints/best_protonet.pt"
CACHE_PATH     = "results/esm2_embeddings.npy"
RESULTS        = "results"

N_EPISODES = 1000
K_VALUES   = [1, 2, 5, 10, 20]
N_WAY      = 5
Q_QUERIES  = 10
BATCH_SIZE   = 16        # sequences per ESM-2 forward pass
ESM_LAYER    = 6         # last transformer layer of esm2_t6_8M
ESM_MAX_LEN  = 512       # truncate for ESM-2 to avoid OOM (3k-residue seqs)
RANDOM_SEED = 42

AA_ORDER  = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
AA_SET    = set(AA_ORDER)
PAD_LEN   = 400

# ── k-mer ─────────────────────────────────────────────────────────────────────

def build_kmer_vocab(k=3):
    return {"".join(t): i
            for i, t in enumerate(itertools.product(list(AA_ORDER), repeat=k))}

KMER_VOCAB = build_kmer_vocab(3)
KMER_DIM   = len(KMER_VOCAB)

def kmer_vector(seq):
    v = np.zeros(KMER_DIM, dtype=np.float64)
    for i in range(len(seq) - 2):
        km = seq[i:i+3]
        if km in KMER_VOCAB:
            v[KMER_VOCAB[km]] += 1
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)

# ── CNN ───────────────────────────────────────────────────────────────────────

def encode_seq_cnn(seq):
    tokens = [AA_TO_IDX.get(aa, 0) for aa in seq[:PAD_LEN]]
    tokens += [0] * (PAD_LEN - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

def load_cnn(device):
    ckpt  = torch.load(CNN_CHECKPOINT, map_location=device)
    cfg   = ckpt.get("config", CONF)
    model = ProteinEncoderCNN(proj_dim=cfg["proj_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def cnn_embed_all(fam_seqs, model, device):
    """Returns dict: family -> (N, 128) float64 array."""
    result = {}
    with torch.no_grad():
        for fam, seqs in fam_seqs.items():
            batches = []
            for i in range(0, len(seqs), BATCH_SIZE):
                batch = torch.stack([encode_seq_cnn(s) for s in seqs[i:i+BATCH_SIZE]]).to(device)
                z = model(batch).cpu().numpy().astype(np.float64)
                batches.append(z)
            result[fam] = np.concatenate(batches, axis=0)
    return result

# ── ESM-2 ─────────────────────────────────────────────────────────────────────

def load_esm2(device):
    print("  Loading ESM-2 (esm2_t6_8M_UR50D)...", flush=True)
    model, alphabet = esm_lib.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def esm2_embed_batch(seqs, model, batch_converter, device):
    # truncate each sequence so attention stays within MPS memory budget
    seqs_trunc = [s[:ESM_MAX_LEN] for s in seqs]
    data = [("s", s) for s in seqs_trunc]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)
    reps = out["representations"][ESM_LAYER]   # (B, L+2, D)
    embs = []
    for i, s in enumerate(seqs_trunc):
        L = min(len(s), tokens.shape[1] - 2)
        embs.append(reps[i, 1:L+1, :].mean(0).float().cpu().numpy())
    return np.stack(embs).astype(np.float64)   # (B, D)

def esm2_embed_all(fam_seqs, model, batch_converter, device):
    """Returns dict: family -> (N, D) float64 array, L2-normalised."""
    result = {}
    families = list(fam_seqs.keys())
    total = sum(len(v) for v in fam_seqs.values())
    done = 0
    for fam in families:
        seqs = fam_seqs[fam]
        batches = []
        for i in range(0, len(seqs), BATCH_SIZE):
            b = esm2_embed_batch(seqs[i:i+BATCH_SIZE], model, batch_converter, device)
            batches.append(b)
            done += len(seqs[i:i+BATCH_SIZE])
        arr = np.concatenate(batches, axis=0)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        result[fam] = arr / (norms + 1e-8)
    print(f"  Embedded {done} sequences across {len(families)} families", flush=True)
    return result

# ── episode runner ─────────────────────────────────────────────────────────────

def l2norm(arr):
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (n + 1e-8)

def proto_acc(emb_sup, emb_qry, sup_labels, qry_labels):
    protos = np.stack([emb_sup[sup_labels == c].mean(0)
                       for c in range(N_WAY)])
    protos = l2norm(protos)
    sims = l2norm(emb_qry) @ protos.T
    return (sims.argmax(1) == qry_labels).mean()

def run_matched_episodes(fam_seqs, eligible, cnn_cache, kmer_cache, esm2_cache, K):
    cnn_accs, km_accs, esm_accs = [], [], []
    for ep in range(N_EPISODES):
        chosen = random.sample(eligible, N_WAY)
        sup_idx, qry_idx, sup_lbl, qry_lbl = {}, {}, [], []
        for lbl, fam in enumerate(chosen):
            n = len(fam_seqs[fam])
            idxs = random.sample(range(n), K + Q_QUERIES)
            sup_idx[fam] = idxs[:K]
            qry_idx[fam] = idxs[K:]
            sup_lbl.extend([lbl] * K)
            qry_lbl.extend([lbl] * Q_QUERIES)
        sy = np.array(sup_lbl)
        qy = np.array(qry_lbl)

        # gather cached embeddings
        cnn_s = np.concatenate([cnn_cache[f][sup_idx[f]] for f in chosen])
        cnn_q = np.concatenate([cnn_cache[f][qry_idx[f]] for f in chosen])
        km_s  = np.concatenate([kmer_cache[f][sup_idx[f]] for f in chosen])
        km_q  = np.concatenate([kmer_cache[f][qry_idx[f]] for f in chosen])
        esm_s = np.concatenate([esm2_cache[f][sup_idx[f]] for f in chosen])
        esm_q = np.concatenate([esm2_cache[f][qry_idx[f]] for f in chosen])

        cnn_accs.append(proto_acc(cnn_s, cnn_q, sy, qy))
        km_accs.append(proto_acc(km_s,  km_q,  sy, qy))
        esm_accs.append(proto_acc(esm_s, esm_q, sy, qy))

    return np.array(cnn_accs), np.array(km_accs), np.array(esm_accs)

# ── stats ──────────────────────────────────────────────────────────────────────

def wilcoxon(a, b):
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    n = len(a)
    r = 1 - (2 * stat) / (n * (n + 1) / 2)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return float(p), float(r), sig

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load sequences
    with open(PROTEINS_JSON) as f:
        raw = json.load(f)
    all_fam_seqs = {fam: [s for s in seqs
                          if isinstance(s, str) and all(a in AA_SET for a in s)
                          and 10 <= len(s) <= 3000]
                   for fam, seqs in raw.items()}
    all_fam_seqs = {f: s for f, s in all_fam_seqs.items() if s}
    # Use test families only
    _, _, fam_seqs = get_splits(all_fam_seqs)
    print(f"  {len(all_fam_seqs)} total families → using {len(fam_seqs)} test families, "
          f"{sum(len(v) for v in fam_seqs.values())} sequences\n")

    # ── cache CNN embeddings ──────────────────────────────────────────────────
    print("[1/3] Caching CNN embeddings...", flush=True)
    cnn_model  = load_cnn(device)
    cnn_cache  = cnn_embed_all(fam_seqs, cnn_model, device)
    print(f"  CNN done ({sum(v.shape[0] for v in cnn_cache.values())} vectors)")

    # ── cache k-mer embeddings ────────────────────────────────────────────────
    print("[2/3] Caching k-mer embeddings...", flush=True)
    kmer_cache = {fam: np.stack([kmer_vector(s) for s in seqs])
                  for fam, seqs in fam_seqs.items()}
    print(f"  k-mer done")

    # ── cache ESM-2 embeddings ────────────────────────────────────────────────
    print("[3/3] Caching ESM-2 embeddings...", flush=True)
    esm_model, batch_converter = load_esm2(device)
    esm2_cache = esm2_embed_all(fam_seqs, esm_model, batch_converter, device)
    print(f"  ESM-2 done\n")

    # ── run matched episodes ──────────────────────────────────────────────────
    results = {}
    print(f"{'K':>3}  {'CNN':>7}  {'k-mer':>7}  {'ESM-2':>7}  "
          f"{'ESM2-CNN':>10}  {'ESM2-kmer':>10}")
    print("-" * 58)

    for K in K_VALUES:
        min_seq   = K + Q_QUERIES
        eligible  = [f for f, s in fam_seqs.items() if len(s) >= min_seq]
        if len(eligible) < N_WAY:
            print(f"K={K}: only {len(eligible)} eligible families, skip"); continue

        print(f"K={K}: {N_EPISODES} matched episodes ({len(eligible)} fam)...", flush=True)
        cnn_a, km_a, esm_a = run_matched_episodes(
            fam_seqs, eligible, cnn_cache, kmer_cache, esm2_cache, K)

        p_ec, r_ec, sig_ec = wilcoxon(esm_a, cnn_a)
        p_ek, r_ek, sig_ek = wilcoxon(esm_a, km_a)

        results[str(K)] = {
            "cnn_mean":  float(cnn_a.mean()), "cnn_std":  float(cnn_a.std()),
            "kmer_mean": float(km_a.mean()),  "kmer_std": float(km_a.std()),
            "esm2_mean": float(esm_a.mean()), "esm2_std": float(esm_a.std()),
            "esm2_vs_cnn":  {"p": p_ec, "r": r_ec, "sig": sig_ec,
                             "diff": float(esm_a.mean() - cnn_a.mean())},
            "esm2_vs_kmer": {"p": p_ek, "r": r_ek, "sig": sig_ek,
                             "diff": float(esm_a.mean() - km_a.mean())},
            "n_episodes": N_EPISODES, "n_eligible": len(eligible),
        }

        print(f"  {K:>3}  {cnn_a.mean():>7.4f}  {km_a.mean():>7.4f}  "
              f"{esm_a.mean():>7.4f}  "
              f"{esm_a.mean()-cnn_a.mean():>+8.4f}{sig_ec:>3}  "
              f"{esm_a.mean()-km_a.mean():>+8.4f}{sig_ek:>3}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/esm2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS}/esm2_results.json")

    print("\n── LaTeX rows ──")
    for K, r in results.items():
        sig_c, sig_k = r["esm2_vs_cnn"]["sig"], r["esm2_vs_kmer"]["sig"]
        print(f"ESM-2 ProtoNet & "
              f"{r['esm2_mean']*100:.1f}$\\pm${r['esm2_std']*100:.1f} & "
              f"{r['esm2_vs_cnn']['diff']*100:+.1f} pp ({sig_c}) & "
              f"{r['esm2_vs_kmer']['diff']*100:+.1f} pp ({sig_k}) \\\\  % K={K}")

if __name__ == "__main__":
    main()
