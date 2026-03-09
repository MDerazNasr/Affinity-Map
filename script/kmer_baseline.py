"""
script/kmer_baseline.py

Implements two alignment-free baselines for few-shot protein family classification,
using the same 5-way 5-shot episodic protocol as the ProtoNet:

  Baseline 1 — k-mer 1-NN:
      For each query, find the most similar support sequence by cosine
      similarity of their 3-mer frequency vectors. Classify as that support's family.

  Baseline 2 — k-mer ProtoNet:
      Compute class prototypes as mean 3-mer vectors, classify each query by
      nearest prototype (cosine). Same protocol as our CNN model, but with
      hand-crafted features instead of learned embeddings.

Results saved to results/kmer_baseline.json

Run from project root:
    python script/kmer_baseline.py
"""

import os, sys, json, random, itertools
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.configs.protonet import CONF
from utils.episodes import loaded_encoded_families

# ── k-mer feature extraction ──────────────────────────────────────────────────

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)

def build_kmer_vocab(k: int) -> dict:
    """Build index for all k-mers over canonical amino acids."""
    kmers = ["".join(p) for p in itertools.product(AA, repeat=k)]
    return {km: i for i, km in enumerate(kmers)}

KMER_VOCAB_3 = build_kmer_vocab(3)
KMER_DIM_3 = len(KMER_VOCAB_3)  # 20^3 = 8000

def seq_to_kmer_vector(seq: str, k: int = 3, vocab: dict = None) -> np.ndarray:
    """Compute normalised k-mer frequency vector for an amino-acid string."""
    if vocab is None:
        vocab = KMER_VOCAB_3
    dim = len(vocab)
    vec = np.zeros(dim, dtype=np.float32)
    n_valid = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in vocab:
            vec[vocab[kmer]] += 1
            n_valid += 1
    if n_valid > 0:
        vec /= n_valid  # relative frequency
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm  # L2-normalise so cosine = dot product
    return vec

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (m, D), B: (n, D) → (m, n) cosine similarity (both already L2-normalised)."""
    return A @ B.T

# ── load raw sequences from proteins.json ────────────────────────────────────

def load_raw_sequences(proteins_path: str) -> dict:
    """Returns {family_name: [seq_str, ...]}"""
    with open(proteins_path) as f:
        data = json.load(f)
    return data

# ── episodic evaluation ───────────────────────────────────────────────────────

def run_kmer_episode(fam_seqs: dict, eligible: list, N: int, K: int, Q: int,
                     method: str = "protonet") -> float:
    """
    Sample one N-way K-shot episode from raw sequences, compute k-mer features,
    then classify queries using either 1-NN or prototype-nearest.

    method: 'protonet' | '1nn'
    """
    chosen = random.sample(eligible, N)

    support_vecs, support_labels = [], []
    query_vecs, query_labels = [], []

    for lbl, fname in enumerate(chosen):
        seqs = fam_seqs[fname]
        idxs = random.sample(range(len(seqs)), K + Q)
        for i in idxs[:K]:
            support_vecs.append(seq_to_kmer_vector(seqs[i]))
            support_labels.append(lbl)
        for i in idxs[K:]:
            query_vecs.append(seq_to_kmer_vector(seqs[i]))
            query_labels.append(lbl)

    S = np.stack(support_vecs)   # (N*K, D)
    Q_mat = np.stack(query_vecs) # (N*Q, D)
    sy = np.array(support_labels)
    qy = np.array(query_labels)

    if method == "protonet":
        # Prototype = mean of support k-mer vectors per class
        protos = np.stack([S[sy == c].mean(0) for c in range(N)])  # (N, D)
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        norms[norms == 0] = 1
        protos /= norms
        logits = cosine_sim_matrix(Q_mat, protos)  # (N*Q, N)
        preds = logits.argmax(1)

    elif method == "1nn":
        # 1-NN: assign query to the class of its single most similar support
        sims = cosine_sim_matrix(Q_mat, S)  # (N*Q, N*K)
        nn_idx = sims.argmax(1)
        preds = sy[nn_idx]

    else:
        raise ValueError(f"Unknown method: {method}")

    return float((preds == qy).mean())

def evaluate_baseline(fam_seqs: dict, conf: dict, method: str,
                       k_values=None, n_episodes: int = 300):
    """Run baseline for multiple K values."""
    if k_values is None:
        k_values = [1, 2, 5, 10, 20]
    N, Q = conf["N"], conf["Q"]
    results = {}
    for K in k_values:
        min_seqs = K + Q
        eligible = [fname for fname, seqs in fam_seqs.items() if len(seqs) >= min_seqs]
        if len(eligible) < N:
            print(f"  [{method}] K={K}: only {len(eligible)} eligible families, skipping")
            continue
        accs = [run_kmer_episode(fam_seqs, eligible, N, K, Q, method=method)
                for _ in range(n_episodes)]
        results[K] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "n_eligible_families": len(eligible),
            "n_episodes": n_episodes,
        }
        print(f"  [{method}] K={K:2d}: {results[K]['mean']:.4f} ± {results[K]['std']:.4f}  "
              f"({len(eligible)} families)")
    return results

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    conf = CONF.copy()
    proteins_path = "data/processed/proteins.json"
    if not os.path.exists(proteins_path):
        raise FileNotFoundError(f"Not found: {proteins_path}")

    print("Loading raw sequences...")
    raw = load_raw_sequences(proteins_path)
    # Keep only families with unique amino-acid sequences (filter empties)
    fam_seqs = {
        fname: [s for s in seqs if len(s) >= 10 and all(aa in AA_SET for aa in s)]
        for fname, seqs in raw.items()
        if isinstance(seqs, list) and len(seqs) > 0
    }
    print(f"Loaded {len(fam_seqs)} families with cleaned sequences.")

    k_values = [1, 2, 5, 10, 20]
    n_episodes = 300

    print("\nRunning k-mer 1-NN baseline...")
    knn_results = evaluate_baseline(fam_seqs, conf, method="1nn",
                                    k_values=k_values, n_episodes=n_episodes)

    print("\nRunning k-mer ProtoNet baseline...")
    proto_results = evaluate_baseline(fam_seqs, conf, method="protonet",
                                      k_values=k_values, n_episodes=n_episodes)

    # Random baseline: 1/N for each K
    random_acc = 1.0 / conf["N"]

    out = {
        "random_baseline": random_acc,
        "kmer_1nn": knn_results,
        "kmer_protonet": proto_results,
        "k_values": k_values,
        "N": conf["N"],
        "Q": conf["Q"],
        "n_episodes": n_episodes,
        "k_mer_size": 3,
    }
    path = "results/kmer_baseline.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")

    # Print summary comparison at K=5
    print("\n── Summary at K=5, 5-way ──────────────────────────────────────────")
    print(f"  Random baseline:    {random_acc:.4f}")
    if 5 in knn_results:
        r = knn_results[5]
        print(f"  k-mer 1-NN:         {r['mean']:.4f} ± {r['std']:.4f}")
    if 5 in proto_results:
        r = proto_results[5]
        print(f"  k-mer ProtoNet:     {r['mean']:.4f} ± {r['std']:.4f}")
    print(f"  CNN ProtoNet:       0.9125 ± 0.0794  (from eval_summary.json)")

if __name__ == "__main__":
    main()
