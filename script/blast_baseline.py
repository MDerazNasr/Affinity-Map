"""
script/blast_baseline.py

BLAST 1-NN baseline for few-shot protein family classification.

Strategy (efficient):
  1. Write all sequences to a single FASTA file.
  2. makeblastdb on that file (protein database).
  3. blastp all sequences against the database → all-vs-all score matrix.
  4. Cache the score matrix to results/blast_scores.npz so it only runs once.
  5. Run the same N-way K-shot episodic protocol used for CNN ProtoNet and
     k-mer baselines, using pre-cached BLAST bit-scores for classification.

Two classifiers on top of BLAST scores:
  - BLAST 1-NN:      query → class of its highest-scoring support sequence.
  - BLAST ProtoNet:  prototype = mean BLAST-score vector over support;
                     query → prototype with highest mean score.

Run from project root:
    python script/blast_baseline.py
"""

import os, sys, json, random, subprocess, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.configs.protonet import CONF

BLASTP   = "blastp"
MAKEDB   = "makeblastdb"
CACHE    = "results/blast_scores.npz"
PROTEINS = "data/processed/proteins.json"
AA_SET   = set("ACDEFGHIKLMNPQRSTVWY")

# ── sequence loading ──────────────────────────────────────────────────────────

def load_sequences(path: str):
    """Returns {family: [seq, ...]}, only canonical AAs, length 10–3000."""
    with open(path) as f:
        raw = json.load(f)
    fam_seqs = {}
    for fam, seqs in raw.items():
        if not isinstance(seqs, list):
            continue
        clean = [s for s in seqs
                 if isinstance(s, str) and 10 <= len(s) <= 3000
                 and all(aa in AA_SET for aa in s)]
        if clean:
            fam_seqs[fam] = clean
    return fam_seqs

# ── FASTA helpers ─────────────────────────────────────────────────────────────

def write_fasta(seq_records: list, path: str):
    """seq_records: [(id_str, seq_str), ...]"""
    with open(path, "w") as f:
        for sid, seq in seq_records:
            f.write(f">{sid}\n{seq}\n")

def seq_id(family: str, idx: int) -> str:
    return f"{family}___{idx}"

def parse_family_idx(sid: str):
    parts = sid.split("___")
    return parts[0], int(parts[1])

# ── build score matrix ────────────────────────────────────────────────────────

def build_blast_score_matrix(fam_seqs: dict, cache_path: str = CACHE):
    """
    Run all-vs-all BLASTP and return a square bit-score matrix.

    Returns:
        ids      : list of seq_id strings (index → identity)
        scores   : np.ndarray (N, N) of bit-scores (0 where no hit)
    """
    if os.path.exists(cache_path):
        print(f"Loading cached BLAST scores from {cache_path} ...")
        data = np.load(cache_path, allow_pickle=True)
        ids    = list(data["ids"])
        scores = data["scores"]
        print(f"  {len(ids)} sequences loaded from cache.")
        return ids, scores

    print("Building BLAST score matrix (this runs once, then cached)...")

    # Collect all sequences with stable IDs
    records = []
    for fam, seqs in sorted(fam_seqs.items()):
        for i, seq in enumerate(seqs):
            records.append((seq_id(fam, i), seq))
    ids = [r[0] for r in records]
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    N = len(ids)
    print(f"  Total sequences: {N}")

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "all.fasta")
        db_path    = os.path.join(tmpdir, "blastdb")
        out_path   = os.path.join(tmpdir, "blast_out.tsv")

        write_fasta(records, fasta_path)

        # Build BLAST database
        print("  Running makeblastdb ...")
        subprocess.run(
            [MAKEDB, "-in", fasta_path, "-dbtype", "prot", "-out", db_path],
            check=True, capture_output=True
        )

        # Run all-vs-all blastp
        # outfmt 6: qseqid sseqid bitscore
        print(f"  Running blastp ({N} queries vs {N} subjects) ...")
        subprocess.run(
            [BLASTP,
             "-query",  fasta_path,
             "-db",     db_path,
             "-out",    out_path,
             "-outfmt", "6 qseqid sseqid bitscore",
             "-evalue", "10",        # permissive: capture even weak hits
             "-num_threads", "4",
             "-max_hsps", "1",       # one HSP per query-subject pair
             "-max_target_seqs", str(N)],
            check=True, capture_output=True
        )

        # Parse results into score matrix
        scores = np.zeros((N, N), dtype=np.float32)
        with open(out_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                q, s, bits = parts[0], parts[1], parts[2]
                if q in id_to_idx and s in id_to_idx:
                    qi, si = id_to_idx[q], id_to_idx[s]
                    scores[qi, si] = max(scores[qi, si], float(bits))

    # Self-hits are not useful for classification; zero the diagonal
    np.fill_diagonal(scores, 0.0)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, ids=np.array(ids), scores=scores)
    print(f"  Saved to {cache_path}")
    return ids, scores

# ── episodic evaluation ───────────────────────────────────────────────────────

def run_blast_episode(fam_seqs: dict, eligible: list,
                      id_to_idx: dict, scores: np.ndarray,
                      N: int, K: int, Q: int,
                      method: str = "1nn") -> float:
    """
    Sample one N-way K-shot episode and classify using BLAST scores.
    method: '1nn' | 'protonet'
    """
    chosen = random.sample(eligible, N)

    support_ids, support_labels = [], []
    query_ids,   query_labels   = [], []

    for lbl, fam in enumerate(chosen):
        seqs = fam_seqs[fam]
        idxs = random.sample(range(len(seqs)), K + Q)
        for i in idxs[:K]:
            support_ids.append(seq_id(fam, i))
            support_labels.append(lbl)
        for i in idxs[K:]:
            query_ids.append(seq_id(fam, i))
            query_labels.append(lbl)

    # Look up row indices in the precomputed matrix
    # Sequences not in the matrix (shouldn't happen) get score 0
    s_idx = [id_to_idx.get(sid, -1) for sid in support_ids]
    q_idx = [id_to_idx.get(sid, -1) for sid in query_ids]

    valid_s = [i for i in s_idx if i >= 0]
    valid_q = [i for i in q_idx if i >= 0]
    if not valid_s or not valid_q:
        return float("nan")

    # scores[q_idx, :][:, s_idx] → (NQ, NK) similarity matrix
    sim = scores[np.ix_(valid_q, valid_s)]  # (NQ, NK)
    sy  = np.array(support_labels)
    qy  = np.array(query_labels)

    if method == "1nn":
        # Predict by class of highest-scoring support sequence
        nn_idx = sim.argmax(axis=1)          # (NQ,)
        preds  = sy[nn_idx]

    elif method == "protonet":
        # Prototype = mean BLAST score over K support sequences per class
        protos = np.stack([sim[:, sy == c].mean(axis=1) for c in range(N)], axis=1)
        preds  = protos.argmax(axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return float((preds == qy).mean())


def evaluate_blast(fam_seqs: dict, id_to_idx: dict, scores: np.ndarray,
                   conf: dict, method: str,
                   k_values=None, n_episodes: int = 300):
    if k_values is None:
        k_values = [1, 2, 5, 10, 20]
    N, Q = conf["N"], conf["Q"]
    results = {}
    for K in k_values:
        min_seqs = K + Q
        eligible = [f for f, seqs in fam_seqs.items() if len(seqs) >= min_seqs]
        if len(eligible) < N:
            print(f"  [blast_{method}] K={K}: only {len(eligible)} eligible, skipping")
            continue
        accs = []
        for _ in range(n_episodes):
            a = run_blast_episode(fam_seqs, eligible, id_to_idx, scores, N, K, Q, method)
            if not np.isnan(a):
                accs.append(a)
        if not accs:
            continue
        results[K] = {
            "mean": float(np.mean(accs)),
            "std":  float(np.std(accs)),
            "n_eligible_families": len(eligible),
            "n_episodes": len(accs),
        }
        print(f"  [blast_{method}] K={K:2d}: {results[K]['mean']:.4f} ± "
              f"{results[K]['std']:.4f}  ({len(eligible)} families)")
    return results

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    conf = CONF.copy()
    k_values   = [1, 2, 5, 10, 20]
    n_episodes = 300

    print("Loading sequences...")
    fam_seqs = load_sequences(PROTEINS)
    print(f"  {len(fam_seqs)} families, "
          f"{sum(len(v) for v in fam_seqs.values())} sequences total.")

    # Build (or load) all-vs-all BLAST score matrix
    ids, scores = build_blast_score_matrix(fam_seqs, cache_path=CACHE)
    id_to_idx   = {sid: i for i, sid in enumerate(ids)}

    print(f"\nScore matrix: {scores.shape}, "
          f"non-zero entries: {(scores > 0).sum():,}")

    print("\nRunning BLAST 1-NN baseline...")
    knn_results = evaluate_blast(fam_seqs, id_to_idx, scores, conf,
                                  method="1nn",
                                  k_values=k_values, n_episodes=n_episodes)

    print("\nRunning BLAST ProtoNet baseline...")
    proto_results = evaluate_blast(fam_seqs, id_to_idx, scores, conf,
                                    method="protonet",
                                    k_values=k_values, n_episodes=n_episodes)

    out = {
        "blast_version": "2.17.0",
        "scoring": "bit-score, evalue<=10, BLOSUM62 (default)",
        "random_baseline": 1.0 / conf["N"],
        "blast_1nn":      knn_results,
        "blast_protonet": proto_results,
        "k_values": k_values,
        "N": conf["N"],
        "Q": conf["Q"],
        "n_episodes": n_episodes,
    }
    path = "results/blast_baseline.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")

    # ── summary at K=5 ──────────────────────────────────────────────────────
    print("\n── Summary at K=5, 5-way ──────────────────────────────────────────")
    print(f"  Random baseline:      20.00%")
    if 5 in knn_results:
        r = knn_results[5]
        print(f"  BLAST 1-NN:           {r['mean']*100:.2f}% ± {r['std']*100:.2f}%")
    if 5 in proto_results:
        r = proto_results[5]
        print(f"  BLAST ProtoNet:       {r['mean']*100:.2f}% ± {r['std']*100:.2f}%")
    print(f"  k-mer 1-NN:           91.38% ± 7.60%   (from kmer_baseline.json)")
    print(f"  k-mer ProtoNet:       92.13% ± 6.82%   (from kmer_baseline.json)")
    print(f"  CNN ProtoNet (ours):  86.87% ± 9.12%   (from kshot_sweep.json)")

if __name__ == "__main__":
    main()
