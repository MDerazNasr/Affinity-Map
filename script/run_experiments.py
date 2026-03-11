"""
script/run_experiments.py

Runs three publishability experiments using the saved checkpoint:
  1. Per-episode accuracy distribution (500 episodes) → results/episode_accuracies.json
  2. K-shot sweep K∈{1,2,5,10,20}        → results/kshot_sweep.json
  3. Named family confusion matrix         → results/named_confusion.npy + .json

Run from project root:
    python script/run_experiments.py
"""

import os, sys, json, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.configs.protonet import CONF
from models.encoder import ProteinEncoderCNN
from utils.episodes import loaded_encoded_families, EpisodeSampler
from utils.splits import get_splits

# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, conf: dict) -> torch.nn.Module:
    device = conf["device"]
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ProteinEncoderCNN(proj_dim=conf["proj_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def encode(model, x, device):
    x = x.long().to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    with torch.no_grad():
        return model(x)

def remap(sy, qy):
    classes = torch.unique(sy, sorted=True)
    mapping = {int(c): i for i, c in enumerate(classes.tolist())}
    sy2, qy2 = sy.clone(), qy.clone()
    for old, new in mapping.items():
        sy2[sy == old] = new
        qy2[qy == old] = new
    return sy2, qy2, len(classes)

def compute_protos(z_s, sy, N, device):
    import torch.nn.functional as F
    D = z_s.size(-1)
    protos = torch.zeros(N, D, device=device, dtype=z_s.dtype)
    for c in range(N):
        m = (sy == c)
        protos[c] = z_s[m].mean(0)
    return F.normalize(protos, p=2, dim=1)

def cosine_logits(z_q, protos):
    import torch.nn.functional as F
    return F.normalize(z_q, p=2, dim=-1) @ F.normalize(protos, p=2, dim=-1).t()

def run_episode(model, sampler, device):
    sx, sy, qx, qy = sampler.sample_episode()
    sy, qy, N = remap(sy, qy)
    z_s = encode(model, sx, device)
    z_q = encode(model, qx, device)
    protos = compute_protos(z_s, sy, N, device)
    logits = cosine_logits(z_q, protos)
    return (logits.argmax(1) == qy).float().mean().item()

# ── 1. Per-episode accuracy distribution ─────────────────────────────────────

def experiment_episode_distribution(model, fams, conf, n_episodes=500):
    print(f"\n[1/3] Per-episode accuracy distribution ({n_episodes} episodes)...")
    device = conf["device"]
    sampler = EpisodeSampler(fams, N=conf["N"], K=conf["K"], Q=conf["Q"], device=device)
    accs = [run_episode(model, sampler, device) for _ in range(n_episodes)]
    out = {
        "n_episodes": n_episodes,
        "N": conf["N"], "K": conf["K"], "Q": conf["Q"],
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "per_episode": accs
    }
    path = "results/episode_accuracies.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   mean={out['mean']:.4f} ± {out['std']:.4f}  →  saved {path}")
    return out

# ── 2. K-shot sweep ───────────────────────────────────────────────────────────

def experiment_kshot_sweep(model, fams, conf, k_values=None, n_episodes=300):
    if k_values is None:
        k_values = [1, 2, 5, 10, 20]
    print(f"\n[2/3] K-shot sweep {k_values} ({n_episodes} episodes each)...")
    device = conf["device"]
    results = {}
    for K in k_values:
        # filter families that have enough sequences for this K
        min_seqs = K + conf["Q"]
        eligible = {
            name: pack for name, pack in fams.items()
            if (pack["X"].shape[0] if isinstance(pack, dict) else len(pack)) >= min_seqs
        }
        if len(eligible) < conf["N"]:
            print(f"   K={K:2d}: only {len(eligible)} eligible families (need {conf['N']}), skipping")
            continue
        sampler = EpisodeSampler(eligible, N=conf["N"], K=K, Q=conf["Q"], device=device)
        accs = [run_episode(model, sampler, device) for _ in range(n_episodes)]
        results[K] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "n_eligible_families": len(eligible),
            "n_episodes": n_episodes,
        }
        print(f"   K={K:2d}: {results[K]['mean']:.4f} ± {results[K]['std']:.4f}  "
              f"({len(eligible)} families)")

    path = "results/kshot_sweep.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved {path}")
    return results

# ── 3. Named family confusion matrix ─────────────────────────────────────────

def experiment_named_confusion(model, fams, conf, n_episodes=300):
    print(f"\n[3/3] Named confusion matrix ({n_episodes} episodes)...")
    import torch.nn.functional as F
    device = conf["device"]
    N, K, Q = conf["N"], conf["K"], conf["Q"]

    # Only families with enough sequences
    eligible = {
        name: pack for name, pack in fams.items()
        if (pack["X"].shape[0] if isinstance(pack, dict) else len(pack)) >= K + Q
    }
    family_names = sorted(eligible.keys())
    name_to_idx = {n: i for i, n in enumerate(family_names)}
    n_families = len(family_names)
    print(f"   {n_families} eligible families: {family_names}")

    # Accumulate pairwise confusion counts: confusion[true_fam][pred_fam]
    confusion = np.zeros((n_families, n_families), dtype=np.float64)
    pair_counts = np.zeros((n_families, n_families), dtype=np.float64)

    for ep in range(n_episodes):
        # manually sample so we know the family names
        chosen_names = random.sample(family_names, N)
        sx_list, qx_list, sy_list, qy_list = [], [], [], []

        for lbl, fname in enumerate(chosen_names):
            pack = eligible[fname]
            if isinstance(pack, dict):
                seqs = [pack["X"][i] for i in range(pack["X"].shape[0])]
            else:
                seqs = pack

            idxs = random.sample(range(len(seqs)), K + Q)
            sup_idxs, qry_idxs = idxs[:K], idxs[K:]

            def pad(t, L=400):
                t = t[:L]
                if t.numel() < L:
                    t = torch.cat([t, torch.zeros(L - t.numel(), dtype=torch.long)])
                return t

            sx_list += [pad(seqs[i]) for i in sup_idxs]
            qx_list += [pad(seqs[i]) for i in qry_idxs]
            sy_list += [lbl] * K
            qy_list += [lbl] * Q

        sx = torch.stack(sx_list).long().to(device)
        qx = torch.stack(qx_list).long().to(device)
        sy = torch.tensor(sy_list, dtype=torch.long, device=device)
        qy = torch.tensor(qy_list, dtype=torch.long, device=device)

        z_s = encode(model, sx, device)
        z_q = encode(model, qx, device)
        protos = compute_protos(z_s, sy, N, device)
        logits = cosine_logits(z_q, protos)
        preds = logits.argmax(1)  # indices into chosen_names

        # Map episode-local indices → global family indices
        for q_idx in range(len(qy)):
            true_ep_lbl = qy[q_idx].item()
            pred_ep_lbl = preds[q_idx].item()
            true_fam_idx = name_to_idx[chosen_names[true_ep_lbl]]
            pred_fam_idx = name_to_idx[chosen_names[pred_ep_lbl]]
            confusion[true_fam_idx, pred_fam_idx] += 1
            pair_counts[true_fam_idx, :] += 0  # handled per-row below

    # Normalise each row by number of times that family appeared as true class
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for families never sampled
    confusion_norm = confusion / row_sums

    np.save("results/named_confusion.npy", confusion_norm)
    out = {
        "family_names": family_names,
        "n_episodes": n_episodes,
        "confusion_matrix": confusion_norm.tolist(),
    }
    with open("results/named_confusion.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"   Saved results/named_confusion.npy and results/named_confusion.json")
    return confusion_norm, family_names

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    torch.manual_seed(42)

    conf = CONF.copy()
    checkpoint = "checkpoints/best_protonet.pt"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading model from {checkpoint} on {conf['device']}...")
    all_fams = loaded_encoded_families(conf["encoded_dir"])
    _, _, test_fams = get_splits(all_fams)
    model = load_model(checkpoint, conf)
    print(f"Loaded {len(all_fams)} families → using {len(test_fams)} test families.")

    experiment_episode_distribution(model, test_fams, conf, n_episodes=500)
    experiment_kshot_sweep(model, test_fams, conf, k_values=[1, 2, 5, 10, 20], n_episodes=300)
    experiment_named_confusion(model, test_fams, conf, n_episodes=300)

    print("\nAll experiments complete.")

if __name__ == "__main__":
    main()
