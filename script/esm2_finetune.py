"""
script/esm2_finetune.py

ESM-2 episodic fine-tuning via LoRA + ProtoNet.

Uses HuggingFace ESM-2 (facebook/esm2_t6_8M_UR50D) with PEFT LoRA adapters,
trained episodically under the same N-way K-shot protocol as the CNN baseline.
Produces matched-episode comparison against frozen ESM-2 and k-mer ProtoNet
on the held-out test families.

Run from project root:
    python script/esm2_finetune.py

Outputs:
    results/esm2_finetune_results.json   -- K-shot accuracy + Wilcoxon tests
    results/esm2_lora_curves.csv         -- per-epoch train/val curves
    checkpoints/best_esm2_lora/          -- PEFT saved adapter
"""

import json, random, os, sys, csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from scipy import stats

# MPS workaround for some ESM-2 attention ops
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import EsmModel, EsmTokenizer
from peft import LoraConfig, get_peft_model

from utils.splits import get_splits
from models.protonet import compute_prototypes, prototypical_logits

# ── Config ──────────────────────────────────────────────────────────────────
HF_MODEL  = "facebook/esm2_t6_8M_UR50D"
DATA_PATH = "data/processed/proteins.json"
OUT_PATH  = "results/esm2_finetune_results.json"
CKPT_DIR  = "checkpoints/best_esm2_lora"
CURVE_PATH = "results/esm2_lora_curves.csv"

# Episodic training
N        = 5          # ways
K_TRAIN  = 5          # shots during training
Q        = 10         # queries per class
EPOCHS   = 30
EP_TRAIN = 200        # episodes per epoch
EP_VAL   = 100        # val episodes per epoch
LR       = 1e-4       # lower LR than CNN (pre-trained weights)
GRAD_CLIP = 1.0
SEED     = 42

# LoRA
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROP    = 0.1
LORA_TARGETS = ["query", "value"]

# Evaluation
K_SWEEP  = [1, 2, 5, 10, 20]
EVAL_EP  = 1000        # matched episodes

ESM_MAX_LEN = 512
BATCH_SIZE  = 16       # sequences per forward pass
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
CKPT_EVERY = 5         # save intermediate checkpoint every N epochs

device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {device}")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_families(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    cleaned = {}
    for fam, seqs in raw.items():
        clean = [
            s for s in seqs
            if isinstance(s, str)
            and all(a in AA_SET for a in s)
            and 50 <= len(s) <= 3000
        ]
        if len(clean) >= 30:
            cleaned[fam] = clean
    return cleaned


# ── ESM-2 embedding ──────────────────────────────────────────────────────────

def embed_sequences(model, tokenizer, seqs, batch_size=BATCH_SIZE):
    """
    Embed raw sequences -> (N, 320) L2-normalised tensor.
    Runs in the current grad mode (supports train and eval).
    """
    truncated = [s[:ESM_MAX_LEN] for s in seqs]
    all_embs = []

    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ESM_MAX_LEN + 2,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        hidden = out.last_hidden_state          # (B, L, 320)
        mask   = enc["attention_mask"].unsqueeze(-1).float()
        emb    = (hidden * mask).sum(1) / mask.sum(1)  # mean-pool
        emb    = nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb)

    return torch.cat(all_embs, dim=0)


# ── k-mer ────────────────────────────────────────────────────────────────────

import itertools as _it

_AA_ORDER  = "ACDEFGHIKLMNPQRSTVWY"
_KMER_VOCAB = {
    "".join(t): i
    for i, t in enumerate(_it.product(list(_AA_ORDER), repeat=3))
}
_KMER_DIM = len(_KMER_VOCAB)  # 8000


def kmer_vector(seq):
    v = np.zeros(_KMER_DIM, dtype=np.float64)
    for i in range(len(seq) - 2):
        km = seq[i:i + 3]
        if km in _KMER_VOCAB:
            v[_KMER_VOCAB[km]] += 1
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


# ── Episode sampling ─────────────────────────────────────────────────────────

def sample_episode_raw(fams, names, N, K, Q):
    chosen = random.sample(names, N)
    sup_seqs, sup_labels, qry_seqs, qry_labels = [], [], [], []
    for lbl, fam in enumerate(chosen):
        pool = fams[fam]
        idxs = random.sample(range(len(pool)), K + Q)
        for i in idxs[:K]:
            sup_seqs.append(pool[i]); sup_labels.append(lbl)
        for i in idxs[K:]:
            qry_seqs.append(pool[i]); qry_labels.append(lbl)
    return sup_seqs, sup_labels, qry_seqs, qry_labels


# ── Build model ───────────────────────────────────────────────────────────────

def build_lora_model():
    print(f"Loading {HF_MODEL} with LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    tokenizer = EsmTokenizer.from_pretrained(HF_MODEL)
    base      = EsmModel.from_pretrained(HF_MODEL)
    lora_cfg  = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROP,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model.to(device), tokenizer


def build_frozen_model(tokenizer):
    """Frozen ESM-2 (same base, no LoRA) for the comparison baseline."""
    base = EsmModel.from_pretrained(HF_MODEL).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


# ── Training ─────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_fams, val_fams):
    train_names = [f for f in train_fams if len(train_fams[f]) >= K_TRAIN + Q]
    val_names   = [f for f in val_fams   if len(val_fams[f])   >= K_TRAIN + Q]
    print(f"  Eligible: {len(train_names)} train, {len(val_names)} val families")

    opt       = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    curves = []

    with open(CURVE_PATH, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

        for epoch in range(1, EPOCHS + 1):
            # Training
            model.train()
            tr_losses, tr_accs = [], []
            for _ in range(EP_TRAIN):
                sup_seqs, sup_labels, qry_seqs, qry_labels = sample_episode_raw(
                    train_fams, train_names, N, K_TRAIN, Q)

                z_s = embed_sequences(model, tokenizer, sup_seqs)
                z_q = embed_sequences(model, tokenizer, qry_seqs)
                sy  = torch.tensor(sup_labels, device=device)
                qy  = torch.tensor(qry_labels, device=device)

                protos = compute_prototypes(z_s, sy, N)
                logits = prototypical_logits(z_q, protos, "cosine")
                loss   = criterion(logits, qy)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

                tr_losses.append(loss.item())
                tr_accs.append((logits.argmax(1) == qy).float().mean().item())

            scheduler.step()

            # Validation
            model.eval()
            va_accs = []
            with torch.no_grad():
                for _ in range(EP_VAL):
                    sup_seqs, sup_labels, qry_seqs, qry_labels = sample_episode_raw(
                        val_fams, val_names, N, K_TRAIN, Q)
                    z_s = embed_sequences(model, tokenizer, sup_seqs)
                    z_q = embed_sequences(model, tokenizer, qry_seqs)
                    sy  = torch.tensor(sup_labels, device=device)
                    qy  = torch.tensor(qry_labels, device=device)
                    protos = compute_prototypes(z_s, sy, N)
                    logits = prototypical_logits(z_q, protos, "cosine")
                    va_accs.append((logits.argmax(1) == qy).float().mean().item())

            tl = float(np.mean(tr_losses))
            ta = float(np.mean(tr_accs))
            va = float(np.mean(va_accs))
            curves.append({"epoch": epoch, "train_loss": tl, "train_acc": ta, "val_acc": va})
            writer.writerow([epoch, round(tl, 6), round(ta, 6), round(va, 6)])
            cf.flush()

            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={tl:.4f}  train={ta:.3f}  val={va:.3f}")

            if va > best_val:
                best_val = va
                model.save_pretrained(CKPT_DIR)
                print(f"  --> saved best (val={best_val:.3f})")

            # Intermediate checkpoint safeguard
            if epoch % CKPT_EVERY == 0:
                model.save_pretrained(f"{CKPT_DIR}_ep{epoch}")
                print(f"  --> intermediate checkpoint at epoch {epoch}")

    return curves, best_val


# ── Matched evaluation ───────────────────────────────────────────────────────

def wilcoxon_test(a, b):
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    n = len(a)
    r = 1 - (2 * stat) / (n * (n + 1) / 2)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return float(p), float(r), sig


def run_matched_eval(lora_model, tokenizer, frozen_model, test_fams, K):
    eligible = [f for f in test_fams if len(test_fams[f]) >= K + Q]
    lora_accs, frozen_accs, kmer_accs = [], [], []

    lora_model.eval()
    frozen_model.eval()

    for _ in range(EVAL_EP):
        chosen = random.sample(eligible, N)
        sup_seqs, sup_labels, qry_seqs, qry_labels = [], [], [], []
        for lbl, fam in enumerate(chosen):
            pool = test_fams[fam]
            idxs = random.sample(range(len(pool)), K + Q)
            for i in idxs[:K]:
                sup_seqs.append(pool[i]); sup_labels.append(lbl)
            for i in idxs[K:]:
                qry_seqs.append(pool[i]); qry_labels.append(lbl)

        sy = torch.tensor(sup_labels, device=device)
        qy = torch.tensor(qry_labels, device=device)
        all_seqs = sup_seqs + qry_seqs
        nsk = N * K

        with torch.no_grad():
            z_lora   = embed_sequences(lora_model,   tokenizer, all_seqs)
            z_frozen = embed_sequences(frozen_model, tokenizer, all_seqs)

        for z_all, accs_list in [(z_lora, lora_accs), (z_frozen, frozen_accs)]:
            z_s = z_all[:nsk]; z_q = z_all[nsk:]
            protos = compute_prototypes(z_s, sy, N)
            logits = prototypical_logits(z_q, protos, "cosine")
            accs_list.append((logits.argmax(1) == qy).float().mean().item())

        # k-mer ProtoNet (numpy, truly matched sequences)
        ks_s = np.stack([kmer_vector(s) for s in sup_seqs])
        ks_q = np.stack([kmer_vector(s) for s in qry_seqs])
        sy_np = np.array(sup_labels)
        qy_np = np.array(qry_labels)
        protos_km = np.stack([ks_s[sy_np == c].mean(0) for c in range(N)])
        norms = np.linalg.norm(protos_km, axis=1, keepdims=True)
        protos_km = protos_km / (norms + 1e-8)
        nq = np.linalg.norm(ks_q, axis=1, keepdims=True)
        ks_q_n = ks_q / (nq + 1e-8)
        preds_km = (ks_q_n @ protos_km.T).argmax(1)
        kmer_accs.append((preds_km == qy_np).mean())

    return np.array(lora_accs), np.array(frozen_accs), np.array(kmer_accs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load and split data
    print("Loading data...")
    all_fams = load_families(DATA_PATH)
    train_fams, val_fams, test_fams = get_splits(all_fams)
    print(f"  {len(all_fams)} families → train={len(train_fams)} "
          f"val={len(val_fams)} test={len(test_fams)}")

    # Build and train LoRA model
    lora_model, tokenizer = build_lora_model()
    print(f"\nTraining for {EPOCHS} epochs...")
    curves, best_val = train(lora_model, tokenizer, train_fams, val_fams)
    print(f"\nBest val_acc = {best_val:.3f}")

    # Load best checkpoint for eval
    from peft import PeftModel
    base    = EsmModel.from_pretrained(HF_MODEL).to(device)
    lora_model_eval = PeftModel.from_pretrained(base, CKPT_DIR).to(device)
    lora_model_eval.eval()

    # Frozen baseline
    frozen_model = build_frozen_model(tokenizer)

    # Matched evaluation across K values
    print("\nRunning matched evaluation on test families...")
    results = {}
    print(f"{'K':>3}  {'LoRA':>7}  {'Frozen':>7}  {'k-mer':>7}  "
          f"{'LoRA-Frz':>10}  {'LoRA-kmer':>10}")
    print("-" * 58)

    for K in K_SWEEP:
        eligible = [f for f in test_fams if len(test_fams[f]) >= K + Q]
        if len(eligible) < N:
            print(f"K={K}: only {len(eligible)} eligible, skip"); continue

        print(f"K={K}: {EVAL_EP} matched episodes ({len(eligible)} fam)...", flush=True)
        lora_a, frozen_a, kmer_a = run_matched_eval(
            lora_model_eval, tokenizer, frozen_model, test_fams, K)

        p_lf, r_lf, sig_lf = wilcoxon_test(lora_a, frozen_a)
        p_lk, r_lk, sig_lk = wilcoxon_test(lora_a, kmer_a)

        results[str(K)] = {
            "lora_mean":   float(lora_a.mean()),   "lora_std":   float(lora_a.std()),
            "frozen_mean": float(frozen_a.mean()),  "frozen_std": float(frozen_a.std()),
            "kmer_mean":   float(kmer_a.mean()),    "kmer_std":   float(kmer_a.std()),
            "lora_vs_frozen": {"p": p_lf, "r": r_lf, "sig": sig_lf,
                               "diff": float(lora_a.mean() - frozen_a.mean())},
            "lora_vs_kmer":   {"p": p_lk, "r": r_lk, "sig": sig_lk,
                               "diff": float(lora_a.mean() - kmer_a.mean())},
            "n_episodes": EVAL_EP, "n_eligible": len(eligible),
        }

        print(f"  {K:>3}  {lora_a.mean():>7.4f}  {frozen_a.mean():>7.4f}  "
              f"{kmer_a.mean():>7.4f}  "
              f"{lora_a.mean()-frozen_a.mean():>+8.4f}{sig_lf:>3}  "
              f"{lora_a.mean()-kmer_a.mean():>+8.4f}{sig_lk:>3}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")

    print("\n── LaTeX rows ──")
    for K, r in results.items():
        sig_f = r["lora_vs_frozen"]["sig"]
        sig_k = r["lora_vs_kmer"]["sig"]
        print(f"ESM-2 LoRA & "
              f"{r['lora_mean']*100:.1f}$\\pm${r['lora_std']*100:.1f} & "
              f"{r['lora_vs_frozen']['diff']*100:+.1f} pp ({sig_f}) & "
              f"{r['lora_vs_kmer']['diff']*100:+.1f} pp ({sig_k}) \\\\  % K={K}")


if __name__ == "__main__":
    main()
