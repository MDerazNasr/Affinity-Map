# scripts/export_embeddings
import os, json, argparse, sys
from pathlib import Path
import torch

# Resolve project root based on this file's location (…/Protein-fewshot)
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.encoder import ProteinEncoderCNN

def load_checkpoint(ckpt_dir: Path) -> tuple[ProteinEncoderCNN, torch.device]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # try common names
    for name in ["best_protonet.pt", "best_model.pt"]:
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model = ProteinEncoderCNN(proj_dim=128).to(device).eval()
            model.load_state_dict(state)
            return model, device
    raise FileNotFoundError("No checkpoint found in checkpoints/")

def load_family_tensor(pt_path: Path) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu")
    '''
    Accepts:
	•	a single Tensor
	•	a list of Tensors (stacks into shape (num_seq, L))
	•	a dict with "X" tensor, or a dict with "tensors" list (stacks)
    '''
    if isinstance(obj, torch.Tensor):
        return obj.long()
    if isinstance(obj, list):
        return torch.stack(obj).long()
    if isinstance(obj, dict):
        if "X" in obj and isinstance(obj["X"], torch.Tensor):
            return obj["X"].long()
        if "tensors" in obj and isinstance(obj["tensors"], list):
            return torch.stack(obj["tensors"]).long()
    raise ValueError(f"Unsupported .pt format: {pt_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded_dir", default="data/encoded")
    parser.add_argument("--out", default="results/embeddings.json")
    args = parser.parse_args()

    ckpt_dir = (ROOT / "checkpoints")
    enc_dir  = (ROOT / args.encoded_dir)
    out_path = (ROOT / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, device = load_checkpoint(ckpt_dir)
    families = sorted(enc_dir.glob("*.pt"))
    if not families:
        raise FileNotFoundError(f"No .pt files in {enc_dir}")

    embeddings = {}
    with torch.no_grad():
        for pt in families:
            X = load_family_tensor(pt).to(device)
            Z = model(X).cpu().numpy().tolist()
            embeddings[pt.stem] = Z
    with open(out_path, "w") as f:
        json.dump(embeddings, f)
    print(f"✅ Wrote {out_path} with {sum(len(v) for v in embeddings.values())} vectors.")

if __name__ == "__main__":
    main()

#python scripts/export_embeddings.py --encoded_dir data/encoded --out results/embeddings.json
# run this when you retrain or add families



