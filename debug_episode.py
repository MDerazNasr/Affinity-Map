# debug_episode.py
import torch
from utils.episodes import EpisodeSampler, loaded_encoded_families
from data.configs.protonet import CONF

fams = loaded_encoded_families(CONF["encoded_dir"])
sampler = EpisodeSampler(fams, N=CONF["N"], K=CONF["K"], Q=CONF["Q"], device="cpu")

def remap_zero_based(sy, qy):
    classes = torch.unique(sy, sorted=True)
    mapping = {int(c): i for i, c in enumerate(classes.tolist())}
    sy2, qy2 = sy.clone(), qy.clone()
    for old, new in mapping.items():
        sy2[sy == old] = new
        qy2[qy == old] = new
    return sy2, qy2, len(classes)

for i in range(5):
    sx, sy, qx, qy = sampler.sample_episode()
    sy2, qy2, N_eff = remap_zero_based(sy, qy)
    print(f"Episode {i}: N_eff={N_eff} | unique sy={torch.unique(sy).tolist()} | unique qy={torch.unique(qy).tolist()}")
    # check support/query overlap by memory pointer (best-effort)
    print("sx shape:", tuple(sx.shape), "qx shape:", tuple(qx.shape))