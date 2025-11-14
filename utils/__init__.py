#utils/__init__.py
#create package for import for eval notebook
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.eval import (
    encode, 
    ensure_1d_labels,
    remap_to_contiguos,
    compute_prototypes,
    prototypical_logits,
    eval_one_episode,
    run_eval,
    confusion_over_episodes
)

__all__  = [
    "encode",
    "ensure_1d_labels",
    "remap_to_contiguos",
    "compute_prototypes",
    "prototypical_logits",
    "eval_one_episode",
    "run_eval",
    "confusion_over_episodes",
]

