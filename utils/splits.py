"""
utils/splits.py

Single source of truth for the 70/15/15 family-level train/val/test split.
All training and evaluation scripts must import from here to guarantee
that the same families are always in the same split.

Split is deterministic: families are randomly shuffled with a fixed seed,
then sliced.  Random shuffle avoids any alphabetical artefacts (e.g. all
Zinc-finger families landing in test) while remaining fully reproducible.
  - Train : first 70%  (names[:n_train])
  - Val   : next  15%  (names[n_train:n_train+n_val])
  - Test  : last  15%  (names[n_train+n_val:])
"""

import random


def get_splits(fams: dict, train_frac: float = 0.70, val_frac: float = 0.15,
               seed: int = 42):
    """
    Parameters
    ----------
    fams : dict
        {family_name: data}  — any value type (encoded tensors or raw sequences)
    train_frac : float
        Fraction of families for training (default 0.70)
    val_frac : float
        Fraction of families for validation / early-stopping (default 0.15)
        Remainder goes to test.
    seed : int
        RNG seed for reproducibility (default 42).

    Returns
    -------
    train_fams, val_fams, test_fams : dict
        Each is a subset of fams with the same value type.
    """
    names = list(fams.keys())
    rng = random.Random(seed)
    rng.shuffle(names)           # random, reproducible

    n = len(names)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)

    train_names = names[:n_train]
    val_names   = names[n_train:n_train + n_val]
    test_names  = names[n_train + n_val:]

    return (
        {k: fams[k] for k in train_names},
        {k: fams[k] for k in val_names},
        {k: fams[k] for k in test_names},
    )
