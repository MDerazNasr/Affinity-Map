"""
utils/splits.py

Single source of truth for the 70/15/15 family-level train/val/test split.
All training and evaluation scripts must import from here to guarantee
that the same families are always in the same split.

Split is deterministic: families are sorted alphabetically, then sliced.
  - Train : first 70%  (families[:n_train])
  - Val   : next  15%  (families[n_train:n_train+n_val])
  - Test  : last  15%  (families[n_train+n_val:])
"""


def get_splits(fams: dict, train_frac: float = 0.70, val_frac: float = 0.15):
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

    Returns
    -------
    train_fams, val_fams, test_fams : dict
        Each is a subset of fams with the same value type.
    """
    names = sorted(fams.keys())
    n = len(names)
    n_train = int(train_frac * n)
    n_val   = int(val_frac  * n)

    train_names = names[:n_train]
    val_names   = names[n_train:n_train + n_val]
    test_names  = names[n_train + n_val:]

    train_fams = {k: fams[k] for k in train_names}
    val_fams   = {k: fams[k] for k in val_names}
    test_fams  = {k: fams[k] for k in test_names}

    return train_fams, val_fams, test_fams
