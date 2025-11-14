#utils/eval.py - like the brain of the evaluation
'''
Evaluation utilities for ProtoNet-style few shot learning

import from notebooks:
from utils.eval import (
    encode, ensure_1d_labels, remap_to_contiguous,
    compute_prototypes, prototypical_logits,
    eval_one_episode, run_eval, confusion_over_episodes
)
'''
# imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

#Basic helpers

# Take raw inputs, run through model, and if the model gives
# per-token embeddings, average them to get one embedding per sequence.
@torch.no_grad()
def encode(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    '''
    Run encoder and return embeddings of shape (B, D)

    If the model returns (B, L, D) (per-position embeddings),
    we mean-pool over L.
    '''
    z = model(x)
    if z.ndim == 3:
        z = z.mean(dim=1)
    return z
# This function forces labels to be a simple 1D vector.
@torch.no_grad()
def ensure_1d_labels(y: torch.Tensor) -> torch.LongTensor:
    '''
        Ensure labels have shape (B, )
        Common cases:
        - y.shape == (B, ) -> returned as long
        - y.shape == (B, 1) -> squeeze column
        - y.shape == (B, L) -> take first column
    '''
    if y.ndim == 1:
        return y.long()
    if y.ndim == 2:
        return y[:, 0].long()
    raise ValueError(f"Expected 1D or 2D labels, got shape {tuple(y.shape)}")

#“Take weird label IDs and squish them into 0..C-1 so classes in this episode are indexed nicely.”
@torch.no_grad()
def remap_to_contiguos(y: torch.LongTensor):
    '''
    Map arbitrary label ids to 0..C-1

    Returns:
        - y_mapped: LongTensor[B] with labels in {0..C-1}
        - classes: LongTensor[C] with original label ids
        - remap: dict {orig_label -> new_index}
    '''
    y = ensure_1d_labels(y)
    classes = torch.unique(y)
    remap: Dict[int, int] = {int(c.item()): i for i, c in enumerate(classes)}

    y2 = y.clone()
    for orig, idx in remap.items():
        y2[y == orig] = idx
    
    return y2.long(), classes.long(), remap

#Prototypical network core
#prototype = mean embedding of all support examples for a class
@torch.no_grad()
def compute_prototypes(z_s: torch.Tensor, sy_contig: torch.LongTensor) -> torch.Tensor:
    '''
    Compute class prototypes given support embeddings.

    Args:
        z_s: (B,D) support embeddings
        sy_contig: (B,) labels in {0..C-1}
    
    Returns:
        P: (C, D) class prototypes (L2-normalized)
    '''
    sy_contig = ensure_1d_labels(sy_contig)
    C = int(torch.max(sy_contig).item()) + 1
    D = z_s.shape[1]

    P = torch.zeros(C,D, device=z_s.device, dtype=z_s.dtype)
    for c in range(C):
        idx = (sy_contig == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise RuntimeError(f"Empty support set for class {c}")
        proto = z_s.index_select(0, idx).mean(0)
        P[c] = F.normalize(proto, p=2, dim=0)
    return P # (C, D)
#Return P, the matrix of all prototypes

@torch.no_grad()
def prototypical_logits(z_q: torch.Tensor, P: torch.Tensor, metric: str= "cosine") -> torch.Tensor:
    '''
    Compute logits for query embeddings against prototypes.

    Args:
        z_q: (Q, D) query embeddings
        P: (C, D) prototypes
        metric: 'cosine' or 'euclidean'
    
    Returns:
        logits: (Q, C)
    '''
    if metric == "cosine":
        zq = F.normalize(z_q, p=2, dim=1)
        Pn = F.normalize(P, p = 2, dim=1)
        return zq @ Pn.T
    if metric == "euclidean":
        q2 = (z_q ** 2).sum(1, keepdim = True)
        p2 = (P ** 2).sum(1).unsqueeze(0)
        d2 = q2 + p2 - 2.0 * (z_q @ P.T)
        return -d2

    raise ValueError(f"Unknown metric: {metric}")

#Simple-episode evaluation

@torch.no_grad()
def eval_one_episode(
    model: torch.nn.Module,
    sampler,
    metric: str = "cosine",
) -> float:
    '''
    Run a single few-shot episodes and return top-1 accuracy

    sampler.sample_episode() must return:
        sx, sy, qx, qy
        sx: support inputs, shape (NS, L, ...) or (NS, ...)
        sy: support labels (NS, ) or (NS, 1) or (NS, L)
        qx: query inputs
        qy: query labels
    '''

    sx, sy, qx, qy = sampler.sample_episode()

    sy = ensure_1d_labels(sy)
    qy = ensure_1d_labels(qy)

    z_s = encode(model, sx)
    z_q = encode(model, qx)

    sy_contig, classes, remap = remap_to_contiguos(sy)
    qy_contig = qy.clone()
    for orig, idx in remap.items():
        qy_contig[qy == orig] = idx
    
    P = compute_prototypes(z_s, sy_contig)
    L = prototypical_logits(z_q, P, metric)

    pred = L.argmax(1)
    acc = (pred == qy_contig).float().mean().item()
    return acc

#Episode sweeps

@dataclass
class EvalStats:
    metric: str
    episodes: int
    mean: float
    std: float
    per_episode: List[float]

#Running a few episodes to measure performance on average
@torch.no_grad()
def run_eval(model: torch.nn.Module, sampler, episodes: int = 150,metric: str = "cosine") -> EvalStats:
    ''' Run Many episodes and return mean +/- std and all episode accuracies'''
    vals = []
    for _ in range(episodes):
        vals.append(eval_one_episode(model, sampler, metric=metric))
    
    vals_np = np.asarray(vals, dtype=np.float32)
    return EvalStats(
        metric = metric,
        episodes = episodes,
        mean = float(vals_np.mean()),
        std = float(vals_np.std()),
        per_episode = list(vals_np),
    )
#Confusion matrices 
@torch.no_grad()
def confusion_over_episodes(
    model: torch.nn.Module,
    sampler, 
    epsiodes: int = 50,
    metric: str = "cosine",
    verbose: bool = False,
) -> np.ndarray:
    '''
    Average (episode-local) confusion matrix over several episodes.

    Returns:
        C_avg: (C,C) numpy array, where C is the number
                classes per episode (assumed fixed).
    '''
    C_accum = None #running total of confusion matrices
    used = 0 #how many episodes we succesfully processed
    skipped = 0 #how many episodes break (shape mismatcj, empty support sets)

    for ep in range(episodes): #loop over episodes
        try:
            #sample an episode → support inputs + labels, query inputs + labels (50 ep)
            sx, sy, qx, qy = sampler.sample_episode()
            
            # Fix to shape (B, )
            
            sy = ensure_1d_labels(sy)
            qy = ensure_1d_labels(qy)
            # encode support + query seq. -> (B, D) embeddings
            z_s = encode(model, sx)
            z_q = encode(model, qx)
            #remap labels to 0..C-1
            '''
            12 -> 0
            50 -> 1
            7  -> 2
            '''
            sy_c, classes, remap = remap_to_contiguos(sy)
            qy_c = qy.clone()

            for orig, idx in remap.items():
                qy_c[qy == orig] = idx
            
            '''
                - Compute prototypes for each class
                - Compute the logits between queries and prototypes
                - argmax(1) picks the predicted class index (0..C-1)
                - Convert predictions and true labels to NumPy for sklearn   
            '''
            P = compute_prototypes(z_s, sy_c)
            L = prototypical_logits(z_q, P, metric)
            pred = L.argmax(1).cpu().numpy()
            true = qy_c.cpu().numpy()
            #this creates a (C,C) confusion matrix, normalised by row
            #IMP - normalize = true means each row sums to 1
            #Row c = probabilities of predicting each class.
            C_ep = int(P.shape[0])
            cm = confusion_matrix(
                true, 
                pred, 
                labels=list(range(C_ep)),
                normalize="true", #this means each row sums to 1
            )
            #accumulate metrics

            if C_accum is None: #first time -> initialize accumulator
                C_accum = np.zeros_like(cm, dtype=np.float64)
            #add this episodes confusion matrix to the total
            C_accum += cm
            used += 1
        # if anything breaks (invalid episode, missing labels),
        # skipping instead of allowing to crash run
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"[confusion] skip episode {ep}: {type(e).__name__}: {e}")
            continue
    # stop if nothing works
    if used == 0:
        raise RuntimeError(
            f"All episodes skipped (used={used}, Check sampler shapes / label formats."
        )
    #announce when you skip
    if verbose and skipped:
        print(f"[confusion] used={used}, skipped={skipped}")
    
    #average sums -> results in mean confusion matrix for all episodes and return
    return C_accum / used
    '''
        A confusion matrix is a table that shows:
        - For each true class (rows), how often we PREDICTED each class (columns).

        Example for 3 classes. 
            true\pred 0 1 2
            0         10 2 3
            1         0 15 5
            2         1 4 12
            row = what the answer should have been
            column = what the model predicted

            row = what the answer should be
            column - what the model predicted
            
            so:
            - row 0 says: of the class 0 queries -> model predicted
            
            A confusion matrix is a table that shows:
            - For each true class (rows), how often we PREDICTED each class (columns).

            Example for 3 classes. 
                true\pred 0 1 2
                0         10 2 3
                1         0 15 5
                2         1 4 12
                row = what the answer should have been
                column = what the model predicted

                row = what the answer should be
                column - what the model predicted
                
                so:
                - row 0 says: of the class 0 queries
        
        This function tries to compute an average confusion matrix across multiple episodes:

            “On average, when the true class is c, how often does the model predict class d?”
    '''