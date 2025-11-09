#building episode sampler
#loaded_encoded_families - reads your data/encoded/*.pt files into memory
#each .pt is one protein family with a tensor of sequences (shape[N_seq, L], dtype long, where 0 is PAD)

import os, glob, random, torch
from typing import Dict, List, Tuple, Union

Tensor = torch.Tensor
Pack = Union[List[Tensor], Dict[str, Tensor], Tensor]

def _pad1d(x: Tensor, L: int) -> Tensor:
    x = x[:L]
    if x.numel() < L:
        x = torch.cat([x, torch.zeros(L - x.numel(), dtype=torch.long)], dim=0)
    return x
def _as_list(pack: Pack) -> List[Tensor]:
    # Accept {"X": Tensor[N,L]}, list[Tensor(L)], or Tensor[N,L]
    if isinstance(pack, dict) and "X" in pack:
        X = pack["X"]
        if torch.is_tensor(X) and X.ndim == 2:
            return [X[i] for i in range(X.shape[0])]
    if torch.is_tensor(pack) and pack.ndim == 2:
        return [pack[i] for i in range(pack.shape[0])]
    if isinstance(pack, list):
        return pack
    raise TypeError(f"Unsupported family pack type: {type(pack)}")


def loaded_encoded_families(encoded_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
    '''
    Load all family tensors from data/encoded/*.pt
    Returns: {family_name: {"X": Tensor[N, L]}}
    Skips empty/ malformed files safety
    '''
    families = {}
    for path in glob.glob(os.path.join(encoded_dir, "*.pt")):
        pack = torch.load(path)

        #Handle two possible save formats:
        # - dict: {"X": Tensor[N,L]...}
        # X -> data sensor, N -> number of sequences, L -> sequence length (columns)
        # - list: [Tensor[L], Tensor[L], ...]
        # pack = torch.load(os.path.join(encoded_dir, path))
        if isinstance(pack, dict): #saviing under new format: with the main tensor under key "X"
            X = pack.get("X", None)
        elif isinstance(pack, list):
            #if its a legacy list of 1D tensors
            if len(pack) == 0:
                continue
            X = torch.stack(pack)
        elif torch.is_tensor(pack):
            # already a tensor (either [N, L] or [L])
            X = pack.unsqueeze(0) if pack.ndim == 1 else pack
        else:
            continue
        
        #Basic validity checks
        if X is None or not isinstance(X, torch.Tensor):
            continue
        if X.ndim != 2 or X.dtype != torch.long: #ensures your encoder can embed these (embedding requires long IDs)
            continue
        if X.shape[0] == 0:
            continue
            
        fam = os.path.splitext(os.path.basename(path))[0] #strip folder and .pt to get the family name
        families[fam] = {"X":X.long()} #ensure dtype is long (embedding expects longs)
    return families


#Sample output:
#families = {
#    "kinase": {"X": Tensor[N_seq, L]},
#    "transferase": {"X": Tensor[N_seq, L]},...}

#Episode sampler - every time you call sample_episode():
# - randomly chooses N families (classes)
# - within each family, randomly picks K support and Q query sequences
# - returns 4 tensors: support_x, support_y, query_x, query_y

# ... keep your helpers ...

class EpisodeSampler:
    def __init__(self, fams, N, K, Q, device="cpu", max_len=400):
        import torch, random
        self.N,self.K,self.Q = int(N),int(K),int(Q)
        self.device,self.max_len = device,max_len

        def to_list(pack):
            if isinstance(pack, dict) and "X" in pack and torch.is_tensor(pack["X"]) and pack["X"].ndim==2:
                X=pack["X"]; return [X[i] for i in range(X.shape[0])]
            if torch.is_tensor(pack) and pack.ndim==2:
                return [pack[i] for i in range(pack.shape[0])]
            if isinstance(pack, list):
                return pack
            return []

        self.fams = {k: to_list(v) for k,v in fams.items() if len(to_list(v)) >= (self.K+self.Q)}
        self.names = list(self.fams.keys())
        if len(self.names) < self.N:
            raise ValueError(f"Need at least N={self.N} eligible families (>=K+Q={self.K+self.Q}), have {len(self.names)}.")

    def _pad1d(self, x, L):
        import torch
        x = x[:L]
        if x.numel() < L: x = torch.cat([x, torch.zeros(L-x.numel(), dtype=torch.long)], 0)
        return x

    def sample_episode(self):
        import random, torch
        chosen = random.sample(self.names, self.N)
        sx,qx,sy,qy = [],[],[],[]
        for lbl,fam in enumerate(chosen):
            seqs = self.fams[fam]
            n,need = len(seqs), self.K+self.Q
            if n < need:
                raise RuntimeError(f"[Invariant broken] {fam} has {n} < {need}")
            idxs = random.sample(range(n), need)
            sup, qry = idxs[:self.K], idxs[self.K:]
            sx += [self._pad1d(seqs[i], self.max_len) for i in sup]
            qx += [self._pad1d(seqs[i], self.max_len) for i in qry]
            sy += [lbl]*self.K
            qy += [lbl]*self.Q
        sx = torch.stack(sx).to(self.device)
        qx = torch.stack(qx).to(self.device)
        sy = torch.tensor(sy, dtype=torch.long, device=self.device)
        qy = torch.tensor(qy, dtype=torch.long, device=self.device)
        assert len(torch.unique(sy))==self.N and len(torch.unique(qy))==self.N, "Episode collapsed"
        return sx,sy,qx,qy

        
        
        # support_x, support_y = [], []
        # query_x, query_y = [], []

        # for class_label, fam in enumerate(chosen):
        #     #shape [N_seq, L] (N_seq = number of sequences in this family; L = fixed length like 400).
        #     X = self.families[fam]["X"]

        #     # random permutation of indices; take first K+Q
        #     # creates a random permutation of all indices from 0 to N_seq - 1.
        #     perm = torch.randperm(X.shape[0])[: self.K + self.Q]
        #     #Split into K support and Q query
        #     s_idx = perm[: self.K] # keeps only the first (K + Q) indices,because we only need that many sequences for this episode (K support + Q query)
        #     #now split them into two groups: support, and query set    
        #     q_idx = perm[self.K : self.K + self.Q] #take the first K indices out of the shuffled lisrt to be used as support samples

        #     support_x.append(X[s_idx])
        #     query_x.append(X[q_idx])
        #     # makes a list of K identical labels for that class and .extend adds all of those labels tp support_y
        #     support_y.extend([class_label] * self.K)
        #     # Makes a list of Q copies of the current class label and adds them to query y
        #     query_y.extend([class_label] * self.Q)

        #     # Concatenate lists into big tensors and move to target device (cpu/mps )
        #     support_x = torch.cat(support_x, dim = 0).to(self.device) #(N*K, L)
        #     query_x = torch.cat(query_x, dim=0).to(self.device) #(N*Q, L)
        #     support_y = torch.tensor(support_y, dtype=torch.long, device=self.device)
        #     query_y = torch.tensor(query_y, dtype=torch.long, device=self.device)

        '''
            support_x = [Tensor_class0, Tensor_class1, ...]  # each of shape (K, L)
            query_x   = [Tensor_class0, Tensor_class1, ...]  # each of shape (Q, L)
            support_y = [0, 0, 0, 1, 1, 1, 2, ...]           # list of labels
            query_y   = [0, 0, 1, 1, 2, 2, ...]
        '''
            # return support_x, support_y, query_x, query_y

    


'''
we create tiny tasks (episodes) on the fly; 
pick N classes,
take K labeled examples per class (suport)
Q unlabelled per class (query)
Train the model to classify queries by comparing to support prototypes

Why episodes?
Few-shot meta-learning doesn’t learn one big classifier. 
It learns how to quickly build a classifier from a handful of examples (K-shot) and generalize to new classes. 
So we train it on thousands of small supervised problems (episodes).
'''

