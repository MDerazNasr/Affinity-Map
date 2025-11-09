#train_protonet.py
#end 2 end training for Prototypical Networks on your Pfam subset

import os, random, torch, torch.nn as nn, torch.optim as optim
from tqdm import trange

from data.configs.protonet import CONF
from models.encoder import ProteinEncoderCNN
from models.protonet import compute_prototypes, prototypical_logits
from utils.episodes import loaded_encoded_families, EpisodeSampler

# ...
fams = loaded_encoded_families(CONF["encoded_dir"])
sampler_cls = EpisodeSampler
train_sampler = sampler_cls(fams, N=CONF["N"], K=CONF["K"], Q=CONF["Q"], device=CONF["device"], max_len=CONF["max_len"])
val_sampler   = sampler_cls(fams, N=CONF["N"], K=CONF["K"], Q=CONF["Q"], device=CONF["device"], max_len=CONF["max_len"])
device = CONF.get("device") or (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("Device:", device)
#utils
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

def pick_device():
    #on apple, 'mps' uses gpu via metal
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
def remap_zero_based(sy: torch.Tensor, qy: torch.Tensor):
    """
    Map arbitrary class ids in this episode to 0..C-1 consistently
    (and remap both support and query labels).
    """
    classes = torch.unique(sy, sorted=True)          # e.g., tensor([7, 13, 42])
    mapping = {int(c): i for i, c in enumerate(classes.tolist())}
    sy2 = sy.clone()
    qy2 = qy.clone()
    for old, new in mapping.items():
        sy2[sy == old] = new
        qy2[qy == old] = new
    return sy2, qy2, len(classes)

# main training loop
def main():
    cfg = CONF.copy()
    device = pick_device()
    print("Device:", device, "| Metric:", cfg["metric"], "| LR:", cfg["lr"])
    set_seed(cfg.get("seed", 42))

    # 1 - load families and split by family (not by sequence!)
    fams = loaded_encoded_families(cfg["encoded_dir"])
    train_sampler = EpisodeSampler(fams, N=cfg["N"], K=cfg["K"], Q=cfg["Q"],
                                    device=device, max_len=cfg["max_len"])
    val_sampler   = EpisodeSampler(fams, N=cfg["N"], K=cfg["K"], Q=cfg["Q"],
                                    device=device, max_len=cfg["max_len"])

    model = ProteinEncoderCNN(proj_dim=cfg["proj_dim"]).to(device)

    names = sorted(list(fams.keys()))
    if len(names) < cfg["N"] + 1:
        raise ValueError(f"Not enough families to train/val. Found {len(names)}. ")
    
    split = int(0.8 * len(names)) 
    #The first 80% of families → used for training episodes
	#The remaining 20% → used for validation episodes
    train_fams, val_fams = {}, {}
    for k in names[:split]: #my_list[start:end]
        train_fams[k] = fams[k]
    for k in names[split:]:
        val_fams[k] = fams[k]
    print(f"Train families: {len(train_fams)} | Val families: {len(val_fams)}")

    #Make sure each side has enough sequences for K+Q sampling
    train_sampler = EpisodeSampler(train_fams, N=cfg["N"], K=cfg["K"], Q=cfg["Q"], device=device)
    val_sampler =  EpisodeSampler(val_fams, N=cfg["N"], K=cfg["K"], Q=cfg["Q"], device=device)

    # 2 - Model, optimizer, loss
    model = ProteinEncoderCNN(proj_dim=cfg["proj_dim"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    #criterion = “How wrong was the model?” (the error calculator).
	#opt = “How should I fix the model?” (the weight updater).

    best_val = 0.0 #keeping track of the best validation accuracy model has achieved so far
    os.makedirs("checkpoints", exist_ok=True) #	•Checkpoints are saved copies of your model (so you don’t lose progress).


    # 3 - Training epochs
    for epoch in range(1, cfg["epochs"] +1):
        #training phase
        model.train()
        val_accs = []
        running_loss, running_acc = 0.0, 0.0
        # running_acc is a “rolling total” that accumulates accuracy values until you average them out.


        for _ in trange(cfg["episodes_per_epoch"], desc=f"Epoch {epoch}/{cfg['epochs']}"):
            # a - sample an episode
            sx, sy, qx, qy = train_sampler.sample_episode() # sx:(N*K,L), sy:(N*K), qx:(N*Q,L), qy:(N*Q)
            # ensure (B,L) long tensors
            if sx.dim() == 1: sx = sx.unsqueeze(0)
            if qx.dim() == 1: qx = qx.unsqueeze(0)
            sx = sx.long(); qx = qx.long()

            # REMAP labels to 0..C-1 and get effective N (C)
            sy, qy, N_eff = remap_zero_based(sy, qy)


            # b - embed suport & query
            z_s = model(sx) #(N*K, D)
            z_q = model(qx) #(N*Q, D)

            # c - compute protoypes and logits
            protos = compute_prototypes(z_s, sy, N_eff) #(N, D)
            logits = prototypical_logits(z_q, protos, cfg["metric"]) #(N*Q, N)
            val_accs.append((logits.argmax(1) == qy).float().mean().item())
            
            if torch.isnan(logits).any():
                print("[DEBUG] metric:", cfg["metric"], "| device:", device)
                print("[DEBUG] any NaNs in z_s/z_q:", torch.isnan(z_s).any().item(), torch.isnan(z_q).any().item())
                print("[DEBUG] pad% support/query:",
                    (sx.eq(0).float().mean().item()*100),
                    (qx.eq(0).float().mean().item()*100))
                raise RuntimeError("NaNs in logits")

            # also ensure labels are in 0..N-1
            assert sy.min().item() >= 0 and sy.max().item() < cfg["N"], f"Bad support labels: {sy[:8]}"
            assert qy.min().item() >= 0 and qy.max().item() < cfg["N"], f"Bad query labels: {qy[:8]}"


            # d - loss and setup
            loss = criterion(logits, qy) #This line calculates how wrong the model’s predictions are.
            opt.zero_grad() # Before calculating new gradients, you reset the old ones. Otherwise, PyTorch would add new gradients on top of old ones from previous iterations
            loss.backward() #It calculates how much each weight in your model contributed to the loss.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step() #This is where the optimizer actually updates the model’s weights.

            # e - tracking metrics
            with torch.no_grad(): #	Temporarily turns off gradient tracking. no training, just evaluating
                running_loss += loss.item() #loss.item() converts the PyTorch scalar tensor (e.g. tensor(0.5234)) into a plain Python number (0.5234).
                preds = logits.argmax(dim=1) #finds which class index has the largest logit for each query.
                running_acc += (preds == qy).float().mean().item() #Compares predicted vs true labels elementwise → gives a boolean vector:
    train_loss = running_loss / cfg["episodes_per_epoch"]
    train_acc = running_acc / cfg["episodes_per_epoch"]

    #validation phase
    model.eval() #entering eval mode to ensure consistent results
    val_accs = [] #to store accuracy from each validation episode
    with torch.no_grad(): #turns off gradient tracking, no updates being made so memory being save 
        for _ in range(cfg["val_episodes"]): #runs several val episodes for acccurary
            sx, sy, qx, qy = val_sampler.sample_episode() #randomly generate val episde
            '''
            Feeds both support (sx) and query (qx) sequences through the encoder (ProteinEncoderCNN).
	        Produces embeddings:
	        z_s: support embeddings → shape (N*K, D)
            z_q: query embeddings → shape (N*Q, D)
            '''
            
            if sx.dim() == 1: sx = sx.unsqueeze(0)
            if qx.dim() == 1: qx = qx.unsqueeze(0)
            sx = sx.long(); qx = qx.long()

            sy, qy, N_eff = remap_zero_based(sy, qy)
            
            z_s = model(sx); z_q = model(qx)
            protos = compute_prototypes(z_s, sy, N_eff)
            logits = prototypical_logits(z_q, protos, cfg["metric"])
            val_accs.append((logits.argmax(1) == qy).float().mean().item()) #see bottom
    val_acc = sum(val_accs) / len(val_accs)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

    # checkpoint best
    if val_acc > best_val:
        best_val = val_acc
        torch.save(
            {"state_dict": model.state_dict(), "config": cfg},
            "checkpoints/best_protonet.pt"
        )
        print(f"Saved best checkpoint (val_acc={best_val:.3f})")

if __name__ == "__main__":
    main()



'''
In each episode:
- pick N families
- take K support and Q query sequences per class
- embed support+query with your CNN coder
- compute prototypes (mean of support embeddings per class)
- compute logits for queries vs prototypes (euclidean or cosine)
- optimise cross-entropy loss on query labels


other:
- A logit is just a raw score that a model produces before 
turning it into a probability.

Cross-entropy loss measures how well the predicted probabilities match the true labels.
So the closer the model’s predicted probability for the true class is to 1,
the smaller (better) the loss.

- optimise cross-entropy loss on query labels
    1.The model predicts logits for query samples → [N*Q, N].
	2.	Softmax turns logits into probabilities per class.
	3.	Cross-entropy compares those probabilities with the true query_y labels.
	4.	The optimizer adjusts the encoder’s weights to minimize that loss.

    
Note - 
model.eval()
	•	Certain layers (like dropout and batch normalization) behave differently during training vs evaluation:
	•	During training → dropout randomly hides neurons, batchnorm tracks running stats.
	•	During evaluation → dropout is disabled, batchnorm uses stored averages.
	•	So calling model.eval() ensures consistent, stable evaluation results.

    
evaluate loop

model.eval()
	•	This switches the model into evaluation mode.
	•	Certain layers (like dropout and batch normalization) behave differently during training vs evaluation:
	•	During training → dropout randomly hides neurons, batchnorm tracks running stats.
	•	During evaluation → dropout is disabled, batchnorm uses stored averages.
	•	So calling model.eval() ensures consistent, stable evaluation results.

Important: it doesn’t turn off gradient computation by itself — that’s done next.

⸻

val_accs = []
	•	An empty list to store the accuracy from each validation episode.
	•	We’ll average them at the end to get the overall validation accuracy.

⸻

with torch.no_grad():
	•	Temporarily turns off gradient tracking.
	•	We’re not training or updating weights here — just measuring performance.
	•	Saves memory and makes evaluation faster.

⸻

for _ in range(cfg["val_episodes"]):
	•	Runs several validation episodes (e.g. 100) to get an accurate measure.
	•	Each episode samples new N-way, K-shot, Q-query sets from the validation families.

⸻

sx, sy, qx, qy = val_sampler.sample_episode()
	•	Uses the EpisodeSampler to randomly generate a single validation episode.
	•	Returns:
	•	sx = support inputs (N×K sequences)
	•	sy = support labels (N×K labels)
	•	qx = query inputs (N×Q sequences)
	•	qy = query labels (N×Q labels)

The model never sees these samples during training — they’re from the held-out families.

⸻

z_s = model(sx); z_q = model(qx)
	•	Feeds both support (sx) and query (qx) sequences through the encoder (ProteinEncoderCNN).
	•	Produces embeddings:
	•	z_s: support embeddings → shape (N*K, D)
	•	z_q: query embeddings → shape (N*Q, D)

⸻

protos = compute_prototypes(z_s, sy, cfg["N"])
	•	Computes prototypes for each class:
	•	For each class c, find all its support embeddings (z_s[sy == c])
	•	Take their mean to get one vector (the class prototype)
	•	Output: a tensor of shape (N, D) — one prototype per class.

⸻
logits = prototypical_logits(z_q, protos, cfg["metric"])
	•	Computes logits for each query embedding against all prototypes.
	•	Measures similarity (using cfg["metric"]: either "euclidean" or "cosine").
	•	Output shape: (N*Q, N) — each row is a query, each column a class score.

⸻

val_accs.append((logits.argmax(1) == qy).float().mean().item())

Let’s unpack that part carefully:
	1.	logits.argmax(1) → gets the predicted class for each query (index of highest score).
	2.	(logits.argmax(1) == qy) → compares predicted labels with true labels → boolean tensor (True/False).
	3.	.float() → converts True → 1.0, False → 0.0.
	4.	.mean() → computes the accuracy for this episode (fraction correct).
	5.	.item() → converts it to a regular Python number.
	6.	.append(...) → adds that episode’s accuracy to the list val_accs.

So after the loop, val_accs holds one accuracy per episode.

val_acc = sum(val_accs) / len(val_accs)
	•	Averages all validation episode accuracies to get the mean validation accuracy for this epoch.
	•	This gives a stable estimate of how well your model generalizes to unseen families.

'''