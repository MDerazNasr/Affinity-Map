#models/ protonet.py
#Core math for Prototypical Networks

import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_prototypes(z_support: torch.Tensor, 
                       y_support: torch.Tensor, 
                       N: int):
    '''
    Compute one prototype per class as the mean of its support embeddings.
    prototype - average embedding per class
    Args:
        z_support: Tensor (N*K, D)  -> embeddings of support samples
        y_support: Tensor (N*K,)    -> integer labels 0..N-1 #contains class ids for each embeddign
        
        tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4]) - means first 5 embeddigs
        N: number of classes
    Returns:
        prototypes: Tensor (N, D)

    For each class c, we select embeddings where y_support == c 
    and take the mean along the 0-axis → that’s the class prototype (a centroid in embedding space).
	'''
    # D = z_support.shape[-1] #embedding dimension
    D = z_support.size(-1)
    protos = torch.zeros(N,D, device= z_support.device, dtype=z_support.dtype)
    for c in range(N):
        #computing the average embedding for class c
        #“For class c, find all embeddings that belong to that class and take their average.”
        m = (y_support == c)
        if not torch.any(m):
            raise RuntimeError(f"Empty support set for class {c}")
        protos[c] = z_support[m].mean(0) 
    protos = F.normalize(protos, p=2, dim=1, eps=1e-8)
    return protos

def prototypical_logits(z_query: torch.Tensor,
                          protos: torch.Tensor, 
                          metric="cosine"):
    """
        Compute similarity (negative distance) between each query and each prototype.
        Args:
            z_query: Tensor (N*Q, D)
            protos:  Tensor (N, D)
            metric:  "euclidean" or "cosine"
        Returns:
            logits:  Tensor (N*Q, N)  -> higher = more similar
    """
    m = (metric or "cosine").lower()
    if m in ("cos", "cosine", "cos_sim", "cos_similarity"):
        #Normalize to unit length, then take dot product
        zq = F.normalize(z_query, p=2, dim=-1, eps=1e-8)
        pp = F.normalize(protos, p=2, dim=-1, eps=1e-8)
        return zq @ pp.t() #cosine similarity as logits(N*Q, N)
    #Euclidean version
    if m in ("euclid", "euclidean", "l2", "sqeuclidean", "sq_l2"):
        # Stable negative squared Euclidean (no torch.cdist → safer on Apple MPS)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        q2 = (z_query**2).sum(dim=1, keepdim=True)        # (Q,1)
        p2 = (protos**2).sum(dim=1, keepdim=True).t() # (1,N)
        cross = z_query @ protos.t()                      # (Q,N)
        d2 = q2 + p2 - 2.0 * cross
        return -d2
    raise ValueError(f"Unknown metric: {metric} (supported: cosine, euclidean)")
    # q2 = (z_query**2).sum(1, keepdim=True) #(N*Q, 1)
    # p2 = (protos**2).sum(1).unsqueeze(0) #(1, N)
    # qp = z_query @ protos.T #(N*Q, N)
    # return - (q2 + p2 - 2*qp) #negative L2 distance

    '''
        Case 1: Cosine metric
        •	F.normalize(z_query, p=2, dim=-1) normalizes each vector to unit length.
        •	The matrix multiplication q @ p.T gives all pairwise cosines between queries and prototypes — higher = more similar.

        Case 2: Euclidean metric
        We use the formula for squared distance:
        \|a-b\|^2 = a^2 + b^2 - 2ab
        •	q2: sum of squares of each query embedding (broadcasted later).
        •	p2: sum of squares of each prototype.
        •	qp: dot product between each query and prototype.
        •	Combine them: -(q2 + p2 - 2*qp) → negative because higher logits = closer = better.
    
    '''







'''
here we define how the model classifies new proteins inside an episode.
when encoder is called sequences, you get embeddings like:
- Zi = F0(Xi)
- each Zi is a vector (128 dim) representing one protein
ProtoNets then:
1. Compute a prototype - avg embedding for each class in the support set
2. Measure distances between query embeddings and those prototypes
3. Predict the class whose prototype is closest

Analogy
1. support set = reference photos
2. query is like the new photo
model = find which family photo album it belongs to by measuring closeness in
embedding space

compute_prototypes -> averages support embeddings by class
prototypical_logits -> computes the negative distances (or cosine similarities) to each 
prototype which behave like logits for softmax classification

prototypes here means the avwerage embedding of a class's K support examples
--> like a center point in space that represents what that class looks like
--> if you have 5 classes, then you have 5 prototypes (0->4)


what does prototypical_logits mean? 

short :
prototypical_logits means “turn the distances between a query and each prototype into scores 
(logits) that can be fed into softmax — so the model can pick the most likely class.”

Let’s say you take one query embedding (a new protein vector).
You want to figure out which class prototype it’s closest to.

So for that query, you:
	1.	Measure its distance (or cosine similarity) to every prototype.
	2.	Turn those distances into numbers that act like logits — the raw scores before softmax classification.
        a. euclidean distance (Smaller = more similar) or cosine similarity (Bigger = more similar)

Because in machine learning, the softmax function expects bigger numbers to mean “more likely.”
But with distance:
	•	Smaller distance = more similar
	•	Larger distance = less simila
so we flip by taking negative distance --> logit = -distance
Closest prototype → largest logit → highest softmax probability

conceptually:
For each query embedding:
	1.	Compare it to every class prototype.
	2.	Compute either:
	•	-euclidean_distance, or
	•	cosine_similarity
	3.	You get a list of numbers — one per class — like:
    [−0.1, −1.2, −3.5, −0.9, −2.7]
    These act like logits for a softmax classifier.
    4. Then the softmax turns those logits into probabilities 
    (which class the query likely belongs to).

    
    Boolean embeddings/ boolean mask/ 
        You’re absolutely right:
        y_support == c does give a boolean tensor.
        Example:
        y_support = tensor([0,0,1,1,2,2])
        c = 1
        y_support == c
        → tensor([False, False, True, True, False, False])

        So far, it’s just a mask of True/False values showing which samples belong to class 1.
        Now the trick: boolean indexing

        In PyTorch (and NumPy), you can use a boolean mask to select rows of another tensor.
        So if z_support is:
        z_support =
        tensor([[0.1, 0.3, 0.2],   # sample 0 (class 0)
                [0.2, 0.4, 0.1],   # sample 1 (class 0)
                [0.9, 0.5, 0.3],   # sample 2 (class 1)
                [1.0, 0.6, 0.4],   # sample 3 (class 1)
                [1.5, 0.8, 0.5],   # sample 4 (class 2)
                [1.7, 0.9, 0.6]])  # sample 5 (class 2)

        Then:
        mask = y_support == 1
        z_support[mask]

        means:
        “Give me all rows in z_support where the corresponding value in y_support is True.”
        Output:
        tensor([[0.9, 0.5, 0.3],
                [1.0, 0.6, 0.4]])

        So z_support[y_support == c] selects all embeddings for class c.

        Then .mean(0)

        Now we average those rows along the 0-axis (vertically) to get one mean vector — the prototype:
        tensor([[0.9, 0.5, 0.3],
                [1.0, 0.6, 0.4]]).mean(0)
        → tensor([0.95, 0.55, 0.35])
        That’s the average embedding for class 1.

        TL;DR

        Expression	What it returns	Meaning
        y_support == c	Boolean mask	Which samples belong to class c
        z_support[y_support == c]	Subset of embeddings	All embeddings of that class
        .mean(0)	One vector	The average embedding (prototype) of that class

        So the boolean mask is just a selector — it tells PyTorch, “give me only the rows in z_support that correspond to class c.”
'''