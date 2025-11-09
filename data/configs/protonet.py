#configs/protonet.py
import torch
'''
This config file will hold:
- our encoded data
- model sizes
- episodic training settings (N, K, Q)
- training schedule (epochs, learning rate)
- distance metric

'''

CONF = dict(
    # loading data
    encoded_dir ="data/encoded", #folder with family *.pt tensors
    # model
    proj_dim=128, #size of final embeddings from encoder
    #episodic sampling
    N = 5, #How many classes (categories) are we choosing 
    K = 5, #How many examples per class are in the support set (small training set)
    Q = 10, #how many test examples per class do we include in the query set (the test protion for that episode)
    
    #Training schedule
    episodes_per_epoch = 200, #how many samples per episode for every epoch
    val_episodes=100,  #episodes to evaluate/validate per epoch after an episode for the epoch
    epochs = 10, #full passes over the (episodes) budget
    lr = 5e-4,  #adam learning rate
    val_N = 5,
    max_len = 400,
    device =  "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),

    #metric/head
    metric = "cosine", # either euclidean or cosine protonet distances
    #reproducibility
    seed = 42, #RNG seed for consistent sampling: Fixes random number generators so your episodes and results are reproducible
)
'''
an epoch is one full round of training
- for normal training it means one pass through of your data
- for few shot it mens one pass through a set of # of episodes

if ep_per_epoch =200 means you train on 200 randomly generated episodes before saying
thats one epoch

a tensor is basically a multidimensional array like a matrixbut can have more dimensions
In PyTorch, every input, output, and weight is stored as a tensor (which can live on the CPU or GPU).
Your encoded .pt files each hold tensors — for example, a tensor might store a protein sequence embedding as a 128-dimensional vector.

What is an embedding?
- An embedding is just a numerical representation of something (in this case a protein seq.)
Instead of raw amino acid letters, the encoder converts each protein into a vector of numbers that captures its meaning or similarity.

In ProtoNet:
	•	The encoder makes embeddings for all examples.
	•	The prototype for each class = the mean embedding of its K support examples.

lr is the learning rate
- it contriols how big of a step the optimizer takes each time it updates the model's to 
reduce loss
0.01 is the standard

Imagine the model is trying to find the bottom of a valley (the lowest loss).
	•	If lr is too high → the model jumps around wildly and might overshoot the bottom (never converges).
	•	If lr is too low → it moves very slowly, taking tiny baby steps, and training takes forever.

could try to keep one config per experiment or note them down in Git
'''