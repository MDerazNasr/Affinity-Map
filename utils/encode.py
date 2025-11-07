import json, torch, os
#json to read the cleaned sequenced
#torch to turn sequences into pytorch sensorx
#os to create folders/files cleanly


#define the amino acid vocab
AA = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {}
for i, a in enumerate(AA): #1-20
    aa_to_idx[a] = i+1
aa_to_idx["PAD"] = 0 #padding token

test = "ACDDEFGHIKLMNPQRS" #encoder function
def encode_sequence(seq, max_len=400):
    """ Convert amino acid string to a fixed length tensor of integers"""
    ids = []
    for a in seq[:max_len]: #:max_len so it truncates longer sequences so it doesnt ruin memory
        ids.append(aa_to_idx.get(a, 0))  #if an amino acid is not found, assign 0 (PAD)
    if len(ids) < max_len: #if the sequence is shorter than max length
        ids += [0] * (max_len - len(ids))  #pad to max length for shorter amino acids
    return torch.tensor(ids) #returns 1D tensor with shape [400] 


#you need to pad because neural nets needs fixed shapes to make mini batches
#padding is a standard solution to this problem
#pad to the right 0 is conventional and works with CNN
# print(encode_sequence(test, 30))

data = json.load(open("data/processed/proteins.json"))
os.makedirs("data/encoded", exist_ok=True)

for fam, seqs in data.items():
    tensors = []
    for s in seqs:
        tensor = encode_sequence(s, max_len=400)
        tensors.append(tensor)
'''
Each protein is a string of amino-acid letters:
- but neural networks can't process letters
They need numbers (tensors) as input

here we convert each amino acids letter into a numeric token ID
and pads all sequences to the same length
which makes them usable by Pytorch

Why Encode?

Think of this like NLP tokenization:
NLP word -> 'hello' -> 1056...
Amino acid -> 'A' -> 1.....'C' -> 3

input = "ACDE..."
output = [1,2,4,5...]

Why Pad Sequences?
Proteins vary in length - some are 80 amino acids, others 350
Neural networks expect all sequences in a batch to be the same size

so you pad shorter ones with zeros (0 = "blank") to reach a fixed max 
length (say 400)

Example:
Sequence A: [1, 3, 5, 8]
Sequence B: [2, 9]
→ Pad to 6
Sequence A: [1, 3, 5, 8, 0, 0]
Sequence B: [2, 9, 0, 0, 0, 0]

This way, both sequences are the same length and can be processed together
'''

