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
def encode_sequence(seq, max_len=3000): #update to 3000 from 400
    """ Convert amino acid string to a fixed length tensor of integers"""
    ids = []
    for a in seq[:max_len]: #:max_len so it truncates longer sequences so it doesnt ruin memory
        ids.append(aa_to_idx.get(a, 0))  #if an amino acid is not found, assign 0 (PAD)
    length = min(len(seq), max_len)
    if len(ids) < max_len: #if the sequence is shorter than max length
        ids += [0] * (max_len - len(ids))  #pad to max length for shorter amino acids
    return torch.tensor(ids), length #returns 1D tensor with shape [400] 


#you need to pad because neural nets needs fixed shapes to make mini batches
#padding is a standard solution to this problem
#pad to the right 0 is conventional and works with CNN
# print(encode_sequence(test, 30))

data = json.load(open("data/processed/proteins.json"))
os.makedirs("data/encoded", exist_ok=True)

for fam, seqs in data.items():
    tensors = []
    for s in seqs: #for each family of proteins
        tensor = encode_sequence(s, max_len=3000) #convert every string to a fixed length (400) tensor
        tensors.append(tensor)
    print(f"{fam}: {len(tensors)} sequences")
    if len(tensors) == 0:
        print(f"Skipping {fam} - no valid sequences")
        continue
    X = torch.stack([e[0] for e in tensors])
    L = torch.tensor([e[1] for e in tensors]) 
    torch.save({"X": X, "lengths":L}, f"data/encoded/{fam}.pt") #save one .pt file per family. each file is a list of tensors. 
    # one file per family to make few-shot smapling easier: pick families, then pick sequences from their respective files 
    # Now we save X - the padded smaples and L - the og length of each protein
print("Encoded sequences have been saved to data/encoded/")


sample = torch.load("data/encoded/C2H2.pt")
print(sample.keys())  # dict_keys(['X', 'lengths'])
print(sample["X"].shape, sample["lengths"].shape)

#test:
# import glob
# x = torch.load("data/encoded/C2H2.pt")[0]
# print(x.shape, x[:30])
# #torch.size([400] tensor)

# for path in glob.glob("data/encoded/*.pt"):
#     arr = torch.load(path)
#     assert all(t.shape[0] == 400 for t in arr), f"Bad length in {path}"
# print("All sequences are length 400")
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

