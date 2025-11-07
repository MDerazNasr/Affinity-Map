'''
	1.	Read the FASTA files.
	2.	Extract the sequence strings (just letters).
	3.	Clean them (remove invalid characters).
	4.	Filter them (only sequences of reasonable length).
	5.	Save them into a simple, structured JSON file you can later feed to your model.
'''

from Bio import SeqIO
from pathlib import Path
import os, json, random
# SeqIO handles reading FASTA files, line by line.

#3.2 loop over your families
root = Path("data/raw")
families = [f.stem for f in root.glob("*.fa")]
cleaned = {}

#3.3 Read each FASTA file
#Each record has:
#	•	record.id → unique identifier.
#	•	record.seq → the actual amino acid sequence.
# We only care about the sequence text.
for fam in families:
    seqs = []
    for record in SeqIO.parse(root / f"{fam}.fa", "fasta"):
        seq = str(record.seq)
#3.4 Clean/filter sequences (remove any non-amino acid characters)
        seq = ''.join([c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY"])
        if 50 < len(seq) < 3000: #filter extremes (note - changed upper limit to 3000 from 400)
            seqs.append(seq)
    ''' 
    - < 50 aa = too small (other fragments, non-functional)
    - > 400 aa = too large (hard to model/slower to train)
    '''
    #3.5 - shuffle and limit (we only need a subset of the protein instead of the thousands available)
    random.shuffle(seqs)
    cleaned[fam] = seqs[:100]  #keep only 1000 sequences per family (keeps dataset manageable)

#3.6 Save cleaned sequences to JSON
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/proteins.json", "w") as f:
    json.dump(cleaned, f)
print(f"Saved cleaned data for {len(cleaned)} families.")

'''
Why does this matter?
think of this as tokenisaation for proteins:
    - you are standardising the vocab
    - ensuring the fixed input constraints for your model (length, quality)
    - reducing noise in the data
    - Creating balanced classes for fair-shot sampling
Without this cleaning:
    - you could encounter invalid amino-acids and crash
    - see sequences of wildly diffrent lengths that make training unstable
    - waste compute on low-quality data
    - learn spurios patters -> poor generalisation
'''

print(record.description)
#you'll see things like:
#sp|P12345|MAPK1_HUMAN Mitogen-activated protein kinase 1 OS=Homo sapiens
# That’s how Pfam and UniProt encode metadata — you’re ignoring that now but it’s useful later if you want to link to taxonomy or structure.




