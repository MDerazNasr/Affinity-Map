# Affinity Map
## Few-Shot Protein Family Classification with Prototypical Networks

Meta-learning framework for protein family classification using Prototypical Networks.
Trains a neural encoder to embed raw amino acid sequences into a metric space where proteins from the same family cluster together — enabling few-shot recognition of unseen families.

<div align="center">
<img src="results/pca_embeddings.png" width="700px">
</div>

## Project Overview

This project applies few-shot learning (Prototypical Networks) to bioinformatics, teaching a model to generalize to new protein families using only a handful of examples.
It leverages deep embeddings and distance-based reasoning to identify functional or structural similarities between proteins.

## Key Features
	•	Protein sequence preprocessing from Pfam FASTA files
	•	1D-CNN encoder trained on amino-acid token sequences
	•	Few-shot learning episodes via Prototypical Networks
	•	Evaluation notebooks for prototype visualization and embeddings
	•	Compatible with PyTorch + MPS/CUDA

## Tech Stack
	•	Python 3.9+
	•	PyTorch
	•	Biopython
	•	NumPy / Matplotlib / Pandas
	•	UMAP-learn (for embedding visualization)
	•	(Optional upcoming): Streamlit / Next.js dashboard for interactive analysis

 ## Quick Start

git clone https://github.com/<your-username>/Protein-fewshot.git
cd Protein-fewshot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


##Method Summary

Protein Sequence (amino acids)
        ↓
Tokenization + Padding
        ↓
1D CNN Encoder → 128-dim embedding
        ↓
Episode Sampler (N-way, K-shot)
        ↓
Prototype calculation (mean embedding per class)
        ↓
Cosine / Euclidean similarity to prototypes
        ↓
Query classification

Implements Prototypical Networks


## Results Summary

5-Way 5-Shot Classification (150 episodes)

Metric	Mean Accuracy	Std. Dev
Cosine Similarity	0.913	±0.079
Euclidean Distance	0.914	±0.087

Both metrics agree →
The embedding space is cleanly separable across families.

##Confusion Matrix

Saved to: results/confusion_cosine.png
Shows which families overlap (useful for structural/functional similarity analysis).


## Failure Case Analysis

Saved to: results/failures.json


##Embedding Visualization

Confusion matrix to measure model accuracy

<div align="center">
<img src="results/confusion_cosine.png" width="420px">
</div>

Installation

git clone https://github.com/<your-username>/protein-fewshot
cd protein-fewshot
pip install -r requirements.txt


