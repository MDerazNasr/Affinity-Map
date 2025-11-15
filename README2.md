### Protein Few-Shot Learning

Meta-learning framework for protein family classification using Prototypical Networks.
Trains a neural encoder to embed raw amino acid sequences into a metric space where proteins from the same family cluster together — enabling few-shot recognition of unseen families.

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
