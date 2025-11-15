### Few-Shot Protein Family Classification with Prototypical Networks

##Metric learning for biological sequence understanding 
    • CNN encoder 
    • Embedding analysis 
    • Interactive visualization

<div align="center">
<img src="results/pca_embeddings.png" width="420px">
</div>

## Overview

Built an end-to-end few-shot learning system for classifying protein families using Prototypical Networks and a 1D CNN encoder.

##Motivation

Most protein families have very few labeled examples. Traditional supervised learning fails when:
	•	a class has <10 sequences
	•	new families emerge with almost zero examples
	•	datasets are heavily imbalanced

Few-shot learning directly addresses this by training the model to generalize from N-way K-shot episodes.

## This project answers:

Can we learn a general-purpose embedding space where proteins cluster by family, and can we classify new families with as few as 5 examples?

Spoiler: Yes

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

Artifacts generated in results:

pca_embeddings.png
confusion_cosine.png
plot_interactive.html
filtered_embeddings.csv
embeddings.json
kq_grid.png
failures.json
summary.json

Notebook:
notebooks/03_visualization_dashboard.ipynb

Installation

git clone https://github.com/<your-username>/protein-fewshot
cd protein-fewshot
pip install -r requirements.txt


## Running the Pipeline

1. Export embeddings

python scripts/export_embeddings.py

2. Run episodic evaluation 

python scripts/eval_episodes.py

3. Explore embeddings 

Open:

notebooks/03_visualization_dashboard.ipynb

4. Dashboard

streamlit run app/streamlit_app.py

