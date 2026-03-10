# Affinity Map – Project Memory

## Project Summary
Few-shot protein family classification using Prototypical Networks + 1D-CNN encoder.
Two experimental settings: S1 = 21 families (BLAST included), S2 = 155 families (scale-up).
Interactive dashboard: https://affinity-map-viz.streamlit.app/

## Final Results (S2 = 155-family scale-up, best checkpoint epoch 13, val=76.6%)

### K-shot sweep (CNN ProtoNet, 300 episodes each):
- K=1:  67.4% ± 13.0%
- K=2:  71.0% ± 12.8%
- K=5:  76.4% ± 10.1%
- K=10: 77.9% ± 10.3%
- K=20: 77.4% ± 11.3%

### k-mer baselines (unchanged, don't depend on training):
- k-mer 1-NN:    K=1: 67.8%, K=2: 75.0%, K=5: 83.0%, K=10: 88.0%, K=20: 91.5%
- k-mer ProtoNet: K=1: 69.5%, K=2: 77.1%, K=5: 86.9%, K=10: 91.6%, K=20: 93.7%

### Statistical significance (1000 matched episodes, paired Wilcoxon):
- K=1:  CNN 66.5%, k-mer 67.6%, diff=-1.1pp, p=0.022 (*),  r=0.18 (small)
- K=2:  CNN 70.9%, k-mer 77.0%, diff=-6.1pp, p=1.6e-37 (***), r=0.54
- K=5:  CNN 75.4%, k-mer 86.7%, diff=-11.3pp, p=5.8e-119 (***), r=0.88
- K=10: CNN 77.2%, k-mer 91.0%, diff=-13.9pp, p=3.1e-148 (***), r=0.96
- K=20: CNN 77.9%, k-mer 93.5%, diff=-15.6pp, p=1.3e-157 (***), r=0.99

### Training dynamics (50 epochs, cosine annealing):
- Best val_acc = 76.6% at epoch 13 (task-distribution overfitting after this)
- Train acc climbs to 93.1% by epoch 50
- Pattern: scale delays but doesn't eliminate overfitting

### S1 (21-family):
- CNN ProtoNet K=5: 86.87%
- BLAST ProtoNet: 97.09% (upper bound)

## Key Files
- `train_protonet.py` — main training loop (50 epochs, cosine annealing)
- `models/encoder.py` — ProteinEncoderCNN (~190k params)
- `models/protonet.py` — compute_prototypes, prototypical_logits
- `data/configs/protonet.py` — CONF dict (N=5, K=5, Q=10, epochs=50, lr=5e-4)
- `checkpoints/best_protonet.pt` — best checkpoint (epoch 13, val=76.6%)
- `paper/affinity_map_paper.tex` — full LaTeX paper (~1000 lines, up to date)
- `results/significance_tests.json` — Wilcoxon test results
- `results/kshot_sweep.json` — CNN ProtoNet K-shot results
- `script/significance_tests.py` — paired Wilcoxon test script
- `script/embed_and_plot.py` — PCA, UMAP, prototype heatmap figures
- `script/run_experiments.py` — episode dist, k-shot sweep, named confusion
- `script/kmer_baseline.py` — k-mer baselines
- `script/generate_figures.py` — matplotlib figure generation

## Architecture
- Token embedding: vocab=21 (20 AA + PAD), dim=64
- Conv1d: 64→128 (k=5), 128→128 (k=5), 128→128 (k=3), all same-padding, dropout=0.1
- Global avg pool → Linear(128,128) → L2 normalize → 128-dim unit vector
- ~190k parameters, trained from scratch on MPS (Apple Silicon)

## LaTeX Paper
Written at `paper/affinity_map_paper.tex`.
Compiler: tectonic. Compile: `cd paper && tectonic affinity_map_paper.tex`
Figures referenced from `../results/` (relative paths).
Author: Mohammed El-Raznasr only (no Claude attribution in paper).
All figures included: learning curves, k-shot sweep, episode hist, confusion matrix,
PCA embeddings, UMAP embeddings, prototype distance heatmap.

## Technical Notes
- k-mer matmul overflows float32: cast to float64 before matmul in significance_tests.py
- k-mer vectors: 3-mer frequency (20^3=8000 dims), L2-normalised
- Eligibility for significance tests: families with ≥ K+10 sequences
- Significance test uses matched episodes (same episode for CNN and k-mer)

## Key Bugs Fixed (historical)
1. train_protonet.py: validation/checkpoint code outside epoch for-loop (bad indent)
2. utils/eval.py: parameter named `epsiodes` instead of `episodes`
3. k-mer significance tests: float32 overflow fixed by .astype(np.float64)
