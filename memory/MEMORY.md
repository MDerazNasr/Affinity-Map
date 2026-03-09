# Affinity Map – Project Memory

## Project Summary
Few-shot protein family classification using Prototypical Networks + 1D-CNN encoder.
Source: Pfam database subset (19 eligible families, up to 100 sequences each).
Interactive dashboard: https://affinity-map-viz.streamlit.app/

## Key Results (updated — real experiments, fixed training loop)
- 5-way accuracy at K=5 (500 episodes): CNN ProtoNet 86.87% ± 9.12%
- k-mer 1-NN baseline at K=5: 91.38% ± 7.60%
- k-mer ProtoNet baseline at K=5: 92.13% ± 6.82%
- CNN ProtoNet leads at K=1: 81.9% vs 76.8-78.4% for k-mer (key finding)
- Learning curves: best val_acc=0.854 at epoch 3; divergence after (overfitting)
- Named confusion: 21 families, biologically interpretable errors
- Original eval_summary.json accuracy (91.25%) was from a broken training loop
  that ran validation only once after all 10 epochs

## Key Files
- `train_protonet.py` — main training loop
- `models/encoder.py` — ProteinEncoderCNN (emb=64, conv=128×3, proj=128, L2-norm)
- `models/protonet.py` — compute_prototypes, prototypical_logits
- `utils/episodes.py` — EpisodeSampler (N-way K-shot)
- `data/configs/protonet.py` — CONF dict (all hyperparameters)
- `results/eval_summary.json` — numerical results
- `results/family_stats.csv` — per-family N, L, PadRatio
- `paper/affinity_map_paper.tex` — full LaTeX research paper (878 lines)

## Architecture
- Token embedding: vocab=21 (20 AA + PAD), dim=64
- Conv1d: 64→128 (k=5), 128→128 (k=5), 128→128 (k=3), all same-padding
- Global avg pool → Linear(128,128) → L2 normalize → 128-dim unit vector
- ~190k parameters, trained from scratch

## Training Config
- N=5 ways, K=5 shots, Q=10 queries
- 10 epochs × 200 train episodes + 100 val episodes
- Adam lr=5e-4, grad clip=1.0, seed=42
- 80/20 family-level train/val split (not sequence-level)

## LaTeX Paper
Written at `paper/affinity_map_paper.tex`.
Compiler: tectonic (installed via brew). Compile: cd paper && tectonic affinity_map_paper.tex
Figures referenced from ../results/ (relative paths).
Author: Mohammed El-Raznasr only (no Claude attribution in paper).

## Experiment Scripts (all run from project root)
- `script/run_experiments.py` — episode dist (500 ep), k-shot sweep, named confusion
- `script/kmer_baseline.py`   — 3-mer 1-NN and ProtoNet baselines across K values
- `script/generate_figures.py` — generates all 4 matplotlib figures (png+pdf)
- `train_protonet.py` — now saves learning_curves.csv (fixed indentation bug)
- `utils/eval.py` — fixed typo: epsiodes → episodes

## Generated Results Files
- `results/learning_curves.csv`        — per-epoch train_loss, train_acc, val_acc
- `results/episode_accuracies.json`    — 500 per-episode accuracies (real histogram)
- `results/kshot_sweep.json`           — CNN ProtoNet accuracy at K=1,2,5,10,20
- `results/kmer_baseline.json`         — k-mer baseline results across K values
- `results/named_confusion.json/.npy`  — 21-family confusion matrix
- `results/fig_*.png/.pdf`             — publication figures

## Key Bugs Fixed
1. train_protonet.py: validation/checkpoint/logging code was outside the epoch
   for-loop (bad indentation) — only ran after epoch 10. Fixed to run each epoch.
2. utils/eval.py: parameter named `epsiodes` but referenced as `episodes` inside.

## Publishability Roadmap
Next steps toward publication:
1. Scale up to 100+ Pfam families (needs more raw FASTA data)
2. Harder episodic sampling (similar-family negative mining)
3. ESM-2 encoder comparison
4. Early stopping at epoch 3 (based on learning curves)
5. Statistical significance tests between methods
