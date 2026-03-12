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

## ESM-2 LoRA Results (REAL — Colab T4 GPU, 30 epochs, 1000 matched episodes)
Full spec: EPOCHS=30, EP_TRAIN=200, EP_VAL=100, EVAL_EP=1000, ESM_MAX_LEN=512, BATCH_SIZE=16
Best val_acc=0.963 at epoch 8. Checkpoint: checkpoints/best_esm2_lora_ep*/

| K  | LoRA           | Frozen | k-mer | ΔFrz       | Δkmer        |
|----|----------------|--------|-------|------------|--------------|
| 1  | 88.8% ± 9.8%  | 86.3%  | 70.9% | +2.5pp *** | +17.9pp ***  |
| 2  | 91.9% ± 7.4%  | 92.6%  | 80.1% | -0.6pp *   | +11.8pp ***  |
| 5  | 93.3% ± 6.2%  | 95.5%  | 87.8% | -2.2pp *** | +5.5pp ***   |
| 10 | 94.3% ± 5.4%  | 96.4%  | 92.2% | -2.1pp *** | +2.1pp **    |
| 20 | 94.6% ± 5.3%  | 96.9%  | 94.6% | -2.3pp *** | 0.0pp ns     |

**Key finding**: K-dependent interaction — LoRA improves single-shot (K=1, +2.5pp p<0.001)
but degrades multi-shot prototype quality (K≥2, -0.6 to -2.3pp, all p≤0.05).

## Key Files
- `train_protonet.py` — main training loop (50 epochs, cosine annealing)
- `models/encoder.py` — ProteinEncoderCNN (~190k params)
- `models/protonet.py` — compute_prototypes, prototypical_logits
- `data/configs/protonet.py` — CONF dict (N=5, K=5, Q=10, epochs=50, lr=5e-4)
- `checkpoints/best_protonet.pt` — best checkpoint (epoch 13, val=76.6%)
- `colab/esm2_finetune_full.py` — full-spec ESM-2 LoRA training (GPU, EPOCHS=30)
- `colab/run_lora.ipynb` — Colab notebook (7 cells, mounts Drive, trains, saves)
- `colab/pack_for_colab.sh` — creates protein_fewshot.zip for Drive upload
- `paper/affinity_map_paper.tex` — full LaTeX paper (UPDATED with real results)
- `results/significance_tests.json` — Wilcoxon test results (CNN vs k-mer)
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
- MPS kills ESM-2 LoRA backprop (attention activation ~8GB). Must use CPU or GPU.
- Colab T4 is the reference hardware for LoRA training

## Key Bugs Fixed (historical)
1. train_protonet.py: validation/checkpoint code outside epoch for-loop (bad indent)
2. utils/eval.py: parameter named `epsiodes` instead of `episodes`
3. k-mer significance tests: float32 overflow fixed by .astype(np.float64)
4. Colab Cell 3: unzip goes to /content/ directly (no subdirectory), os.chdir("/content")
5. Colab Cell 7: needs os.chdir("/content") at top (Drive wipe between cells)
