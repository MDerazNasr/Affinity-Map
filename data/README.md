# 🧬 Protein Few-Shot Dataset (Pfam Subset)

**Source:** Pfam protein family database  
**Families:** 16 total (after cleaning, ~10 non-empty used for training)  
**Sequences per family:** up to 100  
**Average length:** ~230 amino acids  
**Length range kept:** 50 – 700  
**Padding length:** 400  
**Padding ratio:** 0.45 (balanced)  
**Encoding:** 20 canonical amino acids → integer tokens (A = 1 … Y = 20, PAD = 0)  
**File formats:**
- `data/processed/proteins.json` → cleaned raw sequences per family  
- `data/encoded/*.pt` → padded integer tensors for each family  
- `results/length_hist.png` → sequence-length distribution  
- `results/family_stats.csv` → per-family padding stats  

**Purpose:** foundation dataset for few-shot protein family classification using Prototypical Networks.  
**Next stage:** meta-learning training with N-way K-shot episodic sampling.

---

### Processing summary

1. Removed non-canonical amino acids (`B`, `X`, `Z`, etc.)  
2. Kept sequences 50–700 aa long  
3. Shuffled and capped at 100 per family  
4. Encoded and padded/truncated to 400 aa  
5. Verified distributions and padding balance  