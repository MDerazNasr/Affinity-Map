# --- Write a short results README section ---
readme = """# Day 3 – Encoder Evaluation (Few-Shot)

Artifacts:
- `results/umap_embeddings.png` – UMAP of support+query embeddings across episodes
- `results/prototype_distance_heatmap.png` – cosine distances between class prototypes
- `results/episode_log.txt` – per-episode quick accuracy log

Headline:
- Mean few-shot accuracy (cosine): <fill from console>
- Mean few-shot accuracy (euclidean): <fill from console>

Notes:
- Embeddings are L2-normalized; cosine & Euclidean are consistent.
- Episodes are sampled with remapped dense labels to handle sparse draws.
- See notebook for exact N/K/Q and sampler config.
"""

with open("results/README_day3.md","w") as f:
    f.write(readme)
print("Wrote: results/README_day3.md")