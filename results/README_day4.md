# Day 4 – Embedding Visualization & Cluster Analysis

Artifacts:
- `results/pca_embeddings.png` – PCA scatter of all protein embeddings
- `results/umap_embeddings.png` – UMAP (cosine) scatter of embeddings
- `results/prototype_distance_heatmap.png` – cosine distance between family prototypes

Headline metrics (PCA space):
- Silhouette (higher is better): (see notebook cell)
- Davies–Bouldin (lower is better): (see notebook cell)

Notes:
- Embeddings are L2-normalized by the encoder; cosine distance is meaningful.
- Families that cluster together may share domains/functions → candidates for biological discussion.
- See Day 5 for ablations and hard-negative checks.

