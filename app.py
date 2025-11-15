# app.py (fixed)
# Interactive protein embedding explorer

import json, os
import pandas as pd
import plotly.express as px
import umap
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---- Extra projection/caching imports ----
from sklearn.decomposition import PCA
import warnings

# ---- Caching helpers ----
@st.cache_data(show_spinner=False)
def compute_umap(X, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42):
    """Cache UMAP projection to avoid recomputation on Streamlit reruns."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)

@st.cache_data(show_spinner=False)
def compute_pca(X, random_state=42):
    """PCA fallback—robust on macOS, no numba."""
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X)


# ---- Cached loader for embeddings ----
@st.cache_data(show_spinner=False)
def load_embeddings(path: Path):
    """
    Cached loader that returns:
    - data: dict[family] -> list[list[float]]
    - df: flattened DataFrame with columns family,d0..d{D-1}
    - feat_cols: list of embedding column names
    """
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for fam, vecs in data.items():
        for v in vecs:
            rows.append({"family": fam, **{f"d{i}": v[i] for i in range(len(v))}})
    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c.startswith("d")]
    return data, df, feat_cols


# Resolve project root dynamically (one level above)
ROOT = Path(os.getcwd()).resolve()
RESULTS_PATH = ROOT / "results" / "embeddings.json"

# ---- Environment tips expander ----
with st.expander("⚙️ Environment tips (open if the app crashes)", expanded=False):
    st.markdown(
        """
        - On macOS, **UMAP/numba** can crash when Streamlit hot‑reloads or when multiple threads spawn.
        - If you see `resource_tracker ... leaked semaphore` or `zsh: abort`, try:
            1. Switch projection to **PCA (stable)** above.
            2. Limit threads before launching:
               ```
               export NUMBA_NUM_THREADS=1
               export OMP_NUM_THREADS=1
               export OPENBLAS_NUM_THREADS=1
               ```
            3. (Optional) disable file watcher reloads:
               ```
               streamlit run app.py --server.fileWatcherType=none
               ```
        """
    )

st.title("Protein Embedding Explorer")

# Ensure file exists
if not RESULTS_PATH.exists():
    st.error(f"File not found: {RESULTS_PATH}")
    st.info("Run the embedding generation notebook first to create it.")
    st.stop()

# Load embeddings (cached)
data, df, feat_cols = load_embeddings(RESULTS_PATH)

# ---- Projection choice (PCA is safer on macOS; UMAP uses numba and can crash in some envs) ----
st.sidebar.header("Projection")
proj_method = st.sidebar.radio("2D projection", ["PCA (stable)", "UMAP (fast)"], index=0)

X_feats = df[feat_cols].to_numpy()

try:
    if proj_method.startswith("PCA"):
        Z = compute_pca(X_feats)
    else:
        Z = compute_umap(X_feats, n_neighbors=15, min_dist=0.1, metric="cosine")
except Exception as e:
    # Automatic fallback to PCA if UMAP fails
    warnings.warn(f"Projection failed with {type(e).__name__}: {e}. Falling back to PCA.")
    proj_method = "PCA (stable)"
    Z = compute_pca(X_feats)

df["x"], df["y"] = Z[:, 0], Z[:, 1]

# Keep a reference for downloads / search
embeddings = data

# ---- Protein Search Engine (Streamlit UI) ----
# Flatten embeddings and keep a stable global index for plotting/highlighting
embeddings = data  # dict: family -> list[list[float]]
flat_rows = []
for fam, vecs in embeddings.items():
    for v in vecs:
        flat_rows.append((fam, v))
all_vecs = np.array([v for _, v in flat_rows])          # shape: [num_seq, D]
all_fams = [fam for fam, _ in flat_rows]                # parallel labels
df["global_idx"] = np.arange(len(df))                   # 0..num_seq-1 in the same build order

# Sidebar controls
st.sidebar.header("Protein search")
fam_names = sorted(list(embeddings.keys()))
# Optional: filter families by substring
st.sidebar.markdown("---")
fam_filter = st.sidebar.text_input("Filter families (substring)", value="")
filtered_fams = [f for f in fam_names if fam_filter.lower() in f.lower()] if fam_filter else fam_names
if not filtered_fams:
    st.sidebar.warning("No families match the filter; showing all.")
    filtered_fams = fam_names

# Whether to apply the filter to the scatter plot as well (always define)
apply_filter_to_plot = st.sidebar.checkbox("Apply filter to plot", value=False)

if len(fam_names) == 0:
    st.warning("No embeddings found.")
else:
    sel_fam = st.sidebar.selectbox("Family", filtered_fams, index=0)

    # Ensure a valid range per family
    fam_count = len(embeddings[sel_fam])
    sel_idx = st.sidebar.number_input("Sequence index (in family)", min_value=0, max_value=max(0, fam_count-1), value=0, step=1)

    top_k = st.sidebar.slider("Top‑K matches", min_value=5, max_value=50, value=10, step=1)
    run_search = st.sidebar.button("Find similar")

    # Utility: compute the global index of (sel_fam, sel_idx)
    def family_offset(name: str) -> int:
        off = 0
        for f in embeddings:
            if f == name:
                break
            off += len(embeddings[f])
        return off

    # Optionally filter the plot to only show the selected subset of families
    # Base scatter (shown by default)
    if apply_filter_to_plot:
        plot_df = df[df["family"].isin(filtered_fams)].copy()
    else:
        plot_df = df
    base_fig = px.scatter(
        plot_df, x="x", y="y", color="family",
        title="Protein Embeddings (UMAP)", opacity=0.45,
        hover_data=["family", "global_idx"]
    )
    base_fig.update_traces(marker=dict(size=6))

    if run_search and fam_count > 0 and len(all_vecs) > 0:
        # Prepare query vector and compute cosine similarities
        q_global = family_offset(sel_fam) + int(sel_idx)
        query_vec = all_vecs[q_global:q_global+1]                      # shape [1, D]
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec, all_vecs)[0]             # shape [num_seq]
        top_idx = np.argsort(-scores)[:top_k]

        # Results table
        res_df = pd.DataFrame({
            "rank": np.arange(1, top_k+1),
            "family": [all_fams[i] for i in top_idx],
            "cosine": [float(scores[i]) for i in top_idx],
            "global_idx": top_idx
        })
        st.subheader("Nearest neighbours")
        st.dataframe(res_df[["rank", "family", "cosine"]], use_container_width=True)

        # Download results as CSV
        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results (CSV)",
            data=csv_bytes,
            file_name="nearest_neighbours.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Highlight query and top‑K on the scatter
        q_df   = df[df["global_idx"] == q_global]
        top_df = df[df["global_idx"].isin(top_idx)]

        base_fig.add_scatter(x=q_df["x"], y=q_df["y"], mode="markers",
                             marker=dict(size=14, symbol="star"),
                             name="Query", hovertext=["Query"])
        base_fig.add_scatter(x=top_df["x"], y=top_df["y"], mode="markers",
                             marker=dict(size=10),
                             name=f"Top-{top_k}")

    st.plotly_chart(base_fig, use_container_width=True)

    # ---- Downloads ----
    st.subheader("Download embeddings")
    col_json, col_csv = st.columns(2)
    with col_json:
        st.download_button(
            label="Download embeddings.json",
            data=json.dumps(embeddings).encode("utf-8"),
            file_name="embeddings.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_csv:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download embeddings.csv",
            data=csv_bytes,
            file_name="embeddings.csv",
            mime="text/csv",
            use_container_width=True,
        )
