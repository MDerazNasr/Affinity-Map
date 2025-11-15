# streamlit_app.py
# Public demo: interactive explorer for few-shot protein embeddings

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional: UMAP + PCA (UMAP is optional)
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except Exception:
    HAS_PCA = False


# Paths & basic configuration
st.set_page_config(
    page_title="Protein Few-Shot Explorer",
    layout="wide",
    page_icon="🧬",
)

ROOT = Path(__file__).resolve().parent
EMB_PATH = ROOT / "results" / "embeddings.json"
SUMMARY_PATH = ROOT / "results" / "summary.json"
FAILURES_PATH = ROOT / "results" / "failures.json"

st.title("🧬 Protein Few-Shot Embedding Explorer")
st.markdown(
    """
Interactive demo for a **few-shot protein family classifier** using **Prototypical Networks**.
- Each dot = one protein sequence  
- Color = protein family  
- Positions = 2D projection of a learned **128-dim embedding**  
"""
)


# Data loading (cached)

@st.cache_data(show_spinner=True)
def load_embeddings(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found at {path}. "
            "Run the embedding export step first."
        )

    with open(path, "r") as f:
        emb_data = json.load(f)

    rows = []
    for fam, vecs in emb_data.items():
        # vecs is a list of embedding vectors, each a list of floats
        for i, v in enumerate(vecs):
            rows.append({
                "family": fam,
                "seq_idx": i,
                **{f"d{j}": float(v[j]) for j in range(len(v))}
            })

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c.startswith("d")]
    return df, feat_cols


@st.cache_data(show_spinner=True)
def compute_projections(df: pd.DataFrame, feat_cols):
    X = df[feat_cols].to_numpy()

    # Always compute PCA if available
    if HAS_PCA:
        pca = PCA(n_components=2, random_state=42)
        Z_pca = pca.fit_transform(X)
        df["x_pca"] = Z_pca[:, 0]
        df["y_pca"] = Z_pca[:, 1]
        pca_info = pca.explained_variance_ratio_[:2]
    else:
        df["x_pca"] = 0.0
        df["y_pca"] = 0.0
        pca_info = None

    # Optional: UMAP projection
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        Z_umap = reducer.fit_transform(X)
        df["x_umap"] = Z_umap[:, 0]
        df["y_umap"] = Z_umap[:, 1]
    else:
        df["x_umap"] = df["x_pca"]
        df["y_umap"] = df["y_pca"]

    # Embedding norm (for hover)
    df["norm"] = np.linalg.norm(X, axis=1)

    return df, pca_info


@st.cache_data(show_spinner=True)
def load_summary(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=True)
def load_failures(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


# Try to load everything
try:
    df, feat_cols = load_embeddings(EMB_PATH)
except Exception as e:
    st.error(f"❌ Could not load embeddings: {e}")
    st.stop()

df, pca_info = compute_projections(df, feat_cols)
summary = load_summary(SUMMARY_PATH)
failures = load_failures(FAILURES_PATH)

families = sorted(df["family"].unique())
X_vecs = df[feat_cols].to_numpy()  # used later for cosine similarity


# Sidebar controls

st.sidebar.header("Controls")

proj_type = st.sidebar.radio(
    "Projection",
    options=["UMAP (if available)", "PCA only"],
    index=0,
)

if proj_type == "UMAP (if available)" and HAS_UMAP:
    xcol, ycol = "x_umap", "y_umap"
    proj_label = "UMAP"
elif proj_type == "UMAP (if available)" and not HAS_UMAP:
    st.sidebar.warning("UMAP not installed; falling back to PCA.")
    xcol, ycol = "x_pca", "y_pca"
    proj_label = "PCA"
else:
    xcol, ycol = "x_pca", "y_pca"
    proj_label = "PCA"

selected_families = st.sidebar.multiselect(
    "Filter by families",
    options=families,
    default=families,
)

point_size = st.sidebar.slider("Point size", 3, 15, 6)
alpha = st.sidebar.slider("Opacity", 0.2, 1.0, 0.8)

st.sidebar.markdown("---")

if summary is not None:
    st.sidebar.subheader("Eval Summary")
    st.sidebar.write(
        f"N={summary['N']}, K={summary['K']}, Q={summary['Q']}, "
        f"Episodes={summary['episodes']}"
    )
    for m, s in summary["metrics"].items():
        st.sidebar.write(
            f"**{m}**: "
            f"Top-1={s['top1_mean']:.3f}±{s['top1_std']:.3f}, "
            f"Top-3={s['top3_mean']:.3f}±{s['top3_std']:.3f}"
        )


# Layout: Tabs

tab1, tab2, tab3 = st.tabs(["Embedding Explorer", "Similarity Search", "Failure Cases"])


# Tab 1 — Embedding Explorer
with tab1:
    st.subheader("Embedding Explorer")

    dff = df[df["family"].isin(selected_families)].copy()

    st.caption(
        f"Showing **{len(dff)} proteins** "
        f"across **{len(selected_families)} families** "
        f"using **{proj_label} projection**."
    )

    fig = px.scatter(
        dff,
        x=xcol,
        y=ycol,
        color="family",
        hover_data=["family", "seq_idx", "norm"],
        title=f"Protein Embeddings ({proj_label}, color = family)",
        opacity=alpha,
    )
    fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
    fig.update_layout(legend_title_text="Protein Family")

    st.plotly_chart(fig, use_container_width=True)

    # Download filtered CSV
    csv_data = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download filtered embeddings as CSV",
        data=csv_data,
        file_name="filtered_embeddings.csv",
        mime="text/csv",
    )

    if proj_label == "PCA" and pca_info is not None:
        st.caption(
            f"PCA explained variance ratio: "
            f"PC1 = {pca_info[0]:.3f}, PC2 = {pca_info[1]:.3f}"
        )


# Tab 2 — Similarity Search
from sklearn.metrics.pairwise import cosine_similarity  # imported here to keep top tidy

with tab2:
    st.subheader(" Cosine Similarity Search")

    st.markdown(
        """
Pick a **query protein**, and we compute its cosine similarity to all others in the dataset.
This is a simple **protein search engine** in embedding space.
"""
    )

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        fam_q = st.selectbox(
            "Query family",
            options=families,
        )

    # Indices belonging to that family
    fam_indices = df.index[df["family"] == fam_q].tolist()

    with col_q2:
        seq_idx_local = st.slider(
            "Sequence index within family",
            min_value=0,
            max_value=len(fam_indices) - 1,
            value=0,
        )

    # Map local index → global row index in df
    query_idx = fam_indices[seq_idx_local]

    k = st.slider("Number of neighbours (k)", 5, 30, 10)

    if st.button("Run similarity search"):
        query_vec = X_vecs[query_idx : query_idx + 1]  # (1, D)
        scores = cosine_similarity(query_vec, X_vecs)[0]
        top_idx = np.argsort(-scores)[:k]

        nn_rows = []
        for j in top_idx:
            nn_rows.append({
                "rank": len(nn_rows) + 1,
                "index": int(j),
                "family": df.loc[j, "family"],
                "seq_idx": int(df.loc[j, "seq_idx"]),
                "cosine_sim": float(scores[j]),
            })

        nn_df = pd.DataFrame(nn_rows)

        st.markdown(
            f"**Query:** family = `{fam_q}`, global index = `{query_idx}`, "
            f"within-family index = `{seq_idx_local}`"
        )
        st.dataframe(nn_df, use_container_width=True)

        nn_csv = nn_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download neighbours as CSV",
            data=nn_csv,
            file_name="nearest_neighbours.csv",
            mime="text/csv",
        )


# Tab 3 — Failure Cases
with tab3:
    st.subheader(" Misclassifications")

    if failures is None or len(failures) == 0:
        st.info("No failure examples found. Run the failure extraction script to populate `results/failures.json`.")
    else:
        st.markdown(
            """
These are query proteins where the prototype classifier predicted the wrong family.
Useful for:
- discovering **ambiguous** or **borderline** proteins  
- understanding **where the embedding space overlaps**  
"""
        )

        # Show only a subset for readability
        max_rows = st.slider("Max rows to display", 10, 200, 50)
        fails_df = pd.DataFrame(failures[:max_rows])

        st.dataframe(fails_df, use_container_width=True)

        fail_csv = fails_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download failures as CSV",
            data=fail_csv,
            file_name="failures_subset.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "Built with Prototypical Networks over protein sequence embeddings · "
    "All embeddings loaded from `results/embeddings.json`."
)