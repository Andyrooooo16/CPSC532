"""
Visualize sentence embeddings using UMAP + Plotly.

Usage:
    python service/visualize.py                  # target paper only
    python service/visualize.py --include-read   # target + read papers
"""
import argparse
import sys
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

sys.path.insert(0, ".")
from zotero_db import get_collection_id, get_items_with_pdfs, get_read_item_keys
from extractor import extract_sentences
from classifier import classify
from embedder import embed
from ranker import rank
from cache import get as cache_get, store as cache_store

COLLECTION_NAME = "Class - Human-Centered AI"

# Colors for each paper (target is always first)
PAPER_COLORS = [
    "#1f77b4",  # blue   — target
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # teal
]
GREY = "rgba(180, 180, 180, 0.25)"


def process_item(item):
    sents = extract_sentences(item["pdf_path"])
    cached = cache_get(item["attachment_key"], sents)
    if cached:
        return sents, cached["labels"], cached["confidences"], cached["embeddings"]
    print(f"  Classifying {len(sents)} sentences...")
    classifications = classify(sents)
    labels = [c[0] for c in classifications]
    confs = [c[1] for c in classifications]
    print(f"  Embedding {len(sents)} sentences...")
    embs = embed(sents)
    cache_store(item["attachment_key"], sents, embs, labels, confs)
    return sents, labels, confs, embs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-read", action="store_true", help="Include read papers in the plot")
    args = parser.parse_args()

    col_id = get_collection_id(COLLECTION_NAME)
    items = get_items_with_pdfs(col_id)
    read_keys = get_read_item_keys(col_id)

    target_item = next(i for i in items if i["item_key"] not in read_keys)
    read_items = [i for i in items if i["item_key"] in read_keys] if args.include_read else []

    # --- Process target paper ---
    print(f"Processing target: {target_item['item_key']}")
    target_sents, target_labels, target_confs, target_embs = process_item(target_item)

    # --- Process read papers ---
    read_data = []  # list of (item, sents, labels, confs, embs)
    for item in read_items:
        print(f"Processing read paper: {item['item_key']}")
        sents, labels, confs, embs = process_item(item)
        read_data.append((item, sents, labels, confs, embs))

    # --- Rank target sentences ---
    all_read_sents = [s for _, sents, _, _, _ in read_data for s in sents]
    all_read_embs = np.vstack([embs for _, _, _, _, embs in read_data]) if read_data else np.empty((0, 768), dtype=np.float32)

    print("Ranking...")
    results = rank(
        target_sents, target_labels, target_confs, target_embs,
        all_read_sents, all_read_embs,
    )
    top_k_sentences = {r["sentence"] for r in results}

    # --- UMAP ---
    all_embs = np.vstack(
        [target_embs] + [embs for _, _, _, _, embs in read_data]
    )
    print(f"Running UMAP on {len(all_embs)} sentences...")
    reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = reducer.fit_transform(all_embs)

    # --- Build Plotly traces ---
    traces = []

    def make_target_trace(name, sents, labels, confs, coords_slice, color):
        top_x, top_y, top_text = [], [], []
        rest_x, rest_y, rest_text = [], [], []

        for i, sent in enumerate(sents):
            x, y = coords_slice[i]
            hover = (
                f"<b>{'[TOP-K] ' if sent in top_k_sentences else ''}{name}</b><br>"
                f"<b>Label:</b> {labels[i]} ({confs[i]:.2f})<br>"
                f"<b>Sentence:</b> {sent[:120]}{'...' if len(sent) > 120 else ''}"
            )
            if sent in top_k_sentences:
                top_x.append(x); top_y.append(y); top_text.append(hover)
            else:
                rest_x.append(x); rest_y.append(y); rest_text.append(hover)

        traces.append(go.Scatter(
            x=top_x, y=top_y, mode="markers",
            name=f"{name} (top-K)",
            marker=dict(color=color, size=8, opacity=1.0),
            hovertext=top_text, hoverinfo="text",
        ))
        traces.append(go.Scatter(
            x=rest_x, y=rest_y, mode="markers",
            name=f"{name} (other)",
            marker=dict(color=GREY, size=5),
            hovertext=rest_text, hoverinfo="text",
            showlegend=False,
        ))

    def make_read_trace(name, sents, labels, confs, coords_slice, color):
        xs, ys, texts = [], [], []
        for i, sent in enumerate(sents):
            x, y = coords_slice[i]
            hover = (
                f"<b>[READ] {name}</b><br>"
                f"<b>Label:</b> {labels[i]} ({confs[i]:.2f})<br>"
                f"<b>Sentence:</b> {sent[:120]}{'...' if len(sent) > 120 else ''}"
            )
            xs.append(x); ys.append(y); texts.append(hover)

        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=name,
            marker=dict(color=color, size=5, opacity=0.5),
            hovertext=texts, hoverinfo="text",
        ))

    # Target paper
    make_target_trace(
        f"Target: {target_item['item_key']}",
        target_sents, target_labels, target_confs,
        coords[:len(target_sents)],
        PAPER_COLORS[0],
    )

    # Read papers
    offset = len(target_sents)
    for idx, (item, sents, labels, confs, _) in enumerate(read_data):
        color = PAPER_COLORS[(idx + 1) % len(PAPER_COLORS)]
        make_read_trace(
            f"Read: {item['item_key']}",
            sents, labels, confs,
            coords[offset : offset + len(sents)],
            color,
        )
        offset += len(sents)

    # --- Layout ---
    fig = go.Figure(traces)
    fig.update_layout(
        title="Sentence Embeddings (t-SNE) — colored by paper, full color = top-K",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        hovermode="closest",
        legend=dict(itemsizing="constant"),
        width=1000,
        height=700,
    )
    fig.show()


if __name__ == "__main__":
    main()
