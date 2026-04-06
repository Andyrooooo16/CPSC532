#!/usr/bin/env python3
"""Generate both <stem>_all.json and <stem>_ctx_.json for every paper in user-study/public/papers.
Backs up any existing highlight files before overwriting.
"""
import sys
from pathlib import Path
import shutil
import time
import json
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
# Ensure user-study is importable as a module path for generate_highlights
sys.path.insert(0, str(REPO_ROOT / 'user-study'))

import generate_highlights as gh

HIGHLIGHTS_DIR = gh.HIGHLIGHTS_DIR
PAPERS_DIR = gh.PAPERS_DIR

# Build combos: for each paper_key produce both all_highlights and contextual_highlights (no priors)
combos = []
for pk in gh.papers_cfg.keys():
    combos.append((pk, 'all_highlights', ()))
    combos.append((pk, 'contextual_highlights', ()))

# Backup existing files that will be overwritten
backup_dir = HIGHLIGHTS_DIR / f"backup_{int(time.time())}"
backup_dir.mkdir(parents=True, exist_ok=True)
print(f"Backing up existing highlights to {backup_dir}")
for paper_key, condition, prior in combos:
    fname = gh.highlight_filename(paper_key, condition, prior)
    out_path = HIGHLIGHTS_DIR / f"{fname}.json"
    if out_path.exists():
        dest = backup_dir / out_path.name
        print(f"  Backing up {out_path.name} -> {dest.name}")
        shutil.copy(out_path, dest)

# Pre-load all paper data
needed_keys = {paper_key for paper_key, _, _ in combos}
paper_data = {}
for paper_key in needed_keys:
    print(f"Loading data for {paper_key}...")
    sents, embs, labels, confs = gh.get_paper_data(paper_key)
    paper_data[paper_key] = dict(sentences=sents, embeddings=embs, labels=labels, confidences=confs)

# Generate files (overwrite)
for i, (paper_key, condition, prior_stems) in enumerate(sorted(combos), 1):
    fname = gh.highlight_filename(paper_key, condition, prior_stems)
    out_path = HIGHLIGHTS_DIR / f"{fname}.json"

    print(f"[{i}/{len(combos)}] Generating {out_path.name}...")

    data = paper_data[paper_key]
    pdf_path = PAPERS_DIR / gh.papers_cfg[paper_key]['filename']

    # No prior papers here (we're generating base _ctx_ without priors)
    read_sents = []
    read_embs = np.empty((0, 768), dtype=np.float32)

    if condition == 'contextual_highlights':
        ranked = gh.rank_novel(
            data['sentences'], data['labels'], data['confidences'], data['embeddings'],
            read_sents, read_embs,
            novelty_lambda=0.6, top_k_fraction=0.30,
        )
    else:
        ranked = gh.rank(data['sentences'], data['labels'], data['confidences'], data['embeddings'], read_sents, read_embs)

    if gh.MAX_HIGHLIGHTS is not None:
        ranked = ranked[:gh.MAX_HIGHLIGHTS]

    print(f"  {len(ranked)} sentences selected. Locating bounding boxes…")
    highlights = gh.locate_highlights(pdf_path, ranked)
    print(f"  {len(highlights)} sentences located.")

    with open(out_path, 'w') as f:
        json.dump(highlights, f, indent=2)
    print(f"  Wrote {out_path.name}\n")

print("Done.")
