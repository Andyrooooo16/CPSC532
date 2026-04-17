# CPSC532 Research Codebase Guide

This repository contains three connected systems:

1. A Zotero plugin and Python service for contextual highlighting
2. A controlled user-study web app and data pipeline
3. Analysis and figure-generation scripts for reporting results

This README is a practical walkthrough of where data lives, how it is processed, and how statistics and figures are produced.

## 1) Repository Map

- contextual-highlighter/
  - Zotero plugin source (bootstrap and UI glue)
- service/
  - Python backend logic for sentence extraction, classification, embeddings, and ranking
- user-study/
  - Study web app, questions/config, raw sessions, flattened analysis tables, and exports
- scripts/
  - Utility scripts for highlight generation/export and validation checks

## 2) End-to-End Data Flow

### A. Raw study data collection

- Raw sessions are written by the Express app to:
  - user-study/data/sessions/
- Each session JSON includes participant metadata, paper order, condition assignment, attempts, questionnaires, and final responses.

The study server entry point is:
- user-study/server.js

Key study config files:
- user-study/study-config.json (paper metadata and participant counterbalancing)
- user-study/questions.json (question definitions and questionnaires)

### B. Manual grading source of truth

Free-text grading comes from:
- user-study/data/marked sessions/CPSC Short Answer Marked - Sheet1.csv

This CSV is used when flattening sessions into analysis-ready tables.

### C. Flatten raw sessions into analysis tables

Script:
- user-study/scripts/flatten_sessions.py

Inputs:
- user-study/data/sessions/*.json
- user-study/questions.json
- user-study/study-config.json
- user-study/data/marked sessions/CPSC Short Answer Marked - Sheet1.csv

Outputs:
- user-study/data/marked sessions/paper_level.csv
- user-study/data/marked sessions/question_level.csv
- user-study/data/marked sessions/cross_level.csv
- user-study/data/marked sessions/final_survey.csv

### D. Run statistical analysis

Script:
- user-study/scripts/run_stats.py

This script computes:
- Friedman tests for repeated-measures condition effects
- Pairwise Wilcoxon signed-rank post-hoc tests with Bonferroni correction
- Descriptive statistics by condition/group
- Order-effects tests by position

Outputs:
- user-study/data/marked sessions/statistical_results.json
- user-study/data/marked sessions/statistical_results.csv
- user-study/data/marked sessions/statistical_summary.md

### E. Generate paper figures

Script:
- user-study/scripts/generate_hci_figures.py

Reads flattened data and statistical outputs, then writes PNG figures to:
- user-study/exports/figures/

Current figure set includes:
- Total comprehension by condition
- MC first-attempt accuracy by condition
- Free-text order effect
- Key subjective ratings
- Optional within-subject condition comparison

## 3) Commands You Will Use Most

Run from repository root:

1. Flatten sessions into analysis CSVs

python3 user-study/scripts/flatten_sessions.py

2. Generate statistics

python3 user-study/scripts/run_stats.py

3. Generate figures

python3 user-study/scripts/generate_hci_figures.py

4. Export open-ended feedback only

python3 user-study/scripts/export_open_feedback.py

## 4) Where Specific Data Lives

### Raw data

- Session JSONs:
  - user-study/data/sessions/

### Processed tabular data

- Flattened analysis CSVs:
  - user-study/data/marked sessions/paper_level.csv
  - user-study/data/marked sessions/question_level.csv
  - user-study/data/marked sessions/cross_level.csv
  - user-study/data/marked sessions/final_survey.csv

### Statistical outputs

- user-study/data/marked sessions/statistical_results.json
- user-study/data/marked sessions/statistical_results.csv
- user-study/data/marked sessions/statistical_summary.md

### Figure outputs

- user-study/exports/figures/

### Highlight data

- Generated highlight JSONs:
  - user-study/data/highlights/
- Exported annotated PDFs:
  - user-study/exports/highlighted_pdfs/

## 5) Contextual Highlighter System (Plugin + Service)

## Plugin side

Primary files:
- contextual-highlighter/manifest.json
- contextual-highlighter/bootstrap.js

What it does:
- Adds Zotero collection menu actions
- Triggers Python phase scripts
- Loads and applies highlight rectangles in reader tabs
- Responds to Read tag updates by rerunning contextual ranking

Important note:
- contextual-highlighter/bootstrap.js currently contains machine-specific absolute paths for Python and service directories. Update these paths for your environment before use.

## Service side

Core pipeline files in service/:

- phase1.py
  - Extracts paper sentences, classifies rhetorical labels, computes embeddings, stores cache
- phase2.py
  - Ranks unread-paper sentences using read-paper context and writes highlights.json
- extractor.py
  - PDF sentence extraction (with header/footer removal and page cap)
- classifier.py
  - SciBERT-based rhetorical sentence classifier (with stub fallback if model weights missing)
- embedder.py
  - SPECTER2 embeddings
- ranker.py
  - TextRank-style ranking plus novelty-aware reranking
- cache.py
  - embeddings_cache.npz read/write utilities
- zotero_db.py
  - Zotero DB access helpers

Model and cache assets:
- service/scibert-acl-v3-final/
- service/scibert-acl-v2-final/
- service/scibert-pubmed-rtc-final/
- service/embeddings_cache.npz

## 6) Utility and Validation Scripts

Top-level scripts/:

- scripts/generate_all_highlights.py
  - Bulk generation of all and baseline contextual highlight files
- scripts/export_highlighted_pdfs.py
  - Writes annotated PDF exports from highlight JSONs
- scripts/verify_highlights.py
  - Verifies highlight rectangle/text matching quality
- scripts/verify_rects_bounds.py
  - Numeric and page-bounds sanity checks for rectangle coordinates
- scripts/inspect_cache.py
  - Quick inspection of embedding/classification cache contents

## 7) Typical Analysis Workflow

1. Collect sessions via study app (user-study/server.js)
2. Update manual grading CSV in user-study/data/marked sessions/
3. Run flatten_sessions.py
4. Run run_stats.py
5. Run generate_hci_figures.py
6. Use outputs in user-study/data/marked sessions/ and user-study/exports/figures/

## 8) Dependencies

Node app dependency:
- user-study/package.json (Express)

Python dependencies for highlighting/service pipeline:
- service/requirements.txt

Common analysis dependencies used by user-study scripts:
- pandas
- numpy
- scipy
- matplotlib

## 9) Quick Troubleshooting

- Missing or stale analysis outputs:
  - Re-run flatten_sessions.py first, then run_stats.py, then generate_hci_figures.py

- Figure labels or style not as expected:
  - Edit user-study/scripts/generate_hci_figures.py and rerun

- Plugin does not launch Python scripts:
  - Check absolute paths in contextual-highlighter/bootstrap.js

- Classifier not using trained weights:
  - Ensure service/scibert-acl-v3-final contains model.safetensors (or pytorch_model.bin)

## 10) Key Files at a Glance

Study pipeline:
- user-study/server.js
- user-study/scripts/flatten_sessions.py
- user-study/scripts/run_stats.py
- user-study/scripts/generate_hci_figures.py
- user-study/scripts/export_open_feedback.py

Highlighting pipeline:
- service/phase1.py
- service/phase2.py
- user-study/generate_highlights.py
- scripts/generate_all_highlights.py

Quality checks:
- scripts/verify_highlights.py
- scripts/verify_rects_bounds.py

This README is intended to be the main onboarding document for contributors working on study execution, data processing, and statistical reporting.
