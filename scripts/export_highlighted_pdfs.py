#!/usr/bin/env python3
"""Export annotated PDFs for each highlight JSON.

- Reads highlight JSONs from user-study/data/highlights/*.json
- Finds the matching PDF in user-study/public/papers/
- Adds semi-transparent rectangle annotations for each highlight rect using PyMuPDF
- Writes annotated PDFs to user-study/exports/highlighted_pdfs/

Run:
    python3 scripts/export_highlighted_pdfs.py

"""
from pathlib import Path
import json
import fitz  # PyMuPDF

ROOT = Path(__file__).parent.parent
HIGHLIGHTS_DIR = ROOT / 'user-study' / 'data' / 'highlights'
PAPERS_DIR = ROOT / 'user-study' / 'public' / 'papers'
OUT_DIR = ROOT / 'user-study' / 'exports' / 'highlighted_pdfs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color map adapted from public/app/viewer.js: rgba(R,G,B,A)
LABEL_COLORS = {
    'OBJECTIVE': (144/255, 238/255, 144/255, 0.45),
    'BACKGROUND': (255/255, 200/255, 80/255, 0.40),
    'METHODS': (100/255, 180/255, 255/255, 0.40),
    'RESULTS': (255/255, 150/255, 150/255, 0.45),
    'CONCLUSIONS': (200/255, 150/255, 255/255, 0.40),
    'NONE': (200/255, 200/255, 200/255, 0.35),
    'DEFAULT': (255/255, 220/255, 0/255, 0.38),
}


def guess_stem_from_fname(fname: str) -> str:
    # fname is a file stem like 'CiteSee_all' or 'S1 Novel Content_ctx_' or 'Scim_ctx_paper2'
    if fname.endswith('_all'):
        return fname[:-4]
    if fname.endswith('_ctx_'):
        return fname[:-5]
    if '_ctx_' in fname:
        return fname.split('_ctx_')[0]
    # fallback: take prefix before last underscore
    if '_' in fname:
        return fname.rsplit('_', 1)[0]
    return fname


def find_pdf_for_stem(stem: str) -> Path | None:
    # Try exact match
    cand = PAPERS_DIR / f"{stem}.pdf"
    if cand.exists():
        return cand
    # try case-insensitive match
    for p in PAPERS_DIR.glob('*.pdf'):
        if p.stem.lower() == stem.lower():
            return p
    return None


def color_for_label(label: str):
    return LABEL_COLORS.get(label, LABEL_COLORS['DEFAULT'])


def process_highlight_file(hpath: Path):
    try:
        data = json.loads(hpath.read_text(encoding='utf8'))
    except Exception as e:
        print(f"Failed to read {hpath}: {e}")
        return None

    stem = guess_stem_from_fname(hpath.stem)
    pdf = find_pdf_for_stem(stem)
    if pdf is None:
        print(f"PDF not found for stem '{stem}' (from {hpath.name})")
        return None

    out_pdf = OUT_DIR / f"{hpath.stem}.pdf"

    doc = fitz.open(str(pdf))

    for h in data:
        page_num = h.get('page', None)
        if page_num is None:
            continue
        # Page numbers in highlights are 1-indexed
        try:
            page = doc[page_num - 1]
        except Exception:
            print(f"  Warning: page {page_num} out of range for {pdf.name}")
            continue

        label = h.get('label', 'NONE')
        r = color_for_label(label)
        rgb = (r[0], r[1], r[2])
        alpha = r[3]

        rects = h.get('rects', [])
        for rect in rects:
            try:
                rect_obj = fitz.Rect(rect)
                annot = page.add_rect_annot(rect_obj)
                annot.set_colors(stroke=None, fill=rgb)
                annot.set_opacity(alpha)
                annot.update()
            except Exception as e:
                print(f"    Failed to add annot on {pdf.name} p{page_num}: {e}")

    # Save a copy
    try:
        doc.save(str(out_pdf))
        doc.close()
        print(f"Wrote {out_pdf}")
        return out_pdf
    except Exception as e:
        print(f"Failed to save {out_pdf}: {e}")
        return None


if __name__ == '__main__':
    hfiles = sorted(HIGHLIGHTS_DIR.glob('*.json'))
    if not hfiles:
        print('No highlight JSON files found in', HIGHLIGHTS_DIR)
        raise SystemExit(1)

    written = []
    for h in hfiles:
        print('Processing', h.name)
        out = process_highlight_file(h)
        if out:
            written.append(out)

    print('\nDone. Generated', len(written), 'annotated PDFs in', OUT_DIR)
