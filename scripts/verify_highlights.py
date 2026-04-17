#!/usr/bin/env python3
"""Verify highlight rects against PDFs.

For each JSON in user-study/data/highlights, find the corresponding PDF in
user-study/public/papers and verify each rect:
 - rect area > 0
 - the text extracted from the PDF within the rect contains (fuzzy) the
   highlight sentence (or at least shares several words)

Print a concise report per file and examples of failures.
"""
from pathlib import Path
import json
import fitz
import re

ROOT = Path(__file__).parent.parent
HIGHLIGHTS_DIR = ROOT / 'user-study' / 'data' / 'highlights'
PAPERS_DIR = ROOT / 'user-study' / 'public' / 'papers'


def guess_stem_from_fname(fname: str) -> str:
    if fname.endswith('_all'):
        return fname[:-4]
    if fname.endswith('_ctx_'):
        return fname[:-5]
    if '_ctx_' in fname:
        return fname.split('_ctx_')[0]
    if '_' in fname:
        return fname.rsplit('_', 1)[0]
    return fname


def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s or '').strip().lower()


def fuzzy_match(a: str, b: str, min_common=6) -> bool:
    # Return True if a and b share >= min_common alphanumeric characters sequence or words
    if not a or not b:
        return False
    na = re.sub(r'[^a-z0-9 ]', ' ', a.lower())
    nb = re.sub(r'[^a-z0-9 ]', ' ', b.lower())
    set_a = set(na.split())
    set_b = set(nb.split())
    common = set_a & set_b
    if len(common) >= min_common:
        return True
    # fallback: check if first 20 chars of a appear in b
    return na[:20].strip() and na[:20].strip() in nb


def find_pdf_for_stem(stem: str):
    cand = PAPERS_DIR / f"{stem}.pdf"
    if cand.exists():
        return cand
    for p in PAPERS_DIR.glob('*.pdf'):
        if p.stem.lower() == stem.lower():
            return p
    return None


if __name__ == '__main__':
    files = sorted(HIGHLIGHTS_DIR.glob('*.json'))
    if not files:
        print('No highlights found')
        raise SystemExit(1)

    total_files = len(files)
    report = []
    for f in files:
        data = json.loads(f.read_text(encoding='utf8'))
        stem = guess_stem_from_fname(f.stem)
        pdf = find_pdf_for_stem(stem)
        if pdf is None:
            report.append((f.name, 'PDF not found', 0, 0, 0, []))
            continue
        doc = fitz.open(str(pdf))
        file_issues = []
        rect_count = 0
        bad_rects = 0
        no_text_matches = 0
        for h in data:
            page_num = h.get('page')
            if page_num is None or page_num < 1 or page_num > len(doc):
                file_issues.append(f'bad page: {page_num} for sentence: {h.get("text","<no text>")[:40]}')
                continue
            page = doc[page_num - 1]
            for rect in h.get('rects', []):
                rect_count += 1
                try:
                    r = fitz.Rect(rect)
                except Exception as e:
                    bad_rects += 1
                    file_issues.append(f'invalid rect {rect} on p{page_num}: {e}')
                    continue
                w = r.width
                hgt = r.height
                if w <= 0 or hgt <= 0 or (w * hgt) < 1.0:
                    bad_rects += 1
                    file_issues.append(f'empty rect {rect} on p{page_num} for: {h.get("text","<no text>")[:40]}')
                    continue
                # extract text within rect (use clip)
                try:
                    txt = page.get_textbox(r)
                    if not txt or not fuzzy_match(h.get('text',''), txt):
                        no_text_matches += 1
                        file_issues.append(f'no match in rect {rect} on p{page_num}; txt="{txt[:80].replace("\n"," ")}"; want="{h.get("text","")[:80]}"')
                except Exception as e:
                    file_issues.append(f'error extracting text from rect {rect} on p{page_num}: {e}')
        doc.close()
        report.append((f.name, str(pdf.name), rect_count, bad_rects, no_text_matches, file_issues[:10]))

    # Summarize
    for name, pdfname, rect_count, bad_rects, no_matches, issues in report:
        print(f'{name}: pdf={pdfname} rects={rect_count} bad_rects={bad_rects} no_text_matches={no_matches}')
        for i in issues:
            print('  -', i)
    print('\nChecked', total_files, 'highlight files')
