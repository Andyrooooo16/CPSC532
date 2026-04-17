#!/usr/bin/env python3
"""Quick verify highlight rect numeric sanity and page-bounds.
"""
from pathlib import Path
import json
import math
import fitz

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
    any_issues = False
    for f in files:
        data = json.loads(f.read_text(encoding='utf8'))
        stem = guess_stem_from_fname(f.stem)
        pdf = find_pdf_for_stem(stem)
        if pdf is None:
            print(f'{f.name}: PDF not found for stem {stem}')
            continue
        doc = fitz.open(str(pdf))
        issues = []
        for h in data:
            pnum = h.get('page')
            if pnum is None or pnum < 1 or pnum > len(doc):
                issues.append(f'bad page {pnum}')
                continue
            page = doc[pnum - 1]
            width = page.rect.width
            height = page.rect.height
            for rect in h.get('rects', []):
                if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                    issues.append(f'invalid rect shape {rect} p{pnum}')
                    continue
                if any(not isinstance(x, (int, float)) or math.isinf(x) or math.isnan(x) for x in rect):
                    issues.append(f'non-finite rect {rect} p{pnum}')
                    continue
                x0, y0, x1, y1 = rect
                # check reasonable bounds (allow small negative epsilon for PDF margins)
                eps = 1e-3
                if x0 < -eps or y0 < -eps or x1 > width + eps or y1 > height + eps:
                    issues.append(f'out-of-bounds rect {rect} p{pnum} page_size=({width:.1f},{height:.1f})')
                    continue
                if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                    issues.append(f'zero-area rect {rect} p{pnum}')
        doc.close()
        if issues:
            any_issues = True
            print(f'FILE {f.name} issues:')
            for it in issues[:20]:
                print('  -', it)
            if len(issues) > 20:
                print(f'  ... and {len(issues)-20} more')
    if not any_issues:
        print('No numeric/bounds issues detected in any highlight file.')
