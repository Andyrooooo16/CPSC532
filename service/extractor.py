import re
import nltk
import fitz  # PyMuPDF

# Fraction of page height to treat as header/footer zone.
# Blocks whose top edge falls in the top MARGIN or bottom MARGIN are discarded.
MARGIN = 0.10


def extract_sentences(pdf_path: str) -> list[str]:
    """
    Extract sentences from a PDF file.

    Steps:
        1. Open the PDF with PyMuPDF.
        2. For each page, collect text blocks outside the header/footer margins.
        3. Join all kept blocks into one string.
        4. Split into sentences with nltk.sent_tokenize.

    Returns a list of sentence strings.
    """
    doc = fitz.open(pdf_path)
    all_text_parts = []

    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc[page_num]
        print(f"  Page {page_num + 1}/{total_pages}", end="\r", flush=True)
        page_height = page.rect.height
        top_cutoff = page_height * MARGIN
        bottom_cutoff = page_height * (1 - MARGIN)

        # get_text("blocks") returns a list of:
        # (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text, 1 = image
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type != 0:
                continue  # skip image blocks
            if y0 < top_cutoff or y1 > bottom_cutoff:
                continue  # skip header/footer zone
            all_text_parts.append(text.strip())

    print()  # newline after page progress
    doc.close()

    full_text = " ".join(all_text_parts)

    # Rejoin words split across lines with a hyphen (e.g. "in-\nteractive" → "interactive")
    full_text = re.sub(r"-\s*\n\s*", "", full_text)
    sentences = nltk.sent_tokenize(full_text)
    return sentences


if __name__ == "__main__":
    # Quick sanity check — run on the first PDF in your Zotero collection
    import sys
    sys.path.insert(0, ".")
    from zotero_db import get_collection_id, get_items_with_pdfs

    nltk.download("punkt_tab", quiet=True)

    col_id = get_collection_id("Class - Human-Centered AI")
    if col_id is None:
        print("Collection not found")
        sys.exit(1)
    items = get_items_with_pdfs(col_id)
    first = items[0]

    print(f"Extracting from: {first['pdf_path']}")
    sentences = extract_sentences(first["pdf_path"])
    print(f"Total sentences: {len(sentences)}\n")
    print("--- First 10 sentences ---")
    for s in sentences[:10]:
        print(f"  {s}")
