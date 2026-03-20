import sqlite3
import os
from pathlib import Path

ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"
ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"


def _connect():
    """Open a read-only connection to the Zotero database."""
    # immutable=1 allows reading while Zotero has the DB open
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_collection_id(collection_name: str) -> int | None:
    """Return the collectionID for a given collection name, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT collectionID FROM collections WHERE collectionName = ?",
            (collection_name,),
        ).fetchone()
        return row["collectionID"] if row else None


def get_items_with_pdfs(collection_id: int) -> list[dict]:
    """
    Return all items in a collection that have a PDF attachment.

    Each entry contains:
        item_key       - the parent paper's Zotero key (e.g. "49SELEVZ")
        attachment_key - the attachment item's key (used as the storage folder name)
        pdf_path       - absolute path to the PDF file on disk
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                parent.key        AS item_key,
                att.key           AS attachment_key,
                ia.path           AS raw_path
            FROM collectionItems ci
            JOIN items parent ON ci.itemID = parent.itemID
            JOIN itemAttachments ia
                ON ia.parentItemID = parent.itemID
                AND ia.contentType = 'application/pdf'
            JOIN items att ON att.itemID = ia.itemID
            WHERE ci.collectionID = ?
            """,
            (collection_id,),
        ).fetchall()

    results = []
    for row in rows:
        # raw_path is like "storage:filename.pdf" — strip the prefix
        filename = row["raw_path"].removeprefix("storage:")
        pdf_path = ZOTERO_STORAGE / row["attachment_key"] / filename
        if pdf_path.exists():
            results.append({
                "item_key": row["item_key"],
                "attachment_key": row["attachment_key"],
                "pdf_path": str(pdf_path),
            })
        else:
            print(f"[zotero_db] Warning: PDF not found at {pdf_path}")

    return results


def get_read_item_keys(collection_id: int, tag_name: str = "Read") -> set[str]:
    """
    Return the set of item keys in a collection that have the given tag.
    Used to identify papers the user has already read.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT parent.key AS item_key
            FROM collectionItems ci
            JOIN items parent ON ci.itemID = parent.itemID
            JOIN itemTags it ON it.itemID = parent.itemID
            JOIN tags t ON t.tagID = it.tagID
            WHERE ci.collectionID = ?
              AND LOWER(t.name) = LOWER(?)
            """,
            (collection_id, tag_name),
        ).fetchall()

    return {row["item_key"] for row in rows}


if __name__ == "__main__":
    # Quick sanity check
    col_id = get_collection_id("Class - Human-Centered AI")
    print(f"Collection ID: {col_id}")
    if col_id:
        items = get_items_with_pdfs(col_id)
        print(f"Items with PDFs: {len(items)}")
        for item in items[:3]:
            print(f"  {item['item_key']} → {item['pdf_path']}")
        read = get_read_item_keys(col_id)
        print(f"Read items: {read}")

        # Inspect full-text storage for the first item
        first = items[0]
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT fc.content
                FROM items att
                JOIN fulltextContent fc ON fc.itemID = att.itemID
                WHERE att.key = ?
                """,
                (first["attachment_key"],),
            ).fetchone()

        if row:
            print(f"\n--- fulltextContent for {first['item_key']} (first 500 chars) ---")
            print(row["content"][:500])
        else:
            print(f"\nNo fulltextContent row found for {first['item_key']}")
            # Check if fulltext.txt exists on disk instead
            import pathlib
            txt = pathlib.Path(first["pdf_path"]).parent / "fulltext.txt"
            print(f"fulltext.txt exists on disk: {txt.exists()}")
            if txt.exists():
                print(txt.read_text()[:500])
