"""Chunk PDFs and load them into local ChromaDB collections.

    python ingest.py                                   # the warranty contract (default)
    python ingest.py report1.pdf report2.pdf --type inspection_report
    python ingest.py contract.pdf --type contract
"""
import argparse

import chromadb
from chromadb.utils import embedding_functions

from chunking import chunk_document
from chunking_report import chunk_inspection_report

PDF = "home_warranty_contract.pdf"
DB_PATH = "./chroma_db"
COLLECTION = "warranty"
REPORT_COLLECTION = "inspection_reports"
MODEL = "BAAI/bge-small-en-v1.5"


def get_embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL)


def _add(col, chunks):
    col.add(ids=[c.id for c in chunks], documents=[c.text for c in chunks], metadatas=[c.metadata for c in chunks])


def build(path=PDF, db_path=DB_PATH):
    """Contract: single document, so rebuild the collection from scratch (idempotent)."""
    chunks = chunk_document(path)
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, embedding_function=get_embedder(), metadata={"hnsw:space": "cosine"})
    _add(col, chunks)
    print(f"Ingested {col.count()} chunks into '{COLLECTION}' at {db_path}")
    return len(chunks)


def upsert_report_chunks(chunks, db_path=DB_PATH):
    """Replace one report's chunks (matched by doc_id) without touching other reports."""
    doc_ids = {c.metadata["doc_id"] for c in chunks}
    assert len(doc_ids) == 1, f"chunks span several documents: {doc_ids}"
    doc_id = doc_ids.pop()
    col = chromadb.PersistentClient(path=db_path).get_or_create_collection(
        REPORT_COLLECTION, embedding_function=get_embedder(), metadata={"hnsw:space": "cosine"})
    col.delete(where={"doc_id": doc_id})
    _add(col, chunks)
    return doc_id, col.count()


def ingest_report(path, db_path=DB_PATH):
    chunks = chunk_inspection_report(path)
    kinds = {c.metadata["kind"] for c in chunks}
    for needed in ("equipment", "grand_total"):  # the claim pipeline depends on these
        if needed not in kinds:
            print(f"WARNING: {path}: no '{needed}' chunk found; check the report layout")
    doc_id, total = upsert_report_chunks(chunks, db_path)
    print(f"Ingested {path} as '{doc_id}' ({len(chunks)} chunks); '{REPORT_COLLECTION}' now holds {total}")
    return doc_id


DOC_TYPES = {"contract": build, "inspection_report": ingest_report}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", help=f"PDF file(s); defaults to {PDF}")
    ap.add_argument("--type", choices=DOC_TYPES, default="contract")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args(argv)
    for pdf in args.pdfs or [PDF]:
        DOC_TYPES[args.type](pdf, db_path=args.db)


if __name__ == "__main__":
    main()
