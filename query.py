"""Query the local warranty collection. Usage: python query.py "your question" """
import sys

import chromadb

from ingest import COLLECTION, DB_PATH, get_embedder


def get_collection(name=COLLECTION):
    return chromadb.PersistentClient(path=DB_PATH).get_collection(
        name, embedding_function=get_embedder()
    )


def search(query, n=3, where=None, col=None):
    """Return a list of dicts (id, text, metadata, distance); lower distance is closer."""
    col = col or get_collection()
    r = col.query(query_texts=[query], n_results=n, where=where)
    return [
        {"id": i, "text": d, "metadata": m, "distance": dist}
        for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


if __name__ == "__main__":
    for hit in search(" ".join(sys.argv[1:])):
        md = hit["metadata"]
        print(f"\n[{hit['distance']:.3f}] {hit['id']} (p{md['page']}, §{md['section_number']} {md['section_title']})")
        print(hit["text"][:300])
