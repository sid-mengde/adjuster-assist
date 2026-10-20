"""Temporary vector stores for contract and report variants, swapped in behind the real tools.

The tools cache their collections in module globals (`tools._col`, `tools._report_col`), so an environment is
entered by replacing those, and the previous values are restored on exit. The real ./chroma_db is never touched.
"""
import atexit
import hashlib
import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache

import chromadb

import tools
from evals import documents
from ingest import COLLECTION, REPORT_COLLECTION, get_embedder

_tmp = tempfile.mkdtemp(prefix="adjuster-evals-")
atexit.register(shutil.rmtree, _tmp, ignore_errors=True)
_cache = {}


@lru_cache(maxsize=1)
def embedder():
    return get_embedder()  # loading the model is the slow part, so it is shared by every fixture


def _collection(kind, key, name, chunks):
    if key not in _cache:
        path = f"{_tmp}/{kind}-{hashlib.md5(repr(key).encode()).hexdigest()[:10]}"
        col = chromadb.PersistentClient(path=path).create_collection(
            name, embedding_function=embedder(), metadata={"hnsw:space": "cosine"})
        if chunks:
            col.add(ids=[c.id for c in chunks], documents=[c.text for c in chunks], metadatas=[c.metadata for c in chunks])
        _cache[key] = col
    return _cache[key]


def contract_collection(contract_id="base"):
    return _collection("c", ("contract", contract_id), COLLECTION, documents.contract_chunks(contract_id))


def report_collection(report_ids=("base",)):
    ids = tuple(sorted(report_ids))
    chunks = [c for rid in ids for c in documents.report_chunks(rid)]
    return _collection("r", ("reports", ids), REPORT_COLLECTION, chunks)


@contextmanager
def environment(contract="base", reports=("base",)):
    """Run the tools against a given contract variant and set of loaded reports."""
    saved = tools._col, tools._report_col
    tools._col, tools._report_col = contract_collection(contract), report_collection(tuple(reports))
    try:
        yield
    finally:
        tools._col, tools._report_col = saved
