import pytest

from query import get_collection, search

TOP_K = 3


@pytest.fixture(scope="module")
def col():
    return get_collection()


def ids(hits):
    return [h["id"] for h in hits]


# (query, acceptable chunk ids) -- pass if any acceptable id is in the top TOP_K
LOOKUPS = [
    # A. direct facts
    ("What is the HVAC coverage cap?", {"sec2-1"}),
    ("How much is the service fee?", {"sec1-declarations", "sec4-1"}),
    ("How long does the coverage period last?", {"sec1-declarations"}),
    ("What is the agreement number?", {"sec1-declarations"}),
    ("What plan type do I have?", {"sec1-declarations"}),
    # B. coverage
    ("Are leaking pipes covered?", {"sec2-2"}),
    ("Does it cover the circuit breaker panel?", {"sec2-3"}),
    ("Is my dishwasher covered?", {"sec2-4"}),
    # C. exclusions
    ("Is a pre-existing condition covered?", {"sec3-1"}),
    ("Is flood or storm damage covered?", {"sec3-2"}),
    ("What if I never serviced my furnace?", {"sec3-3"}),
    ("Is there a waiting period?", {"sec3-1"}),
    # D. claims process
    ("How do I file a claim?", {"sec4-0"}),
    ("Can I call my own plumber first?", {"sec4-0"}),
    ("What happens after the technician diagnoses the problem?", {"sec4-2"}),
]


@pytest.mark.parametrize("query,expected", LOOKUPS, ids=[q for q, _ in LOOKUPS])
def test_lookup(col, query, expected):
    got = ids(search(query, n=TOP_K, col=col))
    assert expected & set(got), f"expected one of {expected} in top {TOP_K}, got {got}"


# E. metadata filters
def test_filter_section(col):
    hits = search("fire damage", n=TOP_K, where={"section_number": 3}, col=col)
    assert hits and all(h["metadata"]["section_number"] == 3 for h in hits)


def test_filter_declarations(col):
    hits = search("fees", n=TOP_K, where={"kind": "declarations"}, col=col)
    assert len(hits) == 1
    assert hits[0]["metadata"]["service_fee"] == "$85.00 per trade call"


def test_filter_page_two(col):
    hits = search("anything", n=TOP_K, where={"page": 2}, col=col)
    assert ids(hits) == ["sec4-2"]


# F. negative queries: printed for reference only (run with -s), never fail
@pytest.mark.parametrize("query", ["Does it cover my swimming pool pump?", "What's the capital of France?"])
def test_negative_reference(col, query):
    hits = search(query, n=TOP_K, col=col)
    print(f"\nNEGATIVE: {query!r}")
    for h in hits:
        print(f"  {h['distance']:.3f}  {h['id']}")


# G. pipeline sanity
EXPECTED_COUNT = 14


def test_count_and_unique_ids(col):
    got = col.get()["ids"]
    assert len(got) == EXPECTED_COUNT and len(set(got)) == EXPECTED_COUNT


def test_metadata_fields(col):
    required = {"source", "section_number", "section_title", "kind", "page"}
    for md in col.get()["metadatas"]:
        assert required <= set(md)


def test_context_prefix(col):
    for doc in col.get()["documents"]:
        assert doc.startswith("[Section ")
