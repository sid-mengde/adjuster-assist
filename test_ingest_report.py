"""Report chunking + ingestion tests. Ingestion uses a temp DB so the real one is never touched."""
import chromadb
import pytest

import ingest
from chunking import load_text
from chunking_report import chunk_inspection_report, chunk_report_text
from ingest import REPORT_COLLECTION, get_embedder, upsert_report_chunks

REPORT = "HVAC_Inspection_Report.pdf"


@pytest.fixture(scope="module")
def chunks():
    return chunk_inspection_report(REPORT)


def by_kind(chunks, kind):
    return [c for c in chunks if c.metadata["kind"] == kind]


def test_chunk_count_and_unique_ids(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == 13 and len(set(ids)) == 13


def test_header_metadata(chunks):
    md = by_kind(chunks, "header")[0].metadata
    assert md["claim_ref"] == "HW-88301-A" and md["doc_id"] == "hw-88301-a"
    assert md["inspection_date"] == "2026-10-17" and md["technician"] == "Marcus Vance"


def test_equipment_metadata(chunks):
    md = by_kind(chunks, "equipment")[0].metadata
    assert md["manufacturer"] == "Carrier" and md["model_number"] == "24ABC636A003"
    assert md["system_type"] == "Split System Air Conditioner (Condenser)"  # wrapped line rejoined


def test_grand_total_and_items(chunks):
    gt = by_kind(chunks, "grand_total")[0].metadata
    items = by_kind(chunks, "line_item")
    assert gt["grand_total"] == 1800.0 and gt["page"] == 2
    assert [i.metadata["line_total"] for i in items] == [950.0, 75.0, 325.0, 450.0]
    assert gt["total_matches_items"] is True and gt["items_sum"] == 1800.0


def test_item_spanning_page_break_is_page_2(chunks):
    labor = [c for c in by_kind(chunks, "line_item") if c.metadata["item_name"] == "Labor"][0]
    assert labor.metadata["page"] == 2 and "4 Hrs" in labor.text


def test_cause_of_failure_chunk(chunks):
    c = by_kind(chunks, "cause_of_failure")
    assert len(c) == 1 and "wear and tear" in c[0].text


def test_every_chunk_has_context_prefix_and_metadata(chunks):
    for c in chunks:
        assert c.text.startswith("[Report HW-88301-A")
        assert {"doc_type", "doc_id", "claim_ref", "section_title", "kind", "page"} <= set(c.metadata)


def test_mismatched_total_is_flagged():
    text, offsets = load_text(REPORT)
    bad = chunk_report_text(text.replace("$1,800.00\n4. Technician", "$1,900.00\n4. Technician"), offsets, "x.pdf")
    gt = by_kind(bad, "grand_total")[0].metadata
    assert gt["grand_total"] == 1900.0 and gt["total_matches_items"] is False


def dummy_report():
    """A second report: different claim ref, equipment and amounts."""
    text, offsets = load_text(REPORT)
    text = (text.replace("HW-88301-A", "HW-99999-Z").replace("Carrier", "Trane")
            .replace("$1,800.00", "$2,500.00").replace("$450.00", "$1,150.00"))
    return chunk_report_text(text, offsets, "dummy.pdf")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("db"))
    upsert_report_chunks(chunk_inspection_report(REPORT), db_path=path)
    return path


def col_for(path):
    return chromadb.PersistentClient(path=path).get_collection(REPORT_COLLECTION, embedding_function=get_embedder())


def test_ingest_then_retrieve_grand_total(db):
    r = col_for(db).query(query_texts=["What is the grand total of the repair quote?"], n_results=3)
    assert "hw-88301-a-grand-total" in r["ids"][0]


def test_second_report_coexists_and_filters_by_doc(db):
    upsert_report_chunks(dummy_report(), db_path=db)
    col = col_for(db)
    assert col.count() == 26
    totals = {m["claim_ref"]: m["grand_total"]
              for m in col.get(where={"kind": "grand_total"})["metadatas"]}
    assert totals == {"HW-88301-A": 1800.0, "HW-99999-Z": 2500.0}


def test_reingest_same_report_does_not_duplicate(db):
    before = col_for(db).count()
    upsert_report_chunks(chunk_inspection_report(REPORT), db_path=db)
    assert col_for(db).count() == before


def test_cli_ingests_multiple_pdfs_into_given_db(tmp_path):
    ingest.main([REPORT, REPORT, "--type", "inspection_report", "--db", str(tmp_path)])
    assert col_for(str(tmp_path)).count() == 13
