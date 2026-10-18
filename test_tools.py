from datetime import date

import pytest

from tools import check_waiting_period, get_limits, match_component

ACT = date(2026, 1, 15)


# 1. system vs appliance matching against plan tier
@pytest.mark.parametrize("component,plan,verdict,item", [
    ("HVAC compressor", None, "covered", "Heating & Air Conditioning Systems"),
    ("plumbing branch line", None, "covered", "Plumbing System"),
    ("circuit breaker", None, "covered", "Electrical System"),
    ("dishwasher", None, "covered", "Major Home Appliances"),
    ("HVAC compressor", "Systems Only", "covered", "Heating & Air Conditioning Systems"),
    ("dishwasher", "Systems Only", "not_in_plan_tier", "Major Home Appliances"),
    ("refrigerator", "Appliances Only", "covered", "Major Home Appliances"),
    ("water heater leak in drain line", "Appliances Only", "not_in_plan_tier", "Plumbing System"),
])
def test_match_component(component, plan, verdict, item):
    r = match_component(component, plan)
    assert r["verdict"] == verdict and r["matched_item"] == item, r
    assert r["citations"]


def test_unlisted_component_is_no_match():
    r = match_component("What's the capital of France?")
    assert r["verdict"] == "no_match" and r["matched_item"] is None


# 2. waiting period
@pytest.mark.parametrize("breakdown,verdict", [
    (date(2026, 1, 10), "before_activation"),
    (date(2026, 1, 15), "within_waiting_period"),   # day 0
    (date(2026, 2, 13), "within_waiting_period"),   # day 29
    (date(2026, 2, 14), "eligible"),                # day 30
    (date(2026, 12, 31), "eligible"),
    (date(2027, 1, 15), "after_term"),              # 12 months from activation
])
def test_waiting_period(breakdown, verdict):
    r = check_waiting_period(breakdown, ACT)
    assert r["verdict"] == verdict and r["eligible"] == (verdict == "eligible"), r
    assert r["waiting_days"] == 30


# 3. caps and limits
def test_hvac_cap():
    r = get_limits("HVAC compressor")
    assert r["cap"] == {"amount": 2000.0, "applies": "per contract term"}
    assert r["service_fee"]["amount"] == 85.0


def test_uncapped_item_reports_none():
    r = get_limits("plumbing leak")
    assert r["cap"] is None and "No dollar cap" in r["reason"]


def test_list_all_caps():
    r = get_limits()
    assert [c["item"] for c in r["caps"]] == ["Heating & Air Conditioning Systems"]


# 4. repair quote (reads the real DB, which holds report HW-88301-A)
from tools import get_repair_quote


def test_quote_default_single_report():
    r = get_repair_quote()
    assert r["verdict"] == "found" and r["grand_total"] == 1800.0 and r["claim_ref"] == "HW-88301-A"
    assert r["consistent"] is True and "warning" not in r
    assert [i["total"] for i in r["line_items"]] == [950.0, 75.0, 325.0, 450.0]
    c = r["citations"][0]
    assert c["page"] == 2 and c["document"] == "Inspection report HW-88301-A"


@pytest.mark.parametrize("ref", ["HW-88301-A", "#HW-88301-A", " hw-88301-a "])
def test_quote_claim_ref_is_normalized(ref):
    assert get_repair_quote(ref)["grand_total"] == 1800.0


def test_quote_unknown_claim_ref():
    r = get_repair_quote("HW-00000-X")
    assert r["verdict"] == "not_found" and r["available_claim_refs"] == ["HW-88301-A"] and "grand_total" not in r


@pytest.fixture
def tmp_reports(tmp_path, monkeypatch):
    """Point the tool at a temp DB so ambiguity/mismatch cases never touch the real one."""
    import chromadb, tools
    from chunking import load_text
    from chunking_report import chunk_inspection_report, chunk_report_text
    from ingest import REPORT_COLLECTION, get_embedder, upsert_report_chunks
    db = str(tmp_path)
    upsert_report_chunks(chunk_inspection_report("HVAC_Inspection_Report.pdf"), db_path=db)
    monkeypatch.setattr(tools, "_report_collection", lambda: chromadb.PersistentClient(path=db).get_collection(
        REPORT_COLLECTION, embedding_function=get_embedder()))
    return db, load_text("HVAC_Inspection_Report.pdf"), chunk_report_text, upsert_report_chunks


def test_quote_ambiguous_with_two_reports(tmp_reports):
    db, (text, offsets), chunker, upsert = tmp_reports
    upsert(chunker(text.replace("HW-88301-A", "HW-99999-Z").replace("$1,800.00", "$2,500.00")
                   .replace("$450.00", "$1,150.00"), offsets, "b.pdf"), db_path=db)
    r = get_repair_quote()
    assert r["verdict"] == "ambiguous" and r["available_claim_refs"] == ["HW-88301-A", "HW-99999-Z"]
    assert get_repair_quote("HW-99999-Z")["grand_total"] == 2500.0


def test_quote_mismatch_is_warned(tmp_reports):
    db, (text, offsets), chunker, upsert = tmp_reports
    upsert(chunker(text.replace("$1,800.00\n4. Technician", "$1,900.00\n4. Technician"), offsets, "a.pdf"), db_path=db)
    r = get_repair_quote()
    assert r["grand_total"] == 1900.0 and r["consistent"] is False and "does not equal" in r["warning"]


def test_quote_no_reports(tmp_path, monkeypatch):
    import chromadb, tools
    from ingest import REPORT_COLLECTION, get_embedder
    col = chromadb.PersistentClient(path=str(tmp_path)).create_collection(REPORT_COLLECTION, embedding_function=get_embedder())
    monkeypatch.setattr(tools, "_report_collection", lambda: col)
    assert get_repair_quote()["verdict"] == "no_reports"
