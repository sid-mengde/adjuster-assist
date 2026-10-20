"""A. Ingestion and parsing (deterministic)."""
import contextlib
import io
import tempfile

import chromadb

import ingest
from chunking import chunk_document
from chunking_report import chunk_inspection_report, chunk_report_text
from evals import documents as D, fixtures
from evals.common import load, row
from evals.config import CONTRACT_PDF, REPORT_PDF

PARSEABLE = [r for r in D.REPORTS if r in ("base", "d_dup_items", "e_dishwasher") or r.startswith(("q_", "c_", "e_", "m_", "i_", "l_nocomma",
             "l_heading_quote", "l_usd_suffix", "l_qty_ea", "l_lower_grand", "d_no_cause", "d_no_equipment")) and r != "q_discount"]
LAYOUT_CASES = ["l_nocomma", "l_heading_quote", "l_cost_breakdown", "l_usd_suffix", "l_qty_ea", "l_lower_grand",
                "q_discount", "d_total_mismatch", "d_no_grand_total"]


def _kind(chunks, kind):
    return [c for c in chunks if c.metadata["kind"] == kind]


def _num(x):
    return float(str(x).replace("$", "").replace(",", ""))


def run(ctx):
    rows, gold = [], load("golden.json")
    real_contract, real_report = chunk_document(str(CONTRACT_PDF)), chunk_inspection_report(str(REPORT_PDF))

    # A1 contract chunks: real PDF and every synthetic variant keep the same chunk structure and declarations
    exp_ids = [g["id"] for g in gold["contract"]]
    rows.append(row("A1", "real_pdf", [c.id for c in real_contract] == exp_ids, got=[c.id for c in real_contract]))
    for cid, opts in D.CONTRACTS.items():
        ch = D.contract_chunks(cid)
        decl = _kind(ch, "declarations")[0].metadata
        want = {"plan_type": opts.get("plan", "Combo (Systems & Appliances)"),
                "coverage_period": f"{opts.get('term_months', 12)} Months (From Activation)",
                "service_fee": f"${opts.get('fee', 85):.2f} per trade call"}
        got = {k: decl.get(k) for k in want}
        rows.append(row("A1", cid, [c.id for c in ch] == exp_ids and got == want, want=want, got=got))

    # A2 report chunk structure: real PDF, and the template's base variant must equal it
    exp = [(g["id"], g["kind"]) for g in gold["report"]]
    rows.append(row("A2", "real_pdf", [(c.id, c.metadata["kind"]) for c in real_report] == exp))
    rows.append(row("A2", "template_equals_real_pdf", [(c.id, c.text) for c in D.report_chunks("base")] == [(c.id, c.text) for c in real_report]))

    # A3 line items, A4 total consistency flag
    for rid in PARSEABLE:
        ch = D.report_chunks(rid)
        items = [(c.metadata["item_name"], c.metadata["line_total"]) for c in _kind(ch, "line_item")]
        want = [(i["name"], _num(i["total"])) for i in D.REPORTS[rid]["items"]]
        rows.append(row("A3", rid, items == want, want=want, got=items))
    for rid in D.REPORTS:
        gt = _kind(D.report_chunks(rid), "grand_total")
        if not gt or rid == "q_discount":  # the negative line is unreadable by design; A9 checks that it is flagged
            continue
        md, stated, s = gt[0].metadata, D.report_stated_total(rid), D.report_items_sum(rid)
        rows.append(row("A4", rid, md["grand_total"] == stated and md["total_matches_items"] == (abs(stated - s) < 0.005),
                        stated=stated, items_sum=s, got=(md["grand_total"], md["total_matches_items"])))

    # A5 page numbers against the golden file
    rows += [row("A5", f"contract:{g['id']}", next(c for c in real_contract if c.id == g["id"]).metadata["page"] == g["page"])
             for g in gold["contract"]]
    rows += [row("A5", f"report:{g['id']}", next(c for c in real_report if c.id == g["id"]).metadata["page"] == g["page"])
             for g in gold["report"]]

    # A6 re-ingest replaces one report's chunks without touching another's
    saved, ingest.get_embedder = ingest.get_embedder, fixtures.embedder
    try:
        with tempfile.TemporaryDirectory() as db:
            col = lambda: chromadb.PersistentClient(path=db).get_collection(ingest.REPORT_COLLECTION, embedding_function=fixtures.embedder())
            n_base, n_b = len(D.report_chunks("base")), len(D.report_chunks("m_b"))
            ingest.upsert_report_chunks(D.report_chunks("base"), db)
            rows.append(row("A6", "first_ingest", col().count() == n_base))
            ingest.upsert_report_chunks(D.report_chunks("base"), db)
            rows.append(row("A6", "idempotent", col().count() == n_base))
            ingest.upsert_report_chunks(D.report_chunks("m_b"), db)
            rows.append(row("A6", "second_report_added", col().count() == n_base + n_b))
            ingest.upsert_report_chunks(D.report_chunks("base"), db)
            ids = set(col().get()["ids"])
            rows.append(row("A6", "other_report_untouched", col().count() == n_base + n_b
                            and {c.id for c in D.report_chunks("m_b")} <= ids))
            # A7 a layout the pipeline cannot read must warn, a good one must not
            real_chunker = ingest.chunk_inspection_report
            for rid, key in [("base", None), ("d_no_equipment", "equipment"), ("d_no_grand_total", "grand_total"),
                             ("l_cost_breakdown", "grand_total")]:
                ingest.chunk_inspection_report = lambda path, rid=rid: D.report_chunks(rid)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ingest.ingest_report("x.pdf", db_path=db)
                out = buf.getvalue()
                rows.append(row("A7", rid, ("WARNING" in out and key in out) if key else "WARNING" not in out, output=out.strip()))
            ingest.chunk_inspection_report = real_chunker
    finally:
        ingest.get_embedder = saved

    # A8 the claim ref is recognised in several spellings
    text, offsets = D.report_text("base")
    for label, sub in [("with_hash", "#HW-88301-A"), ("no_hash", "HW-88301-A"), ("lowercase", "#hw-88301-a")]:
        ch = chunk_report_text(text.replace("#HW-88301-A", sub), offsets, "x.txt")
        rows.append(row("A8", label, {c.metadata["doc_id"] for c in ch} == {"hw-88301-a"}, got=sorted({c.metadata["doc_id"] for c in ch})))

    # A9 a layout we cannot parse must be flagged, never silently wrong
    silent = 0
    for rid in LAYOUT_CASES:
        ch = D.report_chunks(rid)
        gt = _kind(ch, "grand_total")
        stated, true_sum = D.report_stated_total(rid), D.report_items_sum(rid)
        items_ok = [(c.metadata["item_name"], c.metadata["line_total"]) for c in _kind(ch, "line_item")] == [(i["name"], _num(i["total"])) for i in D.REPORTS[rid]["items"]]
        if not gt:
            outcome = "flagged_no_total"
        elif gt[0].metadata["grand_total"] == stated and items_ok:
            outcome = "parsed_correctly"
        elif not gt[0].metadata["total_matches_items"]:
            outcome = "flagged_inconsistent"
        else:
            outcome = "silent_wrong"
        # a stated total that differs from the true line items is itself a defect that must be flagged
        if outcome == "parsed_correctly" and abs(stated - true_sum) >= 0.005 and gt[0].metadata["total_matches_items"]:
            outcome = "silent_wrong"
        silent += outcome == "silent_wrong"
        rows.append(row("A9", rid, outcome != "silent_wrong", outcome=outcome))
    ctx.extra["A9.silent_wrong"] = silent
    return rows
