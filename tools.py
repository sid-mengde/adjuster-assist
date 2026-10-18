"""Claim-verification tools backed by the vectorized warranty contract.

Each tool returns a dict with a verdict plus `citations` (chunk id, page, section)
so an adjuster can trace every answer back to the contract text.
"""
import re
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

from ingest import REPORT_COLLECTION
from query import get_collection, search

# Max cosine distance for a match to a coverage item. Measured: in-scope probes 0.22-0.34,
# out-of-scope (pool pump, roof, garage door) 0.44-0.49. Re-tune with real claims.
MATCH_MAX_DISTANCE = 0.40

_col = None
_report_col = None


def _collection():
    global _col
    _col = _col or get_collection()
    return _col


def _cite(chunk_or_hit):
    md = chunk_or_hit["metadata"]
    cite = {"chunk": chunk_or_hit["id"], "page": md["page"],
            "section": f"{md['section_number']}. {md['section_title']}"}
    if md.get("doc_type") == "inspection_report":  # section numbers repeat across documents
        cite["document"] = f"Inspection report {md['claim_ref']}"
    return cite


def _report_collection():
    global _report_col
    _report_col = _report_col or get_collection(REPORT_COLLECTION)
    return _report_col


def _declarations():
    hit = search("contract declarations", n=1, where={"kind": "declarations"}, col=_collection())[0]
    return hit


def _coverage_items():
    """Section 2 bullets: [{id, label, category, text, hit}] where category is systems|appliances."""
    r = _collection().get(where={"$and": [{"section_number": 2}, {"kind": "bullet"}]})
    items = []
    for i, doc, md in zip(r["ids"], r["documents"], r["metadatas"]):
        body = doc.split("] ", 1)[1]
        label = body.split(":", 1)[0]
        items.append({"id": i, "label": label, "text": body, "metadata": md,
                      "category": "appliances" if "appliance" in label.lower() else "systems"})
    return items


def _plan_allows(plan_type):
    """'Combo (Systems & Appliances)' -> {'systems','appliances'}; 'Systems Only' -> {'systems'}."""
    p = plan_type.lower()
    if "combo" in p:
        return {"systems", "appliances"}
    allowed = set()
    if "system" in p:
        allowed.add("systems")
    if "appliance" in p:
        allowed.add("appliances")
    return allowed


def _caps(text):
    """Dollar caps in a coverage bullet, e.g. 'maximum cap of $2,000 per contract term'."""
    m = re.search(r"(?:maximum|max\.?)\s+cap\s+of\s+\$([\d,]+(?:\.\d\d)?)\s*(per [a-z ]+?)?(?:[.;]|$)", text, re.I)
    return {"amount": float(m.group(1).replace(",", "")), "applies": (m.group(2) or "").strip()} if m else None


# ---------------------------------------------------------------- tool 1
def match_component(component, plan_type=None):
    """Is the broken component covered under the customer's plan tier?

    verdict: covered | not_in_plan_tier | no_match (needs human review)
    plan_type defaults to the one in the contract declarations.
    """
    decl = _declarations()
    plan_type = plan_type or decl["metadata"]["plan_type"]
    allowed = _plan_allows(plan_type)
    items = {it["id"]: it for it in _coverage_items()}

    hits = search(component, n=1,
                  where={"$and": [{"section_number": 2}, {"kind": "bullet"}]}, col=_collection())
    best = hits[0]
    out = {"component": component, "plan_type": plan_type, "plan_allows": sorted(allowed),
           "distance": round(best["distance"], 3)}
    if best["distance"] > MATCH_MAX_DISTANCE:
        return {**out, "verdict": "no_match", "matched_item": None,
                "reason": "No listed coverage item is a close match; refer to a human adjuster.",
                "closest": items[best["id"]]["label"], "citations": [_cite(best)]}

    item = items[best["id"]]
    ok = item["category"] in allowed
    return {**out, "verdict": "covered" if ok else "not_in_plan_tier",
            "matched_item": item["label"], "category": item["category"],
            "reason": (f"'{item['label']}' is {'an' if item['category'] == 'appliances' else 'a'} {item['category'][:-1]} item and the {plan_type} plan "
                       + ("includes it." if ok else "does not include it.")),
            "citations": [_cite(best), _cite(decl)]}


# ---------------------------------------------------------------- tool 2
def check_waiting_period(breakdown_date, activation_date):
    """Did the breakdown occur after the initial waiting window (and within the coverage term)?

    Dates are datetime.date. The window length and term come from the contract text.
    """
    hit = search("initial waiting period pre-existing", n=1, where={"section_number": 3}, col=_collection())[0]
    m = re.search(r"(\d+)-day waiting period", hit["text"])
    if not m:
        return {"verdict": "unknown", "reason": "No waiting period found in contract.", "citations": [_cite(hit)]}
    days = int(m.group(1))
    decl = _declarations()
    term = re.search(r"(\d+)\s+Months?", decl["metadata"]["coverage_period"], re.I)

    waiting_ends = activation_date + timedelta(days=days)
    term_ends = activation_date + relativedelta(months=int(term.group(1))) if term else None
    elapsed = (breakdown_date - activation_date).days

    if breakdown_date < activation_date:
        verdict, why = "before_activation", "Breakdown predates the contract effective date (pre-existing)."
    elif breakdown_date < waiting_ends:
        verdict, why = "within_waiting_period", (
            f"Breakdown on day {elapsed}; the {days}-day waiting period runs until {waiting_ends}.")
    elif term_ends and breakdown_date >= term_ends:
        verdict, why = "after_term", f"Contract term ended {term_ends}."
    else:
        verdict, why = "eligible", f"Breakdown on day {elapsed}, after the {days}-day waiting period."
    return {"verdict": verdict, "eligible": verdict == "eligible", "reason": why,
            "waiting_days": days, "days_since_activation": elapsed,
            "waiting_period_ends": waiting_ends.isoformat(),
            "term_ends": term_ends.isoformat() if term_ends else None,
            "citations": [_cite(hit), _cite(decl)]}


# ---------------------------------------------------------------- tool 3
def get_limits(component=None):
    """Dollar limits that control cost: the service fee plus any per-item cap.

    With a component, returns the cap for its matched coverage item (cap=None means the
    contract states no dollar cap for it). Without one, lists every cap found.
    """
    decl = _declarations()
    fee = {"amount": float(re.search(r"[\d.]+", decl["metadata"]["service_fee"].replace(",", "")).group()),
           "description": decl["metadata"]["service_fee"]}
    items = _coverage_items()
    by_id = {it["id"]: it for it in items}

    if component is None:
        caps = [{"item": it["label"], **c, "citation": _cite(it)} for it in items if (c := _caps(it["text"]))]
        return {"service_fee": fee, "caps": caps, "citations": [_cite(decl)]}

    m = match_component(component)
    if m["verdict"] == "no_match":
        return {"component": component, "service_fee": fee, "cap": None, "matched_item": None,
                "reason": m["reason"], "citations": m["citations"] + [_cite(decl)]}
    item = by_id[next(i for i, it in by_id.items() if it["label"] == m["matched_item"])]
    cap = _caps(item["text"])
    return {"component": component, "matched_item": item["label"], "service_fee": fee, "cap": cap,
            "reason": (f"Maximum payout ${cap['amount']:,.0f} {cap['applies']}." if cap
                       else f"No dollar cap is stated for '{item['label']}'."),
            "citations": [_cite(item), _cite(decl)]}


# ---------------------------------------------------------------- tool 4
def _norm_ref(ref):
    return ref.strip().lstrip("#").strip().upper()


def get_repair_quote(claim_ref=None):
    """Grand total of the technician's repair quotation, read from the inspection report.

    verdict: found | not_found | ambiguous | no_reports. With no claim_ref the tool answers only
    when exactly one report is loaded. The stated grand total is cross-checked against the
    sum of the line items; `consistent` is False (with a warning) if they differ.
    """
    col = _report_collection()
    totals = col.get(where={"kind": "grand_total"})
    reports = {md["claim_ref"]: (i, md) for i, md in zip(totals["ids"], totals["metadatas"])}
    available = sorted(reports)
    if not reports:
        return {"verdict": "no_reports", "reason": "No inspection report with a repair quotation is loaded.",
                "citations": []}

    if claim_ref:
        match = next((r for r in reports if _norm_ref(r) == _norm_ref(claim_ref)), None)
        if not match:
            return {"verdict": "not_found", "claim_ref": claim_ref, "available_claim_refs": available,
                    "reason": f"No inspection report found for claim ref '{claim_ref}'.", "citations": []}
    elif len(reports) == 1:
        match = available[0]
    else:
        return {"verdict": "ambiguous", "available_claim_refs": available,
                "reason": "Several inspection reports are loaded; ask which claim ref to use.", "citations": []}

    gid, gmd = reports[match]
    r = col.get(where={"$and": [{"doc_id": gmd["doc_id"]}, {"kind": "line_item"}]})
    rows = sorted(zip(r["ids"], r["metadatas"]), key=lambda x: x[1]["chunk_index"])
    items_sum = round(sum(md["line_total"] for _, md in rows), 2)
    grand = gmd["grand_total"]
    consistent = abs(items_sum - grand) < 0.005
    out = {"verdict": "found", "claim_ref": match, "grand_total": grand, "currency": "USD",
           "line_items": [{"item": md["item_name"], "total": md["line_total"]} for _, md in rows],
           "items_sum": items_sum, "consistent": consistent,
           "reason": f"The repair quotation grand total is ${grand:,.2f}.",
           "citations": [_cite({"id": gid, "metadata": gmd})]}
    if not consistent:
        out["warning"] = (f"Stated grand total ${grand:,.2f} does not equal the sum of line items "
                          f"${items_sum:,.2f}; verify against the original report.")
    return out


def get_inspection_details(claim_ref=None):
    """Equipment, cause of failure and technician notes from the inspection report (for claim review).

    Resolves the report the same way as get_repair_quote. Returns verdict found | not_found | ambiguous | no_reports.
    """
    q = get_repair_quote(claim_ref)
    if q["verdict"] != "found":
        return {k: v for k, v in q.items() if k in ("verdict", "reason", "available_claim_refs", "claim_ref")} | {"citations": []}
    col = _report_collection()
    doc_id = col.get(ids=[q["citations"][0]["chunk"]])["metadatas"][0]["doc_id"]

    def chunks(kind):
        r = col.get(where={"$and": [{"doc_id": doc_id}, {"kind": kind}]})
        return [{"id": i, "text": d.split("] ", 1)[1], "metadata": m} for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])]

    eq, cause, notes = chunks("equipment"), chunks("cause_of_failure"), chunks("notes")
    md = eq[0]["metadata"] if eq else {}
    return {"verdict": "found", "claim_ref": q["claim_ref"],
            "equipment": {k: md[k] for k in ("system_type", "manufacturer", "model_number", "age_of_unit", "location") if k in md},
            "cause_of_failure": cause[0]["text"] if cause else None,
            "technician_notes": notes[0]["text"] if notes else None,
            "citations": [_cite(c) for c in eq + cause + notes]}


def get_exclusions():
    """The contract's exclusion bullets (Section 3), with citations."""
    r = _collection().get(where={"$and": [{"section_number": 3}, {"kind": "bullet"}]})
    rows = sorted(zip(r["ids"], r["documents"], r["metadatas"]), key=lambda x: x[2]["chunk_index"])
    return {"exclusions": [d.split("] ", 1)[1] for _, d, _ in rows],
            "citations": [_cite({"id": i, "metadata": m}) for i, _, m in rows]}


if __name__ == "__main__":
    import json
    print(json.dumps(match_component("HVAC compressor"), indent=2))
    print(json.dumps(check_waiting_period(date(2026, 2, 1), date(2026, 1, 15)), indent=2))
    print(json.dumps(get_limits("HVAC compressor"), indent=2))
