"""C. Tool correctness (deterministic)."""
import calendar
import random
from datetime import date, timedelta

import tools
from evals import fixtures
from evals.common import load, row


def reference_waiting(act, br, wait=30, term=12):
    """Independent implementation of the waiting-period rule (no dateutil)."""
    if br < act:
        return "before_activation"
    if (br - act).days < wait:
        return "within_waiting_period"
    y, m0 = divmod(act.month - 1 + term, 12)
    end = date(act.year + y, m0 + 1, min(act.day, calendar.monthrange(act.year + y, m0 + 1)[1]))
    return "after_term" if br >= end else "eligible"


def _check_citations(out, contract_col, report_col):
    """Every citation must point at a chunk that exists, with the same page and section."""
    problems = []
    for c in out.get("citations", []):
        col = report_col if c.get("document") else contract_col
        got = col.get(ids=[c["chunk"]])
        if not got["ids"]:
            problems.append(f"{c['chunk']}: no such chunk")
            continue
        md = got["metadatas"][0]
        if md["page"] != c["page"] or c["section"] != f"{md['section_number']}. {md['section_title']}":
            problems.append(f"{c['chunk']}: page/section mismatch {c}")
    return problems


def run(ctx):
    rows = []
    comps = load("components.jsonl")

    with fixtures.environment("base", ["base"]):
        # C1 verdicts on the contract's own plan (Combo)
        false_covered = amb_false_covered = 0
        for c in ctx.pick(comps, key=lambda r: r["group"]):
            out = tools.match_component(c["component"])
            want = c["verdicts"]["Combo"]
            ok = out["verdict"] == want and (want != "covered" or out["matched_item"] == c["item"])
            if c["group"] == "ambiguous":
                amb_false_covered += out["verdict"] == "covered"
                rows.append(row("C1x", c["component"], out["verdict"] == "no_match", got=out["verdict"], distance=out.get("distance")))
                continue
            false_covered += want != "covered" and out["verdict"] == "covered"
            rows.append(row("C1", c["component"], ok, want=want, got=out["verdict"], item=out.get("matched_item"), group=c["group"]))
        ctx.extra["C1.false_covered"] = false_covered
        ctx.extra["C1x.false_covered"] = amb_false_covered

        # C2 plan-tier logic: explicit tier override, and contracts whose declared plan differs
        for c in ctx.pick([c for c in comps if c["group"] != "ambiguous"], key=lambda r: r["group"]):
            for tier in ("Systems Only", "Appliances Only"):
                out = tools.match_component(c["component"], tier)
                rows.append(row("C2", f"{tier}:{c['component']}", out["verdict"] == c["verdicts"][tier], got=out["verdict"], want=c["verdicts"][tier]))
    for contract, tier in (("systems_only", "Systems Only"), ("appliances_only", "Appliances Only")):
        with fixtures.environment(contract, ["base"]):
            for c in [c for c in comps if c["group"] in ("hvac", "plumbing", "appliance", "out_of_scope")][::3]:
                out = tools.match_component(c["component"])
                rows.append(row("C2", f"contract:{contract}:{c['component']}", out["verdict"] == c["verdicts"][tier], got=out["verdict"]))

    # C3 waiting window / C4 term edges and contract variants
    for c in load("dates.jsonl"):
        with fixtures.environment(c["contract"], ["base"]):
            out = tools.check_waiting_period(date.fromisoformat(c["breakdown"]), date.fromisoformat(c["activation"]))
        ev = "C3" if c["contract"] == "base" and c["expected"] in ("within_waiting_period", "eligible") else "C4"
        rows.append(row(ev, c["id"], out["verdict"] == c["expected"], got=out["verdict"], want=c["expected"], note=c["note"]))

    # C5 property test against an independent reference implementation
    rng = random.Random(ctx.seed)
    with fixtures.environment("base", ["base"]):
        for batch in range(5):
            bad = []
            for _ in range(100):
                act = date(2020, 1, 1) + timedelta(days=rng.randrange(0, 3000))
                br = act + timedelta(days=rng.randrange(-60, 500))
                got = tools.check_waiting_period(br, act)["verdict"]
                if got != reference_waiting(act, br):
                    bad.append((str(act), str(br), got, reference_waiting(act, br)))
            rows.append(row("C5", f"batch-{batch}", not bad, mismatches=bad[:5], n=100))

    # C6 limits
    for c in load("limits.jsonl"):
        with fixtures.environment(c["contract"], ["base"]):
            out = tools.get_limits(c["component"])
        e = c["expected"]
        if "listing" in e:
            ok = [(x["item"], x["amount"]) for x in out["caps"]] == [(x["item"], x["amount"]) for x in e["listing"]]
        else:
            got_cap = out["cap"]["amount"] if out.get("cap") else None
            ok = got_cap == e["cap"] and (not e.get("no_match") or out["matched_item"] is None)
        rows.append(row("C6", c["id"], ok and out["service_fee"]["amount"] == e["fee"], got=out.get("cap") or out.get("caps"), want=e))

    # C7 repair quote lookup
    for c in load("quotes.jsonl"):
        with fixtures.environment("base", c["reports"]):
            out = tools.get_repair_quote(c["claim_ref"])
        e = c["expected"]
        ok = out["verdict"] == e["verdict"] and ("total" not in e or out.get("grand_total") == e["total"]) \
            and ("consistent" not in e or (out["consistent"] == e["consistent"] and (e["consistent"] or "warning" in out)))
        rows.append(row("C7", c["id"], ok, got={k: out.get(k) for k in ("verdict", "grand_total", "consistent")}, want=e))

    # C8 inspection details
    for c in load("inspections.jsonl"):
        with fixtures.environment("base", [c["report"]]):
            out = tools.get_inspection_details()
        eq = out.get("equipment", {})
        cause = out.get("cause_of_failure")
        ok = (c["system_type"] in eq.get("system_type", "") and eq.get("manufacturer") == c["manufacturer"]
              and ((cause is None) if c["cause_contains"] is None else (cause is not None and c["cause_contains"] in cause.lower())))
        rows.append(row("C8", c["id"], ok, got=eq, cause=cause))

    # C9 citations exist, point at real chunks, and every answered verdict carries at least one
    with fixtures.environment("base", ["base"]):
        con, rep = fixtures.contract_collection("base"), fixtures.report_collection(("base",))
        calls = [("match_component:hvac", tools.match_component("HVAC compressor")), ("match_component:pool", tools.match_component("pool pump")),
                 ("check_waiting_period", tools.check_waiting_period(date(2026, 3, 1), date(2026, 1, 15))),
                 ("get_limits:none", tools.get_limits()), ("get_limits:hvac", tools.get_limits("HVAC compressor")),
                 ("get_repair_quote", tools.get_repair_quote()), ("get_inspection_details", tools.get_inspection_details()),
                 ("get_exclusions", tools.get_exclusions())]
        for name, out in calls:
            problems = _check_citations(out, con, rep)
            if not out.get("citations") and out.get("verdict") not in ("not_found", "no_reports", "ambiguous"):
                problems.append("no citations")
            rows.append(row("C9", name, not problems, problems=problems))
    return rows
