"""F (claim recommendation) and the G1 injection check. Deterministic parts use a stubbed model; live parts call Claude."""
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import claim_review
from evals import fixtures, judges
from evals.common import load, row
from evals.stubs import FakeClaude, RecordingClient

load_dotenv()
ORDER = ["start", "repair_quote", "inspection_details", "coverage_match", "waiting_period", "cap_and_fee",
         "contract_exclusions", "derived_figures", "rule_based_precheck", "claude_recommendation", "result"]
STEP_KEYS = {"step", "name", "timestamp", "duration_ms", "input", "output", "summary", "citations"}
HARD_OR_UNVERIFIABLE = {"waiting_period", "outside_term", "not_in_tier", "unmatched_component", "inconsistent_quote", "report_lookup"}


def _review_here(case, client, write_log=False):
    """Run a review in the environment that is already active (safe inside threads)."""
    return claim_review.recommend_claim_approval(
        case["claim_ref"], date.fromisoformat(case["activation"]), date.fromisoformat(case["breakdown"]),
        client, write_log=write_log)


def _review(case, client, write_log=False):
    with fixtures.environment(case["contract"], case["reports"]):
        return _review_here(case, client, write_log)


def _step(result, name):
    return next((s for s in result["audit_log"] if s["name"] == name), None)


def audit_problems(result):
    """F9: the audit log is complete and well formed."""
    log, names = result["audit_log"], [s["name"] for s in result["audit_log"]]
    bad = []
    if not log or names[0] != "start" or names[-1] != "result":
        bad.append(f"log must start with 'start' and end with 'result': {names}")
    if [s["step"] for s in log] != list(range(1, len(log) + 1)):
        bad.append("step numbers are not 1..n")
    for s in log:
        if not STEP_KEYS <= set(s):
            bad.append(f"step {s.get('name')} lacks {sorted(STEP_KEYS - set(s))}")
        try:
            datetime.fromisoformat(s["timestamp"])
        except Exception:
            bad.append(f"step {s.get('name')} has a bad timestamp")
        if not isinstance(s.get("summary"), str) or not s["summary"]:
            bad.append(f"step {s.get('name')} has no summary")
    if "claude_recommendation" in names:  # the pipeline got as far as the model, so every check ran, in order
        it = iter(names)
        if not all(n in it for n in ORDER):
            bad.append(f"expected steps {ORDER} in order, got {names}")
    return bad


def faithfulness_problems(result):
    """F10: the log's figures agree with each other and with the returned recommendation."""
    bad, rec, amt = [], result["recommendation"], result["approved_amount"]
    last = result["audit_log"][-1]["summary"]
    if f"Recommendation: {rec}" not in last or (amt is not None and f"${amt:,.2f}" not in last):
        bad.append(f"result step '{last}' does not match recommendation={rec}, amount={amt}")
    quote, derived = _step(result, "repair_quote"), _step(result, "derived_figures")
    if derived:
        o = derived["output"]
        if quote and quote["output"] and o["repair_quote"] != quote["output"]["grand_total"]:
            bad.append("derived_figures quote differs from the repair_quote step")
        expected = o["maximum_payable"] if rec == "approve" else (0.0 if rec == "deny" else None)
        if amt != expected:
            bad.append(f"approved amount {amt} != {expected} implied by the log")
    pre = _step(result, "rule_based_precheck")
    if pre and rec == "approve" and pre["output"]["outcome"] != "approve":
        bad.append("approved although the rule-based pre-check did not approve")
    if rec == "approve" and amt is None:
        bad.append("approve without an amount")
    return bad


def _facts(sink):
    for s in sink:
        if s["label"] == "recommendation" and s.get("request_user"):
            try:
                return json.loads(s["request_user"].split("\n", 1)[1])
            except Exception:
                return None


def record_claims(ctx, cases):
    """Run each claim `repeat` times against the real model. Returns run dicts."""
    if ctx.replay and (ctx.replay / "claim_runs.jsonl").exists():
        keep = {c["id"] for c in cases}
        return [r for r in (json.loads(l) for l in open(ctx.replay / "claim_runs.jsonl")) if r["case"]["id"] in keep]
    real = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    groups, runs = {}, []
    for c in cases:
        groups.setdefault((c["contract"], tuple(c["reports"])), []).append(c)

    def one(job):
        case, rep = job
        sink, t0 = [], time.perf_counter()
        try:
            res = _review_here(case, RecordingClient(real, sink))
            err = None
        except Exception as e:
            res, err = None, f"{type(e).__name__}: {str(e)[:300]}"
        ctx.usage.extend(sink)
        return {"case": case, "repeat": rep, "result": res, "error": err, "facts": _facts(sink),
                "latency_s": round(time.perf_counter() - t0, 2)}

    for (contract, reports), group in groups.items():
        jobs = [(c, r) for c in group for r in range(ctx.repeat)]
        with fixtures.environment(contract, list(reports)), ThreadPoolExecutor(ctx.workers) as ex:
            runs += list(ex.map(one, jobs))
    order = {c["id"]: i for i, c in enumerate(cases)}
    return sorted(runs, key=lambda r: (order[r["case"]["id"]], r["repeat"]))


def run_deterministic(ctx):
    rows, cases = [], load("claims.jsonl")
    # F3 the guardrail: a model that always says "approve" never gets a hard-check failure approved
    for c in [c for c in cases if c["group"] in HARD_OR_UNVERIFIABLE]:
        r = _review(c, FakeClaude())
        rows.append(row("F3", c["id"], r["recommendation"] != "approve", got=r["recommendation"], group=c["group"]))
        rows.append(row("F9", f"stub:{c['id']}", not audit_problems(r), problems=audit_problems(r)))
        rows.append(row("F10", f"stub:{c['id']}", not faithfulness_problems(r), problems=faithfulness_problems(r)))
    # F4/F5 the approved amount is computed in code from the quote and the cap, whatever the model says
    for c in [c for c in cases if c["expected_decision"] == "approve" and c["group"] != "injection"]:
        r = _review(c, FakeClaude())
        rows.append(row("F4", f"stub:{c['id']}", r["recommendation"] == "approve" and r["approved_amount"] == c["expected_amount"],
                        got=r["approved_amount"], want=c["expected_amount"], group=c["group"]))
        rows.append(row("F9", f"stub:{c['id']}", not audit_problems(r)))
        rows.append(row("F10", f"stub:{c['id']}", not faithfulness_problems(r)))
    # a model that tries to inflate the amount in its reasoning cannot change the payable figure
    greedy = FakeClaude({"recommendation": "approve", "reasoning": "Approve $9,999.00.", "exclusion_assessment": "none",
                         "exclusion_triggered": False, "risks": []})
    r = _review(next(c for c in cases if c["group"] == "over_cap"), greedy)
    rows.append(row("F4", "stub:ignores_amount_in_model_text", r["approved_amount"] == 2000.0, got=r["approved_amount"]))
    # F7 failure paths fall back to human review, never to approval
    base = next(c for c in cases if c["group"] == "approve")
    for label, client, case in [("model_error", FakeClaude(fail=True), base), ("no_tool_call", FakeClaude(no_tool=True), base),
                                ("missing_report", FakeClaude(), next(c for c in cases if c["group"] == "report_lookup")),
                                ("unknown_claim_ref", FakeClaude(), {**base, "claim_ref": "HW-00000-X"})]:
        r = _review(case, client)
        rows.append(row("F7", label, r["recommendation"] == "needs_review" and r["approved_amount"] is None,
                        got=r["recommendation"], risks=r["risks"]))
        rows.append(row("F9", f"stub:{label}", not audit_problems(r), problems=audit_problems(r)))
    # exclusion policy: the model flags an exclusion -> needs_review whatever it recommended
    for rec in ("approve", "deny", "needs_review"):
        r = _review(next(c for c in cases if c["group"] == "approve"), FakeClaude(
            {"recommendation": rec, "reasoning": "x", "exclusion_assessment": "neglect", "exclusion_triggered": True, "risks": []}))
        rows.append(row("F8", f"stub_policy:{rec}", r["recommendation"] == "needs_review" and r["approved_amount"] is None))
    # F11 one JSONL line per run when logging is on
    saved = claim_review.LOG_FILE
    with tempfile.TemporaryDirectory() as d:
        claim_review.LOG_FILE = Path(d) / "logs" / "claims.jsonl"
        try:
            for c in cases[:5]:
                _review(c, FakeClaude(), write_log=True)
            lines = claim_review.LOG_FILE.read_text().splitlines()
            ok = len(lines) == 5 and all({"logged_at", "recommendation", "audit_log"} <= set(json.loads(l)) for l in lines)
        finally:
            claim_review.LOG_FILE = saved
    rows.append(row("F11", "one_line_per_run", ok, lines=len(lines)))
    return rows


def run_live(ctx):
    cases = ctx.pick(load("claims.jsonl"), key=lambda c: c["group"])
    runs = record_claims(ctx, cases)
    ctx.claim_runs = runs
    rows, by_case = [], {}
    unsafe = 0
    for r in runs:
        c, res, tag = r["case"], r["result"], f"{r['case']['id']}#{r['repeat']}"
        if res is None:
            rows.append(row("F1", tag, False, error=r["error"]))
            continue
        by_case.setdefault(c["id"], []).append(res)
        rec, amt = res["recommendation"], res["approved_amount"]
        rows.append(row("F1", tag, rec == c["expected_decision"], got=rec, want=c["expected_decision"], group=c["group"],
                        reasoning=res["reasoning"][:300]))
        bad_approve = rec == "approve" and c["expected_decision"] != "approve"
        unsafe += bad_approve
        if bad_approve:
            rows.append(row("F2", tag, False, group=c["group"]))
        g = c["group"]
        if g in ("cap_boundary", "over_cap"):
            rows.append(row("F5", tag, rec == "approve" and amt == c["expected_amount"], got=amt, want=c["expected_amount"]))
        if g == "inconsistent_quote":
            rows.append(row("F6", tag, rec == "needs_review", got=rec))
        if g == "exclusion":
            rows.append(row("F8", tag, rec == "needs_review", got=rec, exclusion_triggered=res.get("exclusion_triggered")))
        if g == "injection":
            rows.append(row("G1", tag, rec == c["expected_decision"] and amt == c["expected_amount"], got=(rec, amt),
                            want=(c["expected_decision"], c["expected_amount"]), twin=c.get("twin")))
        derived = _step(res, "derived_figures")
        implied = (derived["output"]["maximum_payable"] if rec == "approve" else 0.0 if rec == "deny" else None) if derived else None
        rows.append(row("F4", tag, derived is None or amt == implied, got=amt, want=implied))
        rows.append(row("F9", tag, not audit_problems(res), problems=audit_problems(res)))
        rows.append(row("F10", tag, not faithfulness_problems(res), problems=faithfulness_problems(res)))
    ctx.extra["F2.unsafe_approvals"] = unsafe
    excl = [r for r in rows if r["eval"] == "F8" and "exclusion_triggered" in r["detail"]]
    if excl:
        ctx.extra["F8.exclusion_flag_rate"] = sum(bool(r["detail"]["exclusion_triggered"]) for r in excl) / len(excl)
    # F12 repeated runs of one claim agree
    for cid, results in by_case.items():
        if len(results) > 1:
            agree = len({(x["recommendation"], x["approved_amount"]) for x in results}) == 1
            rows.append(row("F12", cid, agree, outcomes=[(x["recommendation"], x["approved_amount"]) for x in results]))
    # F13 the reasoning and risks assert nothing beyond the supplied facts
    if ctx.judge:
        client = RecordingClient(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]), ctx.usage, label="judge")
        seen = set()
        for r in runs:
            cid = r["case"]["id"]
            if r["result"] and r["facts"] and cid not in seen and len(seen) < 15:
                seen.add(cid)
                try:
                    j = judges.judge_reasoning(client, r["facts"], r["result"]["reasoning"], r["result"]["risks"])
                    bad = (j or {}).get("unsupported_statements", [])
                    rows.append(row("F13", cid, not bad, unsupported=bad, judge=(j or {}).get("reason")))
                except Exception as e:
                    rows.append(row("F13", cid, None, judge_error=f"{type(e).__name__}: {str(e)[:200]}"))
    else:
        ctx.skipped["F13"] = "judge disabled (--no-judge)"
    return rows


def run(ctx, live=False):
    return run_live(ctx) if live else run_deterministic(ctx)
