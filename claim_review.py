"""Claim-approval recommendation pipeline.

The checks run in a fixed order in code (nothing is left to the model's discretion), every step is
written to an audit log, and a single Claude call turns the gathered facts into a recommendation.
A rule-based guardrail prevents the model from approving a claim that fails a hard check.
"""
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import tools

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
LOG_FILE = Path("logs/claims.jsonl")
DISCLAIMER = "Advisory recommendation only; a human adjuster makes the final decision."
ASSUMPTIONS = ["No prior payouts have been made against the contract-term cap (per adjuster instruction)."]

RECOMMEND_TOOL = {
    "name": "submit_recommendation",
    "description": "Submit the claim approval recommendation.",
    "input_schema": {"type": "object", "properties": {
        "recommendation": {"type": "string", "enum": ["approve", "deny", "needs_review"]},
        "reasoning": {"type": "string", "description": "2-5 sentences citing the supplied facts"},
        "exclusion_assessment": {"type": "string",
                                 "description": "Whether the technician's findings trigger any contract exclusion"},
        "risks": {"type": "array", "items": {"type": "string"}, "description": "Open concerns; empty if none"},
    }, "required": ["recommendation", "reasoning", "exclusion_assessment", "risks"]},
}

SYSTEM = """You advise a home-warranty claims adjuster. Decide whether to approve the claim using ONLY the \
facts provided: contract coverage tier, waiting period, cap, the technician's findings, the repair quote and \
the contract's exclusions. Do not invent facts or amounts. Judge the technician's cause of failure against \
each contract exclusion. Choose needs_review when facts are missing, contradictory, or a judgment call is \
needed that the facts do not settle. The `rule_based_outcome` is a deterministic pre-check; if it is not \
'approve', you must not recommend approve. Respond by calling submit_recommendation."""


class AuditLog:
    def __init__(self):
        self.steps = []

    def run(self, name, fn, summarize, **inputs):
        start, t0 = datetime.now(timezone.utc), time.perf_counter()
        try:
            out, err = fn(), None
        except Exception as e:
            out, err = None, f"{type(e).__name__}: {e}"
        step = {"step": len(self.steps) + 1, "name": name, "timestamp": start.isoformat(timespec="seconds"),
                "duration_ms": round((time.perf_counter() - t0) * 1000), "input": inputs,
                "output": out, "summary": f"ERROR: {err}" if err else summarize(out),
                "citations": (out or {}).get("citations", []) if isinstance(out, dict) else []}
        if err:
            step["error"] = err
        self.steps.append(step)
        return out

    def note(self, name, summary, **data):
        self.steps.append({"step": len(self.steps) + 1, "name": name,
                           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "duration_ms": 0, "input": {}, "output": data or None, "summary": summary, "citations": []})


def _money(x):
    return f"${x:,.2f}"


def _rule_based(coverage, waiting, quote):
    """Deterministic pre-check: returns (outcome, [reasons])."""
    deny, review = [], []
    if coverage["verdict"] == "not_in_plan_tier":
        deny.append(f"{coverage['matched_item']} is not included in the {coverage['plan_type']} plan.")
    elif coverage["verdict"] != "covered":
        review.append("The component could not be matched to a covered item.")
    if waiting["verdict"] in ("within_waiting_period", "before_activation", "after_term"):
        deny.append(waiting["reason"])
    elif not waiting.get("eligible"):
        review.append("The waiting period could not be verified.")
    if not quote.get("consistent", False):
        review.append(quote.get("warning", "The repair quote could not be verified."))
    return ("deny" if deny else "review" if review else "approve"), deny + review


def _ask_claude(client, facts):
    resp = client.messages.create(
        model=MODEL, max_tokens=1024, system=SYSTEM, tools=[RECOMMEND_TOOL],
        tool_choice={"type": "auto"},  # this model rejects forced tool use; the system prompt requires the call
        messages=[{"role": "user", "content": "Claim facts (JSON):\n" + json.dumps(facts, indent=2, default=str)}])
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if block is None:
        raise ValueError("model did not call submit_recommendation")
    return dict(block.input)


def recommend_claim_approval(claim_ref, activation_date, breakdown_date, client, write_log=True):
    """Run the full review. Dates are datetime.date. Returns the recommendation plus the audit log."""
    log = AuditLog()
    log.note("start", f"Claim review for {claim_ref or 'the loaded report'}: activation {activation_date}, "
                      f"breakdown {breakdown_date}")

    quote = log.run("repair_quote", lambda: tools.get_repair_quote(claim_ref),
                    lambda o: o.get("reason", ""), claim_ref=claim_ref)
    if not quote or quote.get("verdict") != "found":
        return _finish(log, claim_ref, write_log, "needs_review", None,
                       (quote or {}).get("reason", "Repair quotation unavailable."), [], None)
    details = log.run("inspection_details", lambda: tools.get_inspection_details(quote["claim_ref"]),
                      lambda o: f"{o['equipment'].get('system_type')} ({o['equipment'].get('manufacturer')}, "
                                f"{o['equipment'].get('age_of_unit')}); cause of failure retrieved",
                      claim_ref=quote["claim_ref"])
    if not details or details.get("verdict") != "found":
        return _finish(log, claim_ref, write_log, "needs_review", None, "Inspection details unavailable.", [], None)
    component = details["equipment"].get("system_type", "")

    coverage = log.run("coverage_match", lambda: tools.match_component(component),
                       lambda o: o["reason"] if o["verdict"] != "no_match" else o["reason"], component=component)
    waiting = log.run("waiting_period",
                      lambda: tools.check_waiting_period(breakdown_date, activation_date),
                      lambda o: o["reason"], breakdown_date=str(breakdown_date), activation_date=str(activation_date))
    limits = log.run("cap_and_fee", lambda: tools.get_limits(component),
                     lambda o: f"{o['reason']} Service fee {_money(o['service_fee']['amount'])}.", component=component)
    exclusions = log.run("contract_exclusions", tools.get_exclusions,
                         lambda o: f"{len(o['exclusions'])} contract exclusions retrieved")
    if not all([coverage, waiting, limits, exclusions]):
        return _finish(log, claim_ref, write_log, "needs_review", None,
                       "A verification step failed; see the log.", [], None)

    cap = limits.get("cap")
    grand = quote["grand_total"]
    derived = {"repair_quote": grand, "cap": cap["amount"] if cap else None,
               "quote_within_cap": cap is None or grand <= cap["amount"],
               "amount_over_cap": round(max(0.0, grand - cap["amount"]), 2) if cap else 0.0,
               "maximum_payable": min(grand, cap["amount"]) if cap else grand,
               "service_fee": limits["service_fee"]["amount"]}
    log.note("derived_figures", f"Quote {_money(grand)} vs cap "
             + (_money(cap["amount"]) if cap else "none") + f" -> maximum payable {_money(derived['maximum_payable'])}",
             **derived)

    rule, rule_reasons = _rule_based(coverage, waiting, quote)
    log.note("rule_based_precheck", f"Deterministic pre-check: {rule}" + (f" ({'; '.join(rule_reasons)})" if rule_reasons else ""),
             outcome=rule, reasons=rule_reasons)

    facts = {"claim_ref": quote["claim_ref"], "equipment": details["equipment"],
             "technician_cause_of_failure": details["cause_of_failure"],
             "technician_notes": details["technician_notes"],
             "coverage": {k: coverage.get(k) for k in ("verdict", "matched_item", "plan_type", "reason")},
             "waiting_period": {k: waiting.get(k) for k in ("verdict", "days_since_activation", "waiting_days", "reason")},
             "limits_and_quote": derived, "quote_line_items": quote["line_items"],
             "quote_consistent": quote["consistent"], "contract_exclusions": exclusions["exclusions"],
             "assumptions": ASSUMPTIONS, "rule_based_outcome": rule, "rule_based_reasons": rule_reasons}

    rec = log.run("claude_recommendation", lambda: _ask_claude(client, facts),
                  lambda o: f"Claude recommends: {o['recommendation']}", model=MODEL)
    if rec is None:
        return _finish(log, claim_ref, write_log, "needs_review", None,
                       "The recommendation model call failed; defaulting to human review.",
                       ["Model call failed."], facts)

    final, reasoning, risks = rec["recommendation"], rec["reasoning"], list(rec.get("risks", []))
    if final == "approve" and rule != "approve":  # guardrail
        final = "deny" if rule == "deny" else "needs_review"
        risks.append(f"Guardrail: model recommended approve but the rule-based pre-check is '{rule}'.")
        log.note("guardrail_override", f"Model's approve overridden to {final}: " + "; ".join(rule_reasons))
    amount = derived["maximum_payable"] if final == "approve" else (0.0 if final == "deny" else None)
    return _finish(log, claim_ref, write_log, final, amount, reasoning, risks, facts,
                   exclusion_assessment=rec.get("exclusion_assessment"))


def _finish(log, claim_ref, write_log, recommendation, amount, reasoning, risks, facts, **extra):
    log.note("result", f"Recommendation: {recommendation}"
             + (f", approved amount {_money(amount)}" if amount is not None else ""))
    result = {"recommendation": recommendation, "approved_amount": amount, "reasoning": reasoning,
              "risks": risks, "assumptions": ASSUMPTIONS, "disclaimer": DISCLAIMER,
              "claim_ref": claim_ref, "audit_log": log.steps, **extra}
    if write_log:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(json.dumps({"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "facts": facts, **result}, default=str) + "\n")
    return result
