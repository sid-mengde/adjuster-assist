"""Claim review pipeline tests with a stubbed Claude (no network)."""
import json
from datetime import date
from types import SimpleNamespace as NS

import pytest

import claim_review
import tools

ACT, OK_BREAKDOWN, EARLY_BREAKDOWN = date(2026, 1, 15), date(2026, 3, 1), date(2026, 1, 25)


class FakeClaude:
    def __init__(self, rec=None, fail=False):
        self.rec = rec or {"recommendation": "approve", "reasoning": "All checks pass.",
                           "exclusion_assessment": "Wear and tear; no exclusion applies.", "risks": []}
        self.fail, self.calls = fail, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("model down")
        return NS(content=[NS(type="tool_use", input=self.rec)])


def review(client, breakdown=OK_BREAKDOWN):
    return claim_review.recommend_claim_approval("HW-88301-A", ACT, breakdown, client, write_log=False)


def names(r):
    return [s["name"] for s in r["audit_log"]]


def test_approve_happy_path():
    c = FakeClaude()
    r = review(c)
    assert r["recommendation"] == "approve" and r["approved_amount"] == 1800.0
    assert names(r) == ["start", "repair_quote", "inspection_details", "coverage_match", "waiting_period",
                        "cap_and_fee", "contract_exclusions", "derived_figures", "rule_based_precheck",
                        "claude_recommendation", "result"]
    assert [s["step"] for s in r["audit_log"]] == list(range(1, 12))
    assert all(s["timestamp"] for s in r["audit_log"])
    assert r["disclaimer"] and r["assumptions"]


def test_claude_receives_all_facts_and_the_recommend_tool():
    c = FakeClaude()
    review(c)
    call = c.calls[0]
    assert call["tool_choice"] == {"type": "auto"} and call["tools"][0]["name"] == "submit_recommendation"
    facts = json.loads(call["messages"][0]["content"].split("\n", 1)[1])
    assert facts["limits_and_quote"] == {"repair_quote": 1800.0, "cap": 2000.0, "quote_within_cap": True,
                                         "amount_over_cap": 0.0, "maximum_payable": 1800.0, "service_fee": 85.0}
    assert facts["coverage"]["verdict"] == "covered" and facts["waiting_period"]["verdict"] == "eligible"
    assert "wear and tear" in facts["technician_cause_of_failure"]
    assert len(facts["contract_exclusions"]) == 3 and facts["rule_based_outcome"] == "approve"


def test_steps_carry_citations():
    r = review(FakeClaude())
    by = {s["name"]: s for s in r["audit_log"]}
    assert by["repair_quote"]["citations"][0]["document"] == "Inspection report HW-88301-A"
    assert by["coverage_match"]["citations"] and by["waiting_period"]["citations"]


def test_guardrail_overrides_model_approve_inside_waiting_period():
    r = review(FakeClaude(), breakdown=EARLY_BREAKDOWN)  # day 10 of a 30-day window
    assert r["recommendation"] == "deny" and r["approved_amount"] == 0.0
    assert any("Guardrail" in x for x in r["risks"]) and "guardrail_override" in names(r)


def test_model_deny_without_failed_hard_check_goes_to_human_review():
    rec = {"recommendation": "deny", "reasoning": "x", "exclusion_assessment": "y", "risks": []}
    r = review(FakeClaude(rec))
    assert r["recommendation"] == "needs_review" and r["approved_amount"] is None
    assert "exclusion_policy_override" in names(r) and any("Policy" in x for x in r["risks"])


@pytest.mark.parametrize("model_says", ["approve", "deny", "needs_review"])
def test_triggered_exclusion_always_routes_to_human_review(model_says):
    rec = {"recommendation": model_says, "reasoning": "x", "exclusion_assessment": "neglect",
           "exclusion_triggered": True, "risks": []}
    r = review(FakeClaude(rec))
    assert r["recommendation"] == "needs_review" and r["approved_amount"] is None and r["exclusion_triggered"] is True


def test_triggered_exclusion_does_not_soften_a_hard_deny():
    rec = {"recommendation": "deny", "reasoning": "x", "exclusion_assessment": "y", "exclusion_triggered": True, "risks": []}
    r = review(FakeClaude(rec), breakdown=EARLY_BREAKDOWN)  # inside the waiting period: a hard-check failure
    assert r["recommendation"] == "deny" and r["approved_amount"] == 0.0


def test_model_failure_defaults_to_human_review():
    r = review(FakeClaude(fail=True))
    assert r["recommendation"] == "needs_review" and r["approved_amount"] is None
    assert any("ERROR" in s["summary"] for s in r["audit_log"])


def test_unknown_claim_ref_stops_before_model():
    c = FakeClaude()
    r = claim_review.recommend_claim_approval("HW-00000-X", ACT, OK_BREAKDOWN, c, write_log=False)
    assert r["recommendation"] == "needs_review" and c.calls == [] and names(r)[-1] == "result"


def test_over_cap_quote_is_capped(monkeypatch):
    real = tools.get_repair_quote
    monkeypatch.setattr(tools, "get_repair_quote", lambda ref=None: {**real(ref), "grand_total": 2500.0})
    r = review(FakeClaude())
    assert r["recommendation"] == "approve" and r["approved_amount"] == 2000.0  # capped, not 2500


def test_inconsistent_quote_blocks_approval(monkeypatch):
    real = tools.get_repair_quote
    monkeypatch.setattr(tools, "get_repair_quote",
                        lambda ref=None: {**real(ref), "consistent": False, "warning": "totals differ"})
    r = review(FakeClaude())
    assert r["recommendation"] == "needs_review" and r["approved_amount"] is None


def test_log_written_to_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(claim_review, "LOG_FILE", tmp_path / "logs" / "claims.jsonl")
    claim_review.recommend_claim_approval("HW-88301-A", ACT, OK_BREAKDOWN, FakeClaude(), write_log=True)
    line = json.loads((tmp_path / "logs" / "claims.jsonl").read_text().splitlines()[0])
    assert line["recommendation"] == "approve" and len(line["audit_log"]) == 11 and line["facts"]["claim_ref"]


# --- through the agent and API
import api, agent as agent_mod
from fastapi.testclient import TestClient


def test_api_returns_recommendation_with_audit_log(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(claim_review, "LOG_FILE", tmp_path / "claims.jsonl")
    rec = FakeClaude().rec
    script = [
        NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name="recommend_claim_approval",
                                               input={"activation_date": "2026-01-15", "breakdown_date": "2026-03-01"})]),
        NS(content=[NS(type="tool_use", input=rec)]),  # the recommendation call inside the pipeline
        NS(stop_reason="end_turn", content=[NS(type="text", text="Recommend approval for $1,800.00.")]),
    ]
    class Scripted:
        messages = None
        def __init__(self): self.messages = self
        def create(self, **kw): return script.pop(0)
    with TestClient(api.app) as c:
        api.state["agent"].client = Scripted()
        r = c.post("/chat", json={"message": "Should this claim be approved?"}).json()
    assert "approval" in r["reply"] and len(r["audit_log"]) == 11
    assert r["tool_calls"][0]["result"]["recommendation"] == "approve"
    assert "audit_log" not in r["tool_calls"][0]["result"]  # not duplicated


def test_chat_model_sees_summaries_not_full_log():
    out = {"recommendation": "approve", "audit_log": [{"step": 1, "summary": "did a thing", "output": {"big": "x" * 999}}]}
    m = agent_mod._for_model(out)
    assert m["audit_log_summary"] == ["1. did a thing"] and "audit_log" not in m


def test_model_that_skips_the_tool_defaults_to_review():
    c = FakeClaude()
    c.create = lambda **kw: NS(content=[NS(type="text", text="I approve")])
    r = review(c)
    assert r["recommendation"] == "needs_review" and any("did not call" in s["summary"] for s in r["audit_log"])
