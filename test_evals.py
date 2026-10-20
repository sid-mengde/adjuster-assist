"""Tests for the eval framework itself (offline: a scripted fake Anthropic client, no network)."""
import json
from types import SimpleNamespace as NS

import mlflow
import pytest

import agent as agent_mod
from chunking_report import chunk_inspection_report
from evals import conversations as C, documents as D, fixtures, mlflow_utils
from evals.common import Context, aggregate, check_gates, load
from evals.config import DATASETS, REPORT_PDF
from evals.layers import d_agent, f_recommendation as F


def test_report_template_reproduces_the_real_pdf_exactly():
    real, syn = chunk_inspection_report(str(REPORT_PDF)), D.report_chunks("base")
    strip = lambda m: {k: v for k, v in m.items() if k != "source"}
    assert [(c.id, c.text, strip(c.metadata)) for c in real] == [(c.id, c.text, strip(c.metadata)) for c in syn]


def test_datasets_are_well_formed():
    claims, convs = load("claims.jsonl"), load("conversations.jsonl")
    assert len({c["id"] for c in claims}) == len(claims) == 44 and len({c["id"] for c in convs}) == len(convs)
    assert all(c["status"] == "draft" for c in claims), "claim labels stay draft until the claim approver signs off"
    assert {c["expected_decision"] for c in claims} == {"approve", "deny", "needs_review"}
    assert all(c["expected_amount"] is None for c in claims if c["expected_decision"] == "needs_review")
    assert all(c["expected_amount"] == 0.0 for c in claims if c["expected_decision"] == "deny")
    assert all(c["expected_decision"] == "needs_review" for c in claims if c["group"] == "exclusion")  # decided policy
    assert all(c["rationale"] for c in claims)
    known = {"tools", "tools_so_far", "forbid_tools", "no_tools", "args", "forbid_args", "asks_for", "reply_all", "reply_none",
             "tool_verdicts", "recommendation_not", "no_success_calls", "no_crash", "no_secret", "ask_or_call", "judge", "single_call"}
    for c in convs:
        assert c["env"]["contract"] in D.CONTRACTS and all(r in D.REPORTS for r in c["env"]["reports"])
        for t in c["turns"]:
            assert set(t["expect"]) <= known, (c["id"], set(t["expect"]) - known)
    assert json.loads((DATASETS / "claims_signoff.json").read_text())["status"] in ("draft", "approved")


def test_relative_dates_resolve():
    assert C.resolve("2026-03-01") == "2026-03-01"
    assert len(C.resolve("$TODAY-14")) == 10 and C.resolve("$TODAY-14") < C.resolve("$TODAY")
    assert C.resolve("$LAST:Tuesday") < C.resolve("$TODAY")


def turn(reply="", calls=(), err=None):
    return {"user": "q", "reply": reply, "error": err, "latency_s": 0.1, "calls": list(calls)}


def call(tool, inp=None, **result):
    return {"tool": tool, "input": inp or {}, "result": result}


def test_score_turn_checks():
    t = turn("It is covered.", [call("match_component", {"component": "Dishwasher"}, verdict="covered")])
    assert C.score_turn(t, {"tools": ["match_component"], "args": {"match_component": {"component": {"any": ["dishwasher"]}}}}, t["calls"]) == []
    assert C.score_turn(t, {"tools": ["get_limits"]}, t["calls"])
    assert C.score_turn(t, {"no_tools": True}, t["calls"])
    assert C.score_turn(t, {"forbid_tools": ["match_component"]}, t["calls"])
    assert C.score_turn(t, {"tool_verdicts": {"match_component": ["no_match"]}}, t["calls"])
    assert C.score_turn(t, {"reply_none": [r"covered"]}, t["calls"])
    ask = turn("What was the activation date and the breakdown date?")
    assert C.score_turn(ask, {"forbid_tools": ["check_waiting_period"], "asks_for": [[r"(?i)activat"], [r"(?i)breakdown"]]}, []) == []
    assert C.score_turn(turn("Done."), {"asks_for": [[r"(?i)activat"]]}, [])
    bad = turn("x", [call("check_waiting_period", {"breakdown_date": "2026-10-17", "activation_date": "2026-01-15"})])
    assert C.score_turn(bad, {"forbid_args": {"check_waiting_period": {"breakdown_date": "2026-10-17"}}}, bad["calls"])
    rec = turn("x", [call("recommend_claim_approval", {}, recommendation="approve")])
    assert C.score_turn(rec, {"recommendation_not": ["approve"]}, rec["calls"])
    assert C.score_turn(turn(err="AuthenticationError: bad key"), {"no_crash": True}, [])
    assert C.score_turn(turn("ok", [call("check_waiting_period", {}, error="bad date")]), {"no_success_calls": ["check_waiting_period"]}, []) == []
    assert C.score_turn(turn("the key is sk-ant-123"), {"no_secret": True}, [])
    # ambiguous date: asking is fine, and so is calling with the stated reading
    fmt = {"ask_or_call": "check_waiting_period", "args": {"check_waiting_period": {"activation_date": "2026-03-04"}}}
    assert C.score_turn(turn("Did you mean March 4 or April 3?"), fmt, []) == []
    assert C.score_turn(turn("ok", [call("check_waiting_period", {"activation_date": "2026-03-04"})]), fmt, []) == []
    assert C.score_turn(turn("ok", [call("check_waiting_period", {"activation_date": "2026-04-03"})]), fmt, [])
    # a judge flag only counts when the judge ran
    assert C.score_turn(turn("x"), {"judge": {"recommends_approval": False}}, []) == []
    j = turn("x"); j["judge"] = {"recommends_approval": True}
    assert C.score_turn(j, {"judge": {"recommends_approval": False}}, [])


def test_grounding_flags_invented_numbers_fake_citations_and_placeholders():
    res = {"verdict": "covered", "citations": [{"chunk": "sec2-4", "page": 1, "section": "2. WHAT IS COVERED"}], "cap": {"amount": 2000.0}}
    def rows_for(reply):
        t = {"id": "x", "eval": "D1", "turns": [{"user": "Is it covered?", "reply": reply, "error": None, "latency_s": 0,
                                                    "calls": [call("get_limits", {}, **res)]}]}
        out = d_agent.grounding_rows(Context(), [t], {})
        return {r["eval"]: r for r in out}
    ok = rows_for("The cap is $2,000 (Section 2, page 1).")
    assert ok["E1"]["passed"] and ok["E3"]["passed"]
    assert not rows_for("The cap is $2,500.")["E1"]["passed"]
    assert not rows_for("Waiting is 45 days.")["E1"]["passed"]
    assert not rows_for("See (Section 3, page 2).")["E3"]["passed"]
    assert not rows_for("See (Section N, page P).")["E3"]["passed"]


def test_audit_and_faithfulness_detectors_catch_a_broken_log():
    from datetime import date
    import claim_review
    from evals.stubs import FakeClaude
    case = next(c for c in load("claims.jsonl") if c["group"] == "approve")
    good = F._review(case, FakeClaude())
    assert not F.audit_problems(good) and not F.faithfulness_problems(good)
    broken = json.loads(json.dumps(good, default=str))
    broken["audit_log"].pop(4)
    assert F.audit_problems(broken)
    wrong = json.loads(json.dumps(good, default=str))
    wrong["approved_amount"] = 9999.0
    assert F.faithfulness_problems(wrong)


def test_gates_and_regression_comparison():
    metrics = {"C1.false_covered": 1, "A1.pass_rate": 1.0, "B1.recall_at_3": 0.97}
    res = {n: ok for n, _, _, _, ok in check_gates(metrics)}
    assert res == {"A1.pass_rate": True, "B1.recall_at_3": True, "C1.false_covered": False}
    base = {"B1.recall_at_3": 0.99, "C1.false_covered": 0, "A1.pass_rate": 1.0, "F2.unsafe_approvals": 0}
    now = {"B1.recall_at_3": 0.93, "C1.false_covered": 1, "A1.pass_rate": 0.99, "F2.unsafe_approvals": 0}
    assert {k for k, _, _ in mlflow_utils.compare(now, base)} == {"B1.recall_at_3", "C1.false_covered"}  # 0.01 drop is tolerated


class FakeAnthropic:
    """Scripted chat model: dishwasher questions trigger match_component, everything else asks for dates."""
    def __init__(self):
        self.messages = self

    def create(self, **kw):
        last = kw["messages"][-1]
        usage = NS(input_tokens=10, output_tokens=5)
        if last["role"] == "user" and isinstance(last["content"], str):
            if "dishwasher" in last["content"].lower():
                return NS(stop_reason="tool_use", usage=usage, content=[NS(
                    type="tool_use", id="t1", name="match_component", input={"component": "dishwasher"})])
            return NS(stop_reason="end_turn", usage=usage, content=[NS(type="text", text="What are the activation and breakdown dates?")])
        return NS(stop_reason="end_turn", usage=usage, content=[NS(type="text", text="Yes, it is covered (Section 2, page 1). The fee is $85.")])


def test_conversation_pipeline_and_mlflow_logging_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_mod, "Anthropic", lambda api_key: FakeAnthropic())
    ctx = Context(workers=2, judge=False)
    convs = [
        {"id": "T-1", "eval": "D1", "tags": [], "env": {"contract": "base", "reports": ["base"]},
         "turns": [{"user": "Is a dishwasher covered?", "expect": {"tools": ["match_component"]}}]},
        {"id": "T-2", "eval": "D3", "tags": [], "env": {"contract": "base", "reports": ["base"]},
         "turns": [{"user": "Was the waiting period satisfied?", "expect": {"forbid_tools": ["check_waiting_period"],
                                                                           "asks_for": [[r"(?i)activat"], [r"(?i)breakdown"]]}}]},
        {"id": "T-3", "eval": "D1", "tags": [], "env": {"contract": "base", "reports": ["base"]},
         "turns": [{"user": "Is a dishwasher covered?", "expect": {"tools": ["get_limits"]}}]},  # fails on purpose
    ]
    trajs = C.record(ctx, convs)
    assert [t["id"] for t in trajs] == ["T-1", "T-2", "T-3"] and trajs[0]["turns"][0]["calls"][0]["tool"] == "match_component"
    assert len(ctx.usage) >= 4 and all(u["label"] == "chat" for u in ctx.usage)
    scored = {c["id"]: C.score(t, c)[0] for t, c in zip(trajs, convs)}
    assert scored == {"T-1": True, "T-2": True, "T-3": False}
    by_id = {c["id"]: c for c in convs}
    e = {(r["eval"], r["case"]): r["passed"] for r in d_agent.grounding_rows(ctx, trajs, by_id)}
    assert e[("E3", "T-1#0")] is True and e[("E1", "T-1#0")] is False  # "$85" is not in the match_component result

    # MLflow: log into a throw-away store, then compare to a baseline
    monkeypatch.setattr(mlflow_utils, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(mlflow_utils, "RESULTS", tmp_path)
    monkeypatch.setattr(mlflow_utils, "EXPERIMENT", "evals-selftest")
    from evals.common import row
    rows_ = [row("A1", "x", True), row("A1", "y", True), row("C1", "z", False, why="boom")]
    m = aggregate(rows_)
    (tmp_path / "run").mkdir()
    kw = dict(suite="det", layers=["A", "C"], rows=rows_, metrics=m, skipped={"H8": "n/a"}, run_dir=tmp_path / "run",
              params={"sample": None}, tags={}, report_md="# r")
    rid, gates, regs = mlflow_utils.log_suite(set_baseline=True, compare_baseline=False, **kw)
    run = mlflow.get_run(rid)
    assert run.data.metrics["A1.pass_rate"] == 1.0 and run.data.tags["eval_baseline"] == "true"
    worse = dict(m, **{"A1.pass_rate": 0.5})
    _, _, regs = mlflow_utils.log_suite(set_baseline=False, compare_baseline=True, **{**kw, "metrics": worse})
    assert [k for k, _, _ in regs] == ["A1.pass_rate"]
    kids = mlflow.search_runs(search_all_experiments=True, filter_string=f"tags.mlflow.parentRunId = '{rid}'")
    assert sorted(kids["tags.layer"]) == ["A", "C"]
