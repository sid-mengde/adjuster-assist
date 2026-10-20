"""H. Service behaviour (deterministic, stub agent) and H5/H6 latency and cost (derived from the live LLM calls)."""
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

import api
from evals.common import percentile, row
from evals.config import PRICE_IN, PRICE_OUT


class StubAgent:
    def __init__(self, exc=None):
        self.exc, self.seen = exc, []

    def chat(self, messages):
        if self.exc:
            raise self.exc
        self.seen.append(len(messages))
        text = messages[-1]["content"]
        messages.append({"role": "assistant", "content": "echo:" + text})
        calls = []
        if text.startswith("recommend"):
            calls = [{"tool": "recommend_claim_approval", "input": {}, "result": {"recommendation": "approve", "audit_log": [{"step": 1}]}}]
        return "echo:" + text, calls


class AuthError(Exception):
    message = "invalid x-api-key"


def _client(agent):
    api.state["agent"], api.state["sessions"] = agent, {}
    return TestClient(api.app)


def run(ctx):
    rows = []
    saved = dict(api.state)
    try:
        agent = StubAgent()
        c = _client(agent)
        # H1 request and response contract
        r = c.post("/chat", json={"message": "hi"})
        j = r.json()
        rows.append(row("H1", "valid_request", r.status_code == 200 and {"session_id", "reply", "tool_calls", "audit_log"} <= set(j) and j["reply"] == "echo:hi"))
        rows.append(row("H1", "health", c.get("/health").json() == {"status": "ok"}))
        for label, body in [("empty_message", {"message": ""}), ("missing_message", {}), ("too_long", {"message": "x" * 4001}),
                            ("wrong_type", {"message": 123})]:
            rows.append(row("H1", label, c.post("/chat", json=body).status_code == 422))
        r = c.post("/chat", json={"message": "recommend please"}).json()
        rows.append(row("H1", "audit_log_moved_out_of_tool_calls", r["audit_log"] == [{"step": 1}] and "audit_log" not in r["tool_calls"][0]["result"]))
        status = c.post("/chat", json={"message": "   "}).status_code  # a blank message should not reach the LLM
        rows.append(row("H1", "whitespace_message_rejected", status == 422, status=status))

        # H2 sessions are isolated and stateful
        agent = StubAgent()
        c = _client(agent)
        a1 = c.post("/chat", json={"message": "a1"}).json()["session_id"]
        b1 = c.post("/chat", json={"message": "b1"}).json()["session_id"]
        c.post("/chat", json={"message": "a2", "session_id": a1})
        c.post("/chat", json={"message": "a3", "session_id": a1})
        c.post("/chat", json={"message": "b2", "session_id": b1})
        rows.append(row("H2", "history_lengths", a1 != b1 and agent.seen == [1, 1, 3, 5, 3], seen=agent.seen))

        # H3 session delete and unknown ids
        c.delete(f"/sessions/{a1}")
        c.post("/chat", json={"message": "after-delete", "session_id": a1})
        rows.append(row("H3", "restart_after_delete", agent.seen[-1] == 1, seen=agent.seen))
        rows.append(row("H3", "delete_unknown", c.delete("/sessions/does-not-exist").status_code == 200))
        rows.append(row("H3", "unknown_id_starts_fresh", c.post("/chat", json={"message": "x", "session_id": "never-seen"}).status_code == 200))

        # H4 upstream failures surface as 502 with a reason and leave the history usable
        for label, exc, needle in [("auth_error", AuthError(), "invalid x-api-key"), ("timeout", TimeoutError("slow"), "TimeoutError"),
                                   ("runtime_error", RuntimeError("boom"), "RuntimeError")]:
            agent = StubAgent(exc)
            c = _client(agent)
            r = c.post("/chat", json={"message": "hi", "session_id": "s"})
            ok = r.status_code == 502 and needle in r.json()["detail"]
            agent.exc = None
            c.post("/chat", json={"message": "retry", "session_id": "s"})
            rows.append(row("H4", label, ok and agent.seen == [1], status=r.status_code, detail=r.json().get("detail")))

        # H7 ten parallel requests do not cross-talk
        c = _client(StubAgent())
        with ThreadPoolExecutor(10) as ex:
            res = list(ex.map(lambda i: c.post("/chat", json={"message": f"msg-{i}"}).json(), range(10)))
        rows.append(row("H7", "parallel_requests", [x["reply"] for x in res] == [f"echo:msg-{i}" for i in range(10)]
                        and len({x["session_id"] for x in res}) == 10))
    finally:
        api.state.clear()
        api.state.update(saved)
    return rows


def usage_metrics(ctx):
    """H5/H6: latency percentiles and token/cost totals over every recorded LLM call (report-only)."""
    out = {}
    for label in sorted({u["label"] for u in ctx.usage}):
        calls = [u for u in ctx.usage if u["label"] == label]
        out[f"H5.{label}_call_p50_s"] = percentile([u["latency_s"] for u in calls], 0.5)
        out[f"H5.{label}_call_p95_s"] = percentile([u["latency_s"] for u in calls], 0.95)
        out[f"H6.{label}_calls"] = len(calls)
        out[f"H6.{label}_input_tokens"] = sum(u["input_tokens"] for u in calls)
        out[f"H6.{label}_output_tokens"] = sum(u["output_tokens"] for u in calls)
        if PRICE_IN or PRICE_OUT:
            out[f"H6.{label}_cost_usd"] = sum(u["input_tokens"] for u in calls) / 1e6 * PRICE_IN \
                + sum(u["output_tokens"] for u in calls) / 1e6 * PRICE_OUT
    turns = [t["latency_s"] for tr in ctx.trajectories for t in tr["turns"]]
    if turns:
        out["H5.chat_turn_p50_s"], out["H5.chat_turn_p95_s"] = percentile(turns, 0.5), percentile(turns, 0.95)
    recs = [r["latency_s"] for r in ctx.claim_runs if r.get("result")]
    if recs:
        out["H5.recommendation_p50_s"], out["H5.recommendation_p95_s"] = percentile(recs, 0.5), percentile(recs, 0.95)
        out["H6.input_tokens_per_recommendation"] = None
    return {k: v for k, v in out.items() if v is not None}
