"""API tests with a stubbed Claude client (no network, no API key needed)."""
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

import agent as agent_mod
import api


def block_text(t): return NS(type="text", text=t)
def block_tool(name, **inp): return NS(type="tool_use", id="tu1", name=name, input=inp)


class FakeClient:
    """First call asks for a tool, second call answers."""
    def __init__(self, script): self.script, self.messages = list(script), self
    def create(self, **kw): return self.script.pop(0)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_uses_tool_and_keeps_session(client):
    api.state["agent"].client = FakeClient([
        NS(stop_reason="tool_use", content=[block_tool("get_limits", component="HVAC compressor")]),
        NS(stop_reason="end_turn", content=[block_text("HVAC is capped at $2,000 (Section 2, page 1).")]),
    ])
    r = client.post("/chat", json={"message": "What is the HVAC cap?"}).json()
    assert "$2,000" in r["reply"]
    assert r["tool_calls"][0]["tool"] == "get_limits"
    assert r["tool_calls"][0]["result"]["cap"]["amount"] == 2000.0
    assert len(api.state["sessions"][r["session_id"]]) == 4  # user, assistant(tool), tool_result, assistant


def test_bad_tool_input_returns_error_to_model():
    out = agent_mod._run_tool("check_waiting_period", {"breakdown_date": "nope", "activation_date": "2026-01-01"})
    assert "error" in out


def test_llm_failure_gives_502_and_rolls_back(client):
    class Boom:
        messages = None
        def __init__(self): self.messages = self
        def create(self, **kw): raise RuntimeError("down")
    api.state["agent"].client = Boom()
    r = client.post("/chat", json={"message": "hi", "session_id": "s1"})
    assert r.status_code == 502 and api.state["sessions"]["s1"] == []


def test_empty_message_rejected(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_agent_exposes_and_dispatches_quote_tool():
    assert "get_repair_quote" in [t["name"] for t in agent_mod.TOOLS]
    out = agent_mod._run_tool("get_repair_quote", {})
    assert out["grand_total"] == 1800.0
