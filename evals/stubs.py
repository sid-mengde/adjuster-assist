"""Test doubles for the Anthropic client (no network) and a recording wrapper for real calls."""
import time
from types import SimpleNamespace as NS


class FakeClaude:
    """Stands in for the recommendation call. `rec` is the submit_recommendation input, a dict or a callable(kwargs)."""

    def __init__(self, rec=None, fail=False, no_tool=False):
        self.rec = rec or {"recommendation": "approve", "reasoning": "All checks pass.",
                           "exclusion_assessment": "No exclusion applies.", "exclusion_triggered": False, "risks": []}
        self.fail, self.no_tool, self.calls = fail, no_tool, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("model down")
        if self.no_tool:
            return NS(content=[NS(type="text", text="I recommend approval.")])
        rec = self.rec(kw) if callable(self.rec) else self.rec
        return NS(content=[NS(type="tool_use", input=rec)])


def always_approve():
    return FakeClaude()


class RecordingClient:
    """Wraps a real Anthropic client; appends one dict per call (latency, tokens, request, response text) to `sink`."""

    def __init__(self, inner, sink, label=None):
        self.inner, self.sink, self.label = inner, sink, label
        self.messages = self

    def create(self, **kw):
        t0 = time.perf_counter()
        resp = self.inner.messages.create(**kw)
        names = [t["name"] for t in kw.get("tools", [])]
        usage = getattr(resp, "usage", None)
        self.sink.append({
            "label": self.label or ("recommendation" if "submit_recommendation" in names else "chat"),
            "model": kw.get("model"), "latency_s": round(time.perf_counter() - t0, 3),
            "input_tokens": getattr(usage, "input_tokens", 0) or 0, "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "request_user": kw["messages"][0]["content"] if kw.get("messages") and isinstance(kw["messages"][0]["content"], str) else None,
        })
        return resp
