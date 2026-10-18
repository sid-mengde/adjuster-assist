"""Claude-powered claim assistant that answers from the warranty contract via tools."""
import json
import os
from datetime import date

from anthropic import Anthropic
from dotenv import load_dotenv

import claim_review
import tools

load_dotenv()
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 6  # tool-use round trips per user message

TOOLS = [
    {
        "name": "match_component",
        "description": "Check whether a broken component (e.g. 'HVAC compressor', 'dishwasher') is covered "
                       "under the customer's plan tier (systems vs appliances). Returns verdict "
                       "covered | not_in_plan_tier | no_match, with contract citations.",
        "input_schema": {"type": "object", "properties": {
            "component": {"type": "string", "description": "The broken part or appliance"},
            "plan_type": {"type": "string", "description": "Only if the user states a tier different from the contract, e.g. 'Systems Only'"},
        }, "required": ["component"]},
    },
    {
        "name": "check_waiting_period",
        "description": "Check whether a breakdown occurred after the initial waiting window and within the coverage term.",
        "input_schema": {"type": "object", "properties": {
            "breakdown_date": {"type": "string", "description": "YYYY-MM-DD"},
            "activation_date": {"type": "string", "description": "YYYY-MM-DD, when the policy took effect"},
        }, "required": ["breakdown_date", "activation_date"]},
    },
    {
        "name": "get_limits",
        "description": "Get dollar limits: the service fee and any per-item cap. Pass a component for its cap, "
                       "or omit to list all caps.",
        "input_schema": {"type": "object", "properties": {
            "component": {"type": "string", "description": "Optional broken part or appliance"},
        }},
    },
    {
        "name": "recommend_claim_approval",
        "description": "Run the full claim review and return an approve / deny / needs_review recommendation "
                       "with an audit log. Use when the user asks for a claim approval recommendation or "
                       "whether the claim should be approved. Needs the policy activation date and the "
                       "breakdown date (ask the user if missing). Does NOT use the inspection date as the breakdown date.",
        "input_schema": {"type": "object", "properties": {
            "activation_date": {"type": "string", "description": "YYYY-MM-DD, when the policy took effect"},
            "breakdown_date": {"type": "string", "description": "YYYY-MM-DD, when the equipment failed"},
            "claim_ref": {"type": "string", "description": "Inspection report claim ref; optional if only one report is loaded"},
        }, "required": ["activation_date", "breakdown_date"]},
    },
    {
        "name": "get_repair_quote",
        "description": "Get the repair quotation GRAND TOTAL (and line items) from the technician's inspection "
                       "report. Use for any question about the quote, estimate, or repair cost. Pass claim_ref "
                       "(e.g. 'HW-88301-A') when the user gives one; omit it if only one report is loaded.",
        "input_schema": {"type": "object", "properties": {
            "claim_ref": {"type": "string", "description": "Work order / claim reference of the inspection report"},
        }},
    },
]


def _run_tool(name, args, client=None):
    try:
        if name == "match_component":
            return tools.match_component(args["component"], args.get("plan_type"))
        if name == "check_waiting_period":
            return tools.check_waiting_period(date.fromisoformat(args["breakdown_date"]),
                                              date.fromisoformat(args["activation_date"]))
        if name == "get_limits":
            return tools.get_limits(args.get("component"))
        if name == "recommend_claim_approval":
            return claim_review.recommend_claim_approval(
                args.get("claim_ref"), date.fromisoformat(args["activation_date"]),
                date.fromisoformat(args["breakdown_date"]), client)
        if name == "get_repair_quote":
            return tools.get_repair_quote(args.get("claim_ref"))
        return {"error": f"unknown tool {name}"}
    except Exception as e:  # surfaced to the model so it can ask the user to fix input
        return {"error": f"{type(e).__name__}: {e}"}


def _for_model(out):
    """The full audit log is returned to the client separately; the chat model only needs the step summaries."""
    if isinstance(out, dict) and "audit_log" in out:
        return {**{k: v for k, v in out.items() if k != "audit_log"},
                "audit_log_summary": [f"{s['step']}. {s['summary']}" for s in out["audit_log"]]}
    return out


def system_prompt():
    return f"""You are a claims-verification assistant for adjusters at Premier Home Shield, answering \
questions about one home warranty contract. Today's date is {date.today().isoformat()}.

Rules:
- Answer ONLY from tool results. Never invent coverage, dates, dollar amounts or policy terms.
- Use match_component for "is X covered", check_waiting_period for waiting-window questions, \
get_limits for fees and caps, get_repair_quote for the repair quotation amount. Combine tools when a question needs more than one.
- check_waiting_period needs both the activation date and the breakdown date. If either is missing, \
ask the user; do not guess. Resolve relative dates ("last Tuesday") using today's date.
- Quote amounts come ONLY from get_repair_quote (the technician's report); never compute or estimate them. \
If it returns ambiguous, ask which claim ref. If it has a warning, surface it.
- For approval recommendations call recommend_claim_approval (never decide yourself). Report its \
recommendation, approved amount, reasoning and risks faithfully, do not soften or change them, and say the \
full audit log is attached. Never treat the inspection date as the breakdown date: ask for it.
- If a tool returns no_match or an error, say so plainly and recommend human review.
- If a cap is None, say the contract states no dollar cap for that item (not that it is unlimited).
- You only cover coverage-tier, waiting-period, cost-limit and repair-quote and claim-recommendation questions; politely decline others \
(e.g. legal advice, claim approval decisions).
- Be concise. End answers with the actual citations, e.g. (Section 2, page 1); never print a placeholder."""


class Agent:
    def __init__(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env")
        self.client = Anthropic(api_key=key)

    def chat(self, messages):
        """messages: Anthropic-format history ending with the user turn. Mutates it; returns (reply, tool_calls)."""
        calls = []
        for _ in range(MAX_TURNS):
            resp = self.client.messages.create(model=MODEL, max_tokens=1024, system=system_prompt(),
                                               tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text"), calls
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = _run_tool(b.name, b.input, self.client)
                    calls.append({"tool": b.name, "input": b.input, "result": out})
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(_for_model(out), default=str)})
            messages.append({"role": "user", "content": results})
        return "I couldn't complete that request; please rephrase or ask a human adjuster.", calls
