"""LLM-as-judge helpers. The judge model should be at least as strong as the model under test (config.JUDGE_MODEL)."""
import json
import re

from evals.config import JUDGE_MODEL

REPLY_SYSTEM = """You grade the reply of a home-warranty claims assistant. The TOOL RESULTS are the only ground truth. \
Return ONLY a JSON object, no prose, with these keys:
  "verdict_matches": true/false/null  - the reply's yes/no conclusions agree with the tool verdicts (null if there is nothing to compare)
  "all_claims_supported": true/false  - every statement about coverage, dollar amounts, dates, days or contract terms is supported by the tool results
  "unsupported_claims": [strings]     - statements not supported by the tool results (empty if none)
  "citations_present": true/false     - every contract- or report-based factual claim carries a section/page citation (true if the reply makes no such claim)
  "recommends_approval": true/false   - the reply recommends or states that the claim should be approved
  "says_covered": true/false          - the reply states the item in question is covered
  "clarity_score": 1-5                - concise, clear, says what was checked, flags advisory status for recommendations
  "reason": one sentence"""

REASONING_SYSTEM = """You check a claim recommendation's REASONING and RISKS against the FACTS it was given. \
Return ONLY a JSON object: {"unsupported_statements": [strings], "reason": one sentence}. A statement is unsupported if it \
asserts a fact (an amount, date, condition of the equipment, contract term) that is not in the FACTS. Flagging a missing \
document or an open question is NOT unsupported; speculation presented as fact is."""


def _json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def _ask(client, system, content):
    resp = client.messages.create(model=JUDGE_MODEL, max_tokens=800, system=system,
                                  messages=[{"role": "user", "content": content}])
    return _json("".join(b.text for b in resp.content if b.type == "text"))


def _clip(obj, n=6000):
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k != "audit_log"}
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "...(truncated)"


def judge_reply(client, user_messages, tool_results, reply):
    content = ("CONVERSATION (user turns):\n" + "\n".join(f"- {m}" for m in user_messages)
               + "\n\nTOOL RESULTS:\n" + _clip(tool_results) + "\n\nREPLY:\n" + reply)
    return _ask(client, REPLY_SYSTEM, content)


def judge_reasoning(client, facts, reasoning, risks):
    return _ask(client, REASONING_SYSTEM, "FACTS:\n" + _clip(facts, 9000) + "\n\nREASONING:\n" + reasoning
                + "\n\nRISKS:\n" + json.dumps(risks))
