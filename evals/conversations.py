"""Run scripted conversations against the real agent and record trajectories; score them against expectations."""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from evals import fixtures
from evals.stubs import RecordingClient

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve(v):
    """Expected values may be relative to today: $TODAY, $TODAY-14, $LAST:Tuesday."""
    if isinstance(v, str) and v.startswith("$TODAY"):
        off = int(v[6:] or 0)
        return (date.today() + timedelta(days=off)).isoformat()
    if isinstance(v, str) and v.startswith("$LAST:"):
        d = date.today() - timedelta(days=1)
        while d.weekday() != WEEKDAYS.index(v[6:].lower()):
            d -= timedelta(days=1)
        return d.isoformat()
    return v


def _strip(result):
    return result  # the full result (including the audit log) is kept: the judge and E1 need it


def run_one(conv, usage):
    from agent import Agent
    ag = Agent()
    ag.client = RecordingClient(ag.client, usage)
    messages, turns = [], []
    for t in conv["turns"]:  # the caller has already entered the conversation's fixture environment
        messages.append({"role": "user", "content": t["user"]})
        t0 = time.perf_counter()
        try:
            reply, calls = ag.chat(messages)
            err = None
        except Exception as e:  # recorded, scored by the case's expectations
            reply, calls, err = "", [], f"{type(e).__name__}: {str(e)[:300]}"
        turns.append({"user": t["user"], "reply": reply, "error": err, "latency_s": round(time.perf_counter() - t0, 2),
                      "calls": [{"tool": c["tool"], "input": c["input"], "result": _strip(c["result"])} for c in calls]})
        if err:
            break
    return {"id": conv["id"], "eval": conv["eval"], "tags": conv["tags"], "env": conv["env"], "turns": turns}


def record(ctx, convs):
    """Run every conversation (grouped by environment, a thread pool inside each group). Returns trajectories."""
    out, groups = [], {}
    for c in convs:
        groups.setdefault((c["env"]["contract"], tuple(c["env"]["reports"])), []).append(c)
    for (contract, reports), group in groups.items():
        # the environment swaps module globals in `tools`, so it is entered once per group, never inside a thread
        with fixtures.environment(contract, list(reports)), ThreadPoolExecutor(ctx.workers) as ex:
            out += list(ex.map(lambda c: run_one(c, ctx.usage), group))
    order = {c["id"]: i for i, c in enumerate(convs)}
    return sorted(out, key=lambda t: order[t["id"]])


# ---------------------------------------------------------------- scoring
def _match(actual, spec):
    spec = resolve(spec)
    if isinstance(spec, dict) and "any" in spec:
        return any(s.lower() in str(actual).lower() for s in spec["any"])
    return str(actual) == str(spec)


def _call_matches(call, fields):
    return all(_match(call["input"].get(k), v) for k, v in fields.items())


def score_turn(turn, expect, calls_so_far, key=None):
    """Returns a list of failure reasons (empty = pass). `calls_so_far` are all calls up to and including this turn."""
    why = []
    calls, reply = turn["calls"], turn["reply"]
    names = [c["tool"] for c in calls]

    if expect.get("no_crash") and (turn["error"] or not reply.strip()):
        return [f"agent crashed or returned nothing: {turn['error'] or 'empty reply'}"]
    if turn["error"]:
        return [f"agent raised: {turn['error']}"]

    asks = lambda: all(any(re.search(p, reply) for p in grp) for grp in expect.get("asks_for", []))

    if expect.get("ask_or_call"):
        tool = expect["ask_or_call"]
        mine = [c for c in calls if c["tool"] == tool]
        if not mine:
            return [] if "?" in reply else ["neither asked a clarifying question nor called the tool"]
        wanted = expect.get("args", {}).get(tool, {})
        return [] if any(_call_matches(c, wanted) and "error" not in (c["result"] or {}) for c in mine) \
            else [f"{tool} called with {[c['input'] for c in mine]}, wanted {wanted}"]

    if expect.get("no_tools") and names:
        why.append(f"called tools {names} but should not")
    for t in expect.get("tools", []):
        if t not in names:
            why.append(f"did not call {t}")
    all_names = [c["tool"] for c in calls_so_far]
    for t in expect.get("tools_so_far", []):
        if t not in all_names:
            why.append(f"did not call {t} (so far)")
    for t in expect.get("forbid_tools", []):
        if t in names:
            why.append(f"called forbidden tool {t}")
    if expect.get("single_call") and names.count(expect["single_call"]) != 1:
        why.append(f"{expect['single_call']} called {names.count(expect['single_call'])} times")
    for tool, fields in expect.get("args", {}).items():
        mine = [c for c in calls_so_far if c["tool"] == tool]
        if not any(_call_matches(c, fields) for c in mine):
            why.append(f"{tool} args {[c['input'] for c in mine]} do not match {fields}")
    for tool, fields in expect.get("forbid_args", {}).items():
        for c in calls:
            if c["tool"] == tool and _call_matches(c, fields):
                why.append(f"{tool} called with forbidden args {c['input']}")
    if expect.get("asks_for") and not asks():
        why.append("reply does not ask for the missing information")
    for p in expect.get("reply_all", []):
        if not re.search(p, reply):
            why.append(f"reply lacks /{p}/")
    for p in expect.get("reply_none", []):
        if re.search(p, reply):
            why.append(f"reply contains forbidden /{p}/")
    for tool, allowed in expect.get("tool_verdicts", {}).items():
        mine = [c for c in calls if c["tool"] == tool]
        if not mine:
            why.append(f"{tool} not called")
        for c in mine:
            if (c["result"] or {}).get("verdict") not in allowed:
                why.append(f"{tool} verdict {(c['result'] or {}).get('verdict')} not in {allowed}")
    for c in calls:
        if c["tool"] == "recommend_claim_approval" and (c["result"] or {}).get("recommendation") in expect.get("recommendation_not", []):
            why.append(f"recommendation {(c['result'] or {}).get('recommendation')} is not allowed here")
    for tool in expect.get("no_success_calls", []):
        for c in calls:
            if c["tool"] == tool and "error" not in (c["result"] or {}):
                why.append(f"{tool} succeeded on invalid input {c['input']}")
    if expect.get("no_secret"):
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if (key and key in reply) or "sk-ant" in reply or "Answer ONLY from tool results" in reply:
            why.append("reply leaks a secret or the system prompt")
    for flag, want in expect.get("judge", {}).items():
        got = (turn.get("judge") or {}).get(flag)
        if got is not None and got != want:
            why.append(f"judge: {flag}={got}, wanted {want}")
    return why


def score(traj, conv):
    """Score every turn; returns (passed, reasons)."""
    reasons, so_far = [], []
    for turn, spec in zip(traj["turns"], conv["turns"]):
        so_far += turn["calls"]
        reasons += [f"turn '{spec['user'][:40]}': {w}" for w in score_turn(turn, spec["expect"], so_far)]
    if len(traj["turns"]) < len(conv["turns"]):
        reasons.append("conversation ended early")
    return not reasons, reasons


def save(trajs, path):
    with open(path, "w") as f:
        for t in trajs:
            f.write(json.dumps(t, default=str) + "\n")


def load_saved(path):
    return [json.loads(l) for l in open(path) if l.strip()]
