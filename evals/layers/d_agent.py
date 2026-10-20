"""D (agent behaviour), E (answer grounding) and G (robustness). One recorded set of conversations feeds all three."""
import json
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv

from agent import MAX_TURNS
from evals import conversations as C, judges
from evals.common import load, row
from evals.config import JUDGE_MODEL
from evals.stubs import RecordingClient

load_dotenv()
D_EVALS = {"D1", "D3", "D4", "D5", "D6", "D7", "D9", "D10"}
G_EVALS = {"G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"}
JUDGE_SAMPLE = 40  # replies judged for E2/E4/E5/E6 (plus every case whose expectation needs a judge flag)

NUM = re.compile(r"\d[\d,]*\.?\d*")
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
DAYS = re.compile(r"(?i)\b(\d+)[\s-]*days?\b|\bday\s+(\d+)\b")
CITE = re.compile(r"(?i)section\s+(\d+)[^\d\n]{0,25}?(?:page|pg\.?|p\.)\s*(\d+)")
PLACEHOLDER = re.compile(r"(?i)section\s+N\b|page\s+P\b|\[citation\]|\(Section X")


def _nums(text):
    out = set()
    for m in NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "").rstrip(".")), 2))
        except ValueError:
            pass
    return out


def run_conversations(ctx):
    """Record (or replay) every D/G conversation once; cached on the context."""
    if "conversations" in ctx.extra:
        return ctx.extra["conversations"]
    convs = load("conversations.jsonl")
    convs = ctx.pick(convs, key=lambda c: c["eval"])
    if ctx.replay and (ctx.replay / "trajectories.jsonl").exists():
        trajs = C.load_saved(ctx.replay / "trajectories.jsonl")
        keep = {c["id"] for c in convs}
        trajs = [t for t in trajs if t["id"] in keep] or trajs
    else:
        trajs = C.record(ctx, convs)
    by_id = {c["id"]: c for c in load("conversations.jsonl")}
    ctx.trajectories = trajs
    ctx.extra["conversations"] = (trajs, by_id)
    return ctx.extra["conversations"]


def run_judges(ctx, trajs, by_id):
    """Judge replies in place (turn['judge']). Skipped when --no-judge or no API key."""
    if not ctx.judge:
        ctx.skipped.update({e: "judge disabled (--no-judge)" for e in ("E2", "E4", "E5", "E6")})
        return
    client = RecordingClient(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]), ctx.usage, label="judge")
    wanted = []
    for t in trajs:
        for turn, spec in zip(t["turns"], by_id[t["id"]]["turns"]):
            if turn["reply"] and spec["expect"].get("judge"):
                wanted.append((t, turn))
    taken = {id(turn) for _, turn in wanted}
    sampled = [(t, turn) for t in trajs for turn in t["turns"] if turn["calls"] and turn["reply"] and id(turn) not in taken]
    wanted += sampled[:JUDGE_SAMPLE]
    for t, turn in wanted:
        users = [x["user"] for x in t["turns"]]
        try:
            turn["judge"] = judges.judge_reply(client, users, [c["result"] for c in turn["calls"]], turn["reply"])
        except Exception as e:  # a judge failure must not look like an agent failure
            turn["judge"] = None
            turn["judge_error"] = f"{type(e).__name__}: {str(e)[:200]}"


def run(ctx, which="D"):
    trajs, by_id = run_conversations(ctx)
    if "judged" not in ctx.extra:
        run_judges(ctx, trajs, by_id)
        ctx.extra["judged"] = True
    rows = []
    if which == "D":
        for t in trajs:
            if t["eval"] in D_EVALS:
                ok, why = C.score(t, by_id[t["id"]])
                rows.append(row(t["eval"], t["id"], ok, reasons=why, replies=[x["reply"][:300] for x in t["turns"]]))
        for t in trajs:  # D8 no loops: within the round-trip limit and no repeated identical call in a turn
            for i, turn in enumerate(t["turns"]):
                seen = [json.dumps([c["tool"], c["input"]], sort_keys=True, default=str) for c in turn["calls"]]
                rows.append(row("D8", f"{t['id']}#{i}", len(seen) <= MAX_TURNS * 2 and len(seen) == len(set(seen)), calls=len(seen)))
    elif which == "G":
        for t in trajs:
            if t["eval"] in G_EVALS:
                ok, why = C.score(t, by_id[t["id"]])
                rows.append(row(t["eval"], t["id"], ok, reasons=why, replies=[x["reply"][:300] for x in t["turns"]]))
    else:  # E
        rows += grounding_rows(ctx, trajs, by_id)
    return rows


# ---------------------------------------------------------------- E: grounding
def grounding_rows(ctx, trajs, by_id):
    rows = []
    for t in trajs:
        users = " ".join(x["user"] for x in t["turns"])
        results_so_far = []
        for i, turn in enumerate(t["turns"]):
            results_so_far += [c["result"] for c in turn["calls"]]
            if not turn["calls"] or not turn["reply"]:
                continue
            case = f"{t['id']}#{i}"
            blob = json.dumps(results_so_far, default=str) + " " + users
            allowed = _nums(blob)
            reply = turn["reply"]

            # E1 every dollar amount and day count in the reply appears in the tool results (or in the user's message)
            claimed = [float(m.replace(",", "")) for m in MONEY.findall(reply)]
            claimed += [float(a or b) for a, b in DAYS.findall(reply)]
            bad = sorted({x for x in claimed if round(x, 2) not in allowed})
            rows.append(row("E1", case, not bad, ungrounded=bad))

            # E3 citations point at real (section, page) pairs from the tool results; no placeholders
            real = set()
            for res in results_so_far:
                for c in (res.get("citations", []) if isinstance(res, dict) else []):
                    real.add((c["section"].split(".")[0], str(c["page"])))
            cites = CITE.findall(reply)
            fake = [c for c in cites if c not in real]
            rows.append(row("E3", case, not fake and not PLACEHOLDER.search(reply), cited=cites, not_in_tool_results=fake,
                            placeholder=bool(PLACEHOLDER.search(reply))))

            # E7 a recommendation reply mentions the attached audit log
            if any(c["tool"] == "recommend_claim_approval" and c["result"].get("recommendation") for c in turn["calls"]):
                rows.append(row("E7", case, bool(re.search(r"(?i)audit|log", reply))))

            j = turn.get("judge")
            if j:
                rows.append(row("E2", case, j.get("verdict_matches") is not False, judge=j.get("reason")))
                rows.append(row("E4", case, bool(j.get("citations_present")), judge=j.get("reason")))
                rows.append(row("E5", case, bool(j.get("all_claims_supported")), unsupported=j.get("unsupported_claims")))
                rows.append(row("E6", case, None, score=j.get("clarity_score"), judge=j.get("reason")))
    scores = [r["detail"]["score"] for r in rows if r["eval"] == "E6" and isinstance(r["detail"].get("score"), (int, float))]
    if scores:
        ctx.extra["E6.mean_score"] = sum(scores) / len(scores)
    return rows
