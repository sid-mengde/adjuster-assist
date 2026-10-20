"""Judge calibration: export judged replies for a person to label, then measure judge-vs-human agreement.

    python -m evals.calibration export --run evals/results/<run>/trajectories.jsonl --n 40
    (fill the human_* columns in evals/datasets/judge_calibration.csv with TRUE/FALSE)
    python -m evals.calibration score
The judge-based evals (E2, E4, E5, F13) are only trustworthy once agreement is at least 90%.
"""
import argparse
import csv
import json

from evals.config import DATASETS

PATH = DATASETS / "judge_calibration.csv"
FLAGS = [("verdict_matches", "human_verdict_matches"), ("all_claims_supported", "human_all_supported"),
         ("citations_present", "human_citations_present")]


def export(run, n):
    rows = []
    for line in open(run):
        t = json.loads(line)
        for i, turn in enumerate(t["turns"]):
            if turn.get("judge") and turn["calls"]:
                j = turn["judge"]
                rows.append({"id": f"{t['id']}#{i}", "user": " | ".join(x["user"] for x in t["turns"][: i + 1]),
                             "tool_results": json.dumps([c["result"] for c in turn["calls"]], default=str)[:3000],
                             "reply": turn["reply"], **{f"judge_{a}": j.get(a) for a, _ in FLAGS},
                             **{b: "" for _, b in FLAGS}})
    rows = rows[:n]
    with open(PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {PATH}; fill the human_* columns with TRUE/FALSE")


def agreement():
    """{flag: agreement} over labelled rows, plus 'n'. None if the file is missing or unlabelled."""
    if not PATH.exists():
        return None
    rows = list(csv.DictReader(open(PATH)))
    out, n = {}, 0
    for a, b in FLAGS:
        pairs = [(r[f"judge_{a}"].strip().lower(), r[b].strip().lower()) for r in rows if r[b].strip() and r[f"judge_{a}"].strip() not in ("", "None")]
        if pairs:
            out[a] = sum(x == y for x, y in pairs) / len(pairs)
            n = max(n, len(pairs))
    return {**out, "n": n} if out else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "score"])
    ap.add_argument("--run")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.run, a.n)
    else:
        print(agreement() or "no labelled rows yet")
