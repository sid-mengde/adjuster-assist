"""Shared plumbing: run context, dataset loading, result rows, aggregation and gate checks."""
import json
import operator
import random
from dataclasses import dataclass, field
from pathlib import Path

from evals.config import DATASETS, GATES

OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, ">": operator.gt, "<": operator.lt}


@dataclass
class Context:
    sample: int | None = None      # cap on cases per dataset (deterministic, seeded); None = all
    repeat: int = 3                # repeats of each claim for the consistency eval (F12)
    seed: int = 0
    judge: bool = True             # call the LLM judge for E2/E4/E5/E6/F13/G flags
    workers: int = 4
    results_dir: Path = Path(".")
    replay: Path | None = None     # reuse recorded trajectories instead of calling the agent
    usage: list = field(default_factory=list)         # one dict per LLM call (latency, tokens)
    trajectories: list = field(default_factory=list)  # recorded conversations (D, E, G)
    claim_runs: list = field(default_factory=list)    # recorded claim reviews (F, G1)
    extra: dict = field(default_factory=dict)         # additional metrics a layer wants to publish
    skipped: dict = field(default_factory=dict)       # eval id -> reason

    def pick(self, rows, key=None):
        """Deterministic subsample that keeps every group in `key` represented."""
        if not self.sample or len(rows) <= self.sample:
            return rows
        rng = random.Random(self.seed)
        if key is None:
            return sorted(rng.sample(rows, self.sample), key=rows.index)
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        per = max(1, self.sample // len(groups))
        chosen = [x for g in groups.values() for x in rng.sample(g, min(per, len(g)))]
        return sorted(chosen, key=rows.index)


def load(name):
    path = DATASETS / name
    if name.endswith(".json"):
        return json.loads(path.read_text())
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row(ev, case, passed, **detail):
    """One scored case. passed=None means 'not scored' (excluded from rates)."""
    return {"eval": ev, "case": case, "passed": None if passed is None else bool(passed), "detail": detail}


def aggregate(rows):
    """Per-eval pass_rate/n/failed over scored rows."""
    out, by = {}, {}
    for r in rows:
        by.setdefault(r["eval"], []).append(r)
    for ev, rs in by.items():
        scored = [r for r in rs if r["passed"] is not None]
        if scored:
            out[f"{ev}.pass_rate"] = sum(r["passed"] for r in scored) / len(scored)
            out[f"{ev}.n"] = len(scored)
            out[f"{ev}.failed"] = sum(not r["passed"] for r in scored)
    return out


def check_gates(metrics, gates=GATES):
    """Returns [(metric, op, threshold, value, ok)] for every gate whose metric was measured."""
    res = []
    for name, (op, thr) in gates.items():
        if name in metrics:
            res.append((name, op, thr, metrics[name], OPS[op](metrics[name], thr)))
    return res


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
