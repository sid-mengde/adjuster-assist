"""MLflow logging: one parent run per suite execution, one nested run per layer, baseline comparison by tag."""
import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from evals.common import check_gates
from evals.config import (AGENT_MODEL, DATASETS, EXPERIMENT, JUDGE_MODEL, REGRESSION_TOLERANCE, RESULTS, ROOT, TRACKING_URI)

COUNT_METRICS = ("false_covered", "unsafe_approvals", "silent_wrong")  # any increase is a regression
RATE_SUFFIXES = (".pass_rate", ".recall_at_1", ".recall_at_3", ".mrr", ".mean_score", ".separation_margin")


def setup():
    RESULTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(RESULTS / "artifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _git():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def dataset_hash():
    h = hashlib.sha256()
    for p in sorted(DATASETS.glob("*.json*")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:12]


def signoff_status():
    try:
        s = json.loads((DATASETS / "claims_signoff.json").read_text())
        return s["status"] if s.get("approved_by") is None or s["status"] != "approved" else f"approved by {s['approved_by']}"
    except Exception:
        return "unknown"


def _clean(metrics):
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def find_baseline(suite):
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], f"tags.eval_baseline = 'true' and tags.eval_suite = '{suite}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0] if runs else None


def compare(current, baseline_metrics):
    """Regressions against the baseline run: [(metric, baseline, current)]."""
    out = []
    for k, now in current.items():
        old = baseline_metrics.get(k)
        if old is None:
            continue
        if k.endswith(COUNT_METRICS) and now > old:
            out.append((k, old, now))
        elif k.endswith(RATE_SUFFIXES) and old - now > REGRESSION_TOLERANCE:
            out.append((k, old, now))
    return out


def log_suite(*, suite, layers, rows, metrics, skipped, run_dir, params, tags, set_baseline, compare_baseline, report_md):
    """rows: all scored rows. Returns (run_id, gate_results, regressions)."""
    setup()
    gates = check_gates(metrics)
    failed = [g for g in gates if not g[4]]
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["eval"][0], []).append(r)

    baseline = find_baseline(suite) if compare_baseline else None
    regressions = compare(metrics, baseline.data.metrics) if baseline else []

    with mlflow.start_run(run_name=f"{suite}-{Path(run_dir).name}") as parent:
        mlflow.set_tags({"eval_suite": suite, "eval_layers": ",".join(layers), "git": _git(), "agent_model": AGENT_MODEL,
                         "judge_model": JUDGE_MODEL, "dataset_hash": dataset_hash(), "claims_labels": signoff_status(),
                         "gates_passed": str(not failed), "regressions": str(len(regressions)), **tags})
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(_clean(metrics))
        mlflow.log_metrics({f"gate.{n}": float(ok) for n, _, _, _, ok in gates})
        mlflow.log_metric("gates.failed", len(failed))
        mlflow.log_metric("gates.measured", len(gates))
        for layer, rs in sorted(by_layer.items()):
            with mlflow.start_run(run_name=f"layer-{layer}", nested=True):
                mlflow.set_tag("layer", layer)
                mlflow.log_metrics(_clean({k: v for k, v in metrics.items() if k.startswith(layer)}))
                mlflow.log_table(data={"eval": [r["eval"] for r in rs], "case": [r["case"] for r in rs],
                                       "passed": [r["passed"] for r in rs],
                                       "detail": [json.dumps(r["detail"], default=str)[:2000] for r in rs]},
                                 artifact_file=f"cases_{layer}.json")
        mlflow.log_dict({"gates": [{"metric": n, "op": o, "threshold": t, "value": v, "ok": ok} for n, o, t, v, ok in gates],
                         "regressions": [{"metric": k, "baseline": a, "current": b} for k, a, b in regressions],
                         "skipped": skipped}, "gates.json")
        mlflow.log_text(report_md, "report.md")
        for name in ("trajectories.jsonl", "claim_runs.jsonl"):
            if (Path(run_dir) / name).exists():
                mlflow.log_artifact(str(Path(run_dir) / name), "recorded")
        if set_baseline:
            client = MlflowClient()
            old = find_baseline(suite)
            if old:
                client.delete_tag(old.info.run_id, "eval_baseline")
            client.set_tag(parent.info.run_id, "eval_baseline", "true")
        return parent.info.run_id, gates, regressions
