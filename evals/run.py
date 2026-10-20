"""Run the eval suite and log it to MLflow.

    python -m evals.run                         # deterministic suite (no API calls): layers A, B, C, F (stubbed), H
    python -m evals.run --suite llm --sample 20 # LLM suite on a sample: F (live), D, E, G
    python -m evals.run --suite all --set-baseline
    python -m evals.run --layers A,C --no-mlflow
    mlflow ui --backend-store-uri sqlite:///evals/results/mlflow.db
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from evals.config import ROOT, RESULTS

# (layer letter, label, kind, module function). kind: det runs in the det suite, llm needs the API.
DET_LAYERS = ["A", "B", "C", "F", "H"]
LLM_LAYERS = ["F", "D", "G", "E"]
NOT_IMPLEMENTED = {"H8": "browser smoke test needs Playwright; run the UI by hand until it is added"}


def _steps(layers, include_det, include_llm):
    from evals.layers import a_ingestion, b_retrieval, c_tools, d_agent, f_recommendation, h_service
    steps = []
    table = {
        ("A", "det"): a_ingestion.run, ("B", "det"): b_retrieval.run, ("C", "det"): c_tools.run,
        ("F", "det"): lambda ctx: f_recommendation.run(ctx, live=False), ("H", "det"): h_service.run,
        ("F", "llm"): lambda ctx: f_recommendation.run(ctx, live=True),
        ("D", "llm"): lambda ctx: d_agent.run(ctx, "D"), ("G", "llm"): lambda ctx: d_agent.run(ctx, "G"),
        ("E", "llm"): lambda ctx: d_agent.run(ctx, "E"),
    }
    for kind, enabled in (("det", include_det), ("llm", include_llm)):
        if enabled:
            steps += [(f"{l}-{kind}", fn) for (l, k), fn in table.items() if k == kind and l in layers]
    return steps


def render_report(suite, metrics, gates, regressions, rows, skipped, ctx, elapsed):
    lines = [f"# Eval report: {suite}", "", f"{len(rows)} scored cases in {elapsed:.0f}s. "
             f"Gates: {sum(g[4] for g in gates)}/{len(gates)} passed.", ""]
    if skipped:
        lines += ["## Not run", ""] + [f"- **{k}**: {v}" for k, v in sorted(skipped.items())] + [""]
    lines += ["## Gates", "", "| metric | requirement | value | |", "|---|---|---|---|"]
    lines += [f"| {n} | {o} {t} | {v:.3f} | {'PASS' if ok else '**FAIL**'} |" for n, o, t, v, ok in gates]
    if regressions:
        lines += ["", "## Regressions vs baseline", ""] + [f"- {k}: {a:.3f} -> {b:.3f}" for k, a, b in regressions]
    by = {}
    for r in rows:
        by.setdefault(r["eval"], []).append(r)
    lines += ["", "## Results by eval", "", "| eval | cases | pass rate |", "|---|---|---|"]
    for ev, rs in sorted(by.items()):
        scored = [r for r in rs if r["passed"] is not None]
        if scored:
            lines.append(f"| {ev} | {len(scored)} | {sum(r['passed'] for r in scored) / len(scored):.1%} |")
    lines += ["", "## Failing cases (first 8 per eval)", ""]
    for ev, rs in sorted(by.items()):
        bad = [r for r in rs if r["passed"] is False][:8]
        for r in bad:
            lines.append(f"- **{ev} {r['case']}**: `{json.dumps(r['detail'], default=str)[:300]}`")
    extras = {k: v for k, v in metrics.items() if k.startswith(("H5", "H6", "B2.", "B1.m", "F8.", "C1x"))}
    if extras:
        lines += ["", "## Other measurements", ""] + [f"- {k}: {v:.4g}" for k, v in sorted(extras.items())]
    return "\n".join(lines) + "\n"


def preflight(judge):
    """One tiny call per model, so a bad key or model name stops the run instead of scoring as 'agent failures'."""
    from anthropic import Anthropic
    from evals.config import AGENT_MODEL, JUDGE_MODEL
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for model in dict.fromkeys([AGENT_MODEL] + ([JUDGE_MODEL] if judge else [])):
        try:
            client.messages.create(model=model, max_tokens=8, messages=[{"role": "user", "content": "ping"}])
        except Exception as e:
            sys.exit(f"Preflight failed for model '{model}': {getattr(e, 'message', None) or type(e).__name__}\n"
                     f"Fix the key or model (CLAUDE_MODEL / EVAL_JUDGE_MODEL), or use --suite det.")
    print("preflight ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["det", "llm", "all"], default="det")
    ap.add_argument("--layers", help="comma-separated layer letters (A,B,C,D,E,F,G,H); default: all in the suite")
    ap.add_argument("--sample", type=int, help="max cases per dataset (seeded, keeps every group represented)")
    ap.add_argument("--repeat", type=int, default=3, help="repeats per claim for the consistency eval (F12)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (E2, E4, E5, E6, F13)")
    ap.add_argument("--replay", type=Path, help="directory with trajectories.jsonl / claim_runs.jsonl to re-score without new agent calls")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--set-baseline", action="store_true", help="tag this run as the baseline for its suite")
    ap.add_argument("--compare-baseline", action="store_true", help="flag metrics that dropped against the baseline run")
    ap.add_argument("--no-fail", action="store_true", help="always exit 0")
    a = ap.parse_args(argv)

    os.chdir(ROOT)  # api.py mounts ./static, claim_review writes ./logs
    sys.path.insert(0, str(ROOT))
    load_dotenv()
    from evals.common import Context, aggregate, check_gates

    use_det, use_llm = a.suite in ("det", "all"), a.suite in ("llm", "all")
    wanted = set(a.layers.upper().split(",")) if a.layers else set("ABCDEFGH")
    if use_llm and not os.getenv("ANTHROPIC_API_KEY") and not a.replay:
        sys.exit("ANTHROPIC_API_KEY is not set (add it to .env), or use --suite det / --replay")

    if use_llm and not a.replay:
        preflight(judge=not a.no_judge)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(sample=a.sample, repeat=a.repeat, seed=a.seed, judge=not a.no_judge, workers=a.workers,
                  results_dir=run_dir, replay=a.replay)
    ctx.skipped.update(NOT_IMPLEMENTED)

    rows, t0 = [], time.time()
    for name, fn in _steps(wanted, use_det, use_llm):
        t1 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        got = fn(ctx)
        rows += got
        print(f"    {len(got)} cases in {time.time() - t1:.0f}s", flush=True)

    from evals.layers.h_service import usage_metrics
    metrics = aggregate(rows)
    metrics.update({k: v for k, v in ctx.extra.items() if isinstance(v, (int, float))})
    metrics.update(usage_metrics(ctx))
    gates = check_gates(metrics)

    from evals import calibration
    agree = calibration.agreement()
    tags = {"judge_calibration": f"{min(v for k, v in agree.items() if k != 'n'):.2f} (n={agree['n']})" if agree else "uncalibrated"}
    suite = a.suite if not a.layers else f"{a.suite}:{','.join(sorted(wanted))}"

    (run_dir / "cases.json").write_text(json.dumps(rows, default=str, indent=1))
    from evals.conversations import save
    if ctx.trajectories:
        save(ctx.trajectories, run_dir / "trajectories.jsonl")
    if ctx.claim_runs:
        with open(run_dir / "claim_runs.jsonl", "w") as f:
            for r in ctx.claim_runs:
                f.write(json.dumps(r, default=str) + "\n")

    regressions = []
    report = render_report(suite, metrics, gates, [], rows, ctx.skipped, ctx, time.time() - t0)
    if not a.no_mlflow:
        from evals import mlflow_utils
        run_id, gates, regressions = mlflow_utils.log_suite(
            suite=suite, layers=sorted(wanted), rows=rows, metrics=metrics, skipped=ctx.skipped, run_dir=run_dir,
            params={"sample": a.sample, "repeat": a.repeat, "seed": a.seed, "judge": not a.no_judge, "workers": a.workers,
                    "replay": a.replay}, tags=tags, set_baseline=a.set_baseline, compare_baseline=a.compare_baseline,
            report_md=render_report(suite, metrics, gates, [], rows, ctx.skipped, ctx, time.time() - t0))
        report = render_report(suite, metrics, gates, regressions, rows, ctx.skipped, ctx, time.time() - t0)
        print(f"MLflow run {run_id}  (mlflow ui --backend-store-uri sqlite:///{RESULTS / 'mlflow.db'})")
    (run_dir / "report.md").write_text(report)

    failed = [g for g in gates if not g[4]]
    print(f"\n{len(rows)} cases, gates {len(gates) - len(failed)}/{len(gates)} passed, report: {run_dir / 'report.md'}")
    for n, o, t, v, _ in failed:
        print(f"  GATE FAIL  {n}: {v:.3f} (needs {o} {t})")
    for k, old, new in regressions:
        print(f"  REGRESSION {k}: {old:.3f} -> {new:.3f}")
    return 0 if (a.no_fail or (not failed and not regressions)) else 1


if __name__ == "__main__":
    sys.exit(main())
