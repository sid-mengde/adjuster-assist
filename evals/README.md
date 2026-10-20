# Evals

Measures whether the claim assistant gives correct, grounded, safe answers. Plan and rationale: [EVALS_PLAN.md](../EVALS_PLAN.md).
Every run is logged to **MLflow** (one parent run per execution, one nested run per layer).

## Run

```bash
source .venv/bin/activate
python -m evals.run                          # deterministic suite, no API calls (~1 min): A, B, C, F (stubbed model), H
python -m evals.run --suite llm --sample 20  # LLM suite on a sample: F (live), D, E, G
python -m evals.run --suite all --repeat 3   # everything, full datasets
python -m evals.run --layers A,C --no-mlflow
python -m evals.run --set-baseline           # tag this run as the baseline for its suite
python -m evals.run --compare-baseline       # flag metrics that dropped against that baseline
python -m evals.run --suite llm --replay evals/results/runs/<stamp>   # re-score recorded trajectories, no agent calls
mlflow ui --backend-store-uri sqlite:///evals/results/mlflow.db
```

Exit code is 1 if any measured gate fails or a regression is found (`--no-fail` to disable). The LLM suite needs a valid
`ANTHROPIC_API_KEY` in `.env`; a preflight call checks the key and both models first. Environment variables:
`CLAUDE_MODEL` (agent), `EVAL_JUDGE_MODEL` (default `claude-opus-5-5`), `EVAL_PRICE_IN_PER_MTOK` / `EVAL_PRICE_OUT_PER_MTOK` (optional cost estimate).

## Layout

| Path | Purpose |
|---|---|
| `run.py` | CLI, gates, report |
| `config.py` | Paths, models, **gates** (pass/fail thresholds) |
| `documents.py` | Synthetic contract variants (13) and report variants (40) |
| `fixtures.py` | Temp vector stores swapped in behind the real tools; the real `chroma_db` is never touched |
| `datasets/build.py` | Generates the JSONL datasets; `python -m evals.datasets.build` |
| `layers/a_ingestion.py` ... `h_service.py` | One module per layer (A ingestion, B retrieval, C tools, D/E/G agent, F recommendation, H service) |
| `conversations.py` | Records scripted conversations against the real agent and scores them against expectations |
| `judges.py`, `calibration.py` | LLM judge prompts; export/score a human-labelled sample to calibrate it |
| `mlflow_utils.py` | MLflow logging, baseline tagging and comparison |
| `results/` | Git-ignored: MLflow database, artifacts, per-run reports and recorded trajectories |

## Reading a result

- `report.md` in the run directory (also an MLflow artifact) lists gates, pass rates and the first failing cases per eval.
- In MLflow: parent-run metrics are named `<EVAL>.<measure>` (for example `C1.false_covered`, `B1.recall_at_3`, `F2.unsafe_approvals`);
  each layer's per-case table is the `cases_<layer>.json` artifact of its nested run; recorded trajectories are under `recorded/`.
- Tags on the parent run say which commit, models, dataset hash, claim-label sign-off state and judge calibration state produced it.

## Things to know

- **Claim labels are drafts** until the claim approver sets `status: approved` in `datasets/claims_signoff.json`. Treat F-layer
  scores as provisional until then. The expected outcome for a triggered exclusion is `needs_review`.
- **Judge-based evals** (E2, E4, E5, E6, F13) are tagged `uncalibrated` until `python -m evals.calibration` shows at least 90% agreement with human labels.
- **Gates are starting values** (EVALS_PLAN.md section 6). A first run is expected to fail some of them: those are findings, not harness errors.
- Labels in the datasets come from the contract's wording, not from tool output, so a disagreement points at the product.
- The report parser is tested on layouts the template can express; it has not seen other vendors' PDFs.
- `H8` (browser smoke test) is not implemented.
