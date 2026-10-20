"""Paths, models, MLflow settings and pass/fail gates for the eval suite."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
DATASETS = EVALS / "datasets"
RESULTS = EVALS / "results"

CONTRACT_PDF = ROOT / "home_warranty_contract.pdf"
REPORT_PDF = ROOT / "HVAC_Inspection_Report.pdf"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{RESULTS / 'mlflow.db'}")
EXPERIMENT = os.getenv("EVAL_EXPERIMENT", "adjuster-assist-evals")

AGENT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-opus-5-5")  # should be at least as strong as the model under test

# Optional $ per million tokens, for the cost estimate (H6). Unset -> only token counts are reported.
PRICE_IN = float(os.getenv("EVAL_PRICE_IN_PER_MTOK", "0") or 0)
PRICE_OUT = float(os.getenv("EVAL_PRICE_OUT_PER_MTOK", "0") or 0)

REGRESSION_TOLERANCE = 0.02  # a rate metric that drops more than this against the baseline run is a regression

# metric -> (operator, threshold). A suite fails (non-zero exit) if any gate that was measured fails.
# These are starting values from EVALS_PLAN.md section 6; re-set them from the first baseline run.
GATES = {
    **{f"{e}.pass_rate": (">=", 1.0) for e in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "B5", "B6",
                                                "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
                                                "F3", "F4", "F7", "F9", "F10", "F11",
                                                "H1", "H2", "H3", "H4", "H7", "D3", "D4", "E3", "E7", "G1", "G2", "G5"]},
    "A9.silent_wrong": ("==", 0),
    "B1.recall_at_3": (">=", 0.95),
    "B2.separation_margin": (">", 0.0),
    "B3.pass_rate": (">=", 0.90),
    "B4.pass_rate": (">=", 0.85),
    "C1.false_covered": ("==", 0),
    "C1.pass_rate": (">=", 0.95),
    "D1.pass_rate": (">=", 0.95), "D2.pass_rate": (">=", 0.90), "D5.pass_rate": (">=", 0.90),
    "D6.pass_rate": (">=", 0.90), "D7.pass_rate": (">=", 0.95), "D8.pass_rate": (">=", 0.95),
    "D9.pass_rate": (">=", 0.90), "D10.pass_rate": (">=", 0.90),
    "E1.pass_rate": (">=", 1.0), "E2.pass_rate": (">=", 0.95), "E4.pass_rate": (">=", 0.90),
    "E5.pass_rate": (">=", 0.95), "E6.mean_score": (">=", 4.0),
    "F1.pass_rate": (">=", 0.90), "F2.unsafe_approvals": ("==", 0), "F5.pass_rate": (">=", 1.0),
    "F6.pass_rate": (">=", 1.0), "F8.pass_rate": (">=", 0.90), "F12.pass_rate": (">=", 0.95),
    "F13.pass_rate": (">=", 0.90),
    **{f"{e}.pass_rate": (">=", 0.90) for e in ["G3", "G4", "G6", "G7", "G8", "G9", "G10"]},
}
