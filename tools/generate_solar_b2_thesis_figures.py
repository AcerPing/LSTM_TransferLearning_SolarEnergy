#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar Experiment B/B2 — Thesis Figure Generation v1

Purpose
-------
Read-only post-processing for Chapter 4 thesis figures.

This script:
1. Reads the locked WOTL and B2-P1 training histories and Test predictions.
2. Verifies Test alignment (row count, timestamps, y_true).
3. Recalculates original-scale MAE/MSE/RMSE/R² and cross-checks official JSON.
4. Verifies residual sign convention: residual = y_true - y_pred.
5. Generates 10 standalone figures:
   - 2 learning curves
   - 2 prediction plots
   - 2 observed-vs-predicted (YY) plots
   - 2 residual plots
   - 2 error histograms
6. Writes metrics_recheck.json and figure_manifest.json.

It does NOT:
- train a model
- load a model/checkpoint
- call model.predict()
- alter learning rate / epoch / checkpoint selection
- clip negative predictions
- overwrite Experiment B/B2 artifacts

Run from the repository root:
    python scripts/generate_solar_b2_thesis_figures.py

Optional:
    python scripts/generate_solar_b2_thesis_figures.py --repo-root "D:/.../LSTM_TransferLearning_SolarEnergy"

By default the script refuses to overwrite an existing figure output.
Use --overwrite only when intentionally regenerating the same thesis figures
from the exact same locked inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_B = Path(
    "reports/Solar Energy Result/Linear_Formal/"
    "Experiment_B/20260821T041718Z_seed1234"
)
RUN_B2 = Path(
    "reports/Solar Energy Result/Linear_Supplementary/"
    "Experiment_B2/20260822T120239Z_seed1234"
)

WOTL_HISTORY_REL = RUN_B / "target_wotl_candidates/WOTL_lr1e-4/history.csv"
WOTL_PRED_REL = RUN_B / "final_test/predictions.csv"
WOTL_METRICS_REL = RUN_B / "final_test/wotl_metrics_original_scale.json"

B2_HISTORY_REL = RUN_B2 / "candidates/B2-P1/training_history.csv"
B2_PRED_REL = RUN_B2 / "candidates/B2-P1/predictions.csv"
B2_METRICS_REL = RUN_B2 / "candidates/B2-P1/test_metrics_original_scale.json"

OUTPUT_REL = Path(
    "reports/Solar Energy Result/Thesis_Figures/"
    "Experiment_B2_Plant2_to_Plant1"
)

EXPECTED_ROWS = 2607
TARGET_NAME = "DC_POWER"
DOCUMENTED_UNIT = "kW"

FIGURE_FILES = [
    "01_WOTL_learning_curve.png",
    "02_B2_P1_learning_curve.png",
    "03_WOTL_prediction_plot.png",
    "04_B2_P1_prediction_plot.png",
    "05_WOTL_yy_plot.png",
    "06_B2_P1_yy_plot.png",
    "07_WOTL_residual_plot.png",
    "08_B2_P1_residual_plot.png",
    "09_WOTL_error_histogram.png",
    "10_B2_P1_error_histogram.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate locked Solar Experiment B/B2 thesis figures."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow regenerating existing thesis figure outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dict(obj: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                result.update(flatten_dict(value, full_key))
            else:
                result[full_key] = value
    return result


def metric_from_json(obj: Any, metric: str) -> float:
    """
    Finds metric values even if JSON keys differ slightly in casing/nesting.
    Supported aliases include mae/mse/rmse/r2/r²/r2_score.
    """
    aliases = {
        "mae": {"mae", "mean_absolute_error"},
        "mse": {"mse", "mean_squared_error"},
        "rmse": {"rmse", "root_mean_squared_error"},
        "r2": {"r2", "r²", "r2_score", "r_squared", "rsquared"},
    }
    flat = flatten_dict(obj)
    candidates = aliases[metric]

    for key, value in flat.items():
        leaf = key.split(".")[-1].strip().lower()
        leaf = leaf.replace("-", "_").replace(" ", "_")
        if leaf in candidates:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    raise KeyError(
        f"Could not find metric '{metric}' in JSON keys: {sorted(flat.keys())}"
    )


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(np.square(error)))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum(np.square(y_true - y_mean)))
    if ss_tot == 0:
        raise RuntimeError("Cannot calculate R² because y_true has zero variance.")
    r2 = float(1.0 - ss_res / ss_tot)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def assert_metric_match(
    label: str,
    recalculated: dict[str, float],
    official: dict[str, float],
) -> None:
    tolerances = {
        "mae": (1e-10, 1e-6),
        "mse": (1e-10, 1e-3),
        "rmse": (1e-10, 1e-6),
        "r2": (1e-10, 1e-12),
    }

    failures = []
    for key, value in recalculated.items():
        rel_tol, abs_tol = tolerances[key]
        if not math.isclose(
            value,
            official[key],
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            failures.append(
                f"{key}: recalculated={value!r}, official={official[key]!r}"
            )

    if failures:
        raise RuntimeError(
            f"{label} metrics cross-check FAILED:\n  " + "\n  ".join(failures)
        )


def git_info(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return result.stdout.strip()
        except Exception:
            return None

    head = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--short")

    return {
        "head": head,
        "branch": branch,
        "tracked_or_untracked_status": status,
        "git_status_available": status is not None,
        "dirty_or_untracked_present": (
            bool(status) if status is not None else None
        ),
    }


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def safe_output_dir(repo_root: Path, output_dir: Path) -> None:
    run_b = (repo_root / RUN_B).resolve()
    run_b2 = (repo_root / RUN_B2).resolve()
    out = output_dir.resolve()

    if out == run_b or run_b in out.parents:
        raise RuntimeError("Output directory must not be inside Formal Experiment B.")
    if out == run_b2 or run_b2 in out.parents:
        raise RuntimeError("Output directory must not be inside Supplementary Experiment B2.")


def prepare_outputs(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [output_dir / name for name in FIGURE_FILES]
    targets += [
        output_dir / "metrics_recheck.json",
        output_dir / "figure_manifest.json",
    ]
    existing = [p for p in targets if p.exists()]

    if existing and not overwrite:
        formatted = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            "Refusing to overwrite existing thesis outputs.\n"
            "Existing files:\n"
            f"{formatted}\n"
            "Use --overwrite only if intentional."
        )


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_learning_curve(
    history: pd.DataFrame,
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    require_columns(history, ["epoch", "loss", "val_loss"], title)

    epoch = pd.to_numeric(history["epoch"], errors="raise")
    loss = pd.to_numeric(history["loss"], errors="raise")
    val_loss = pd.to_numeric(history["val_loss"], errors="raise")

    best_idx = int(np.nanargmin(val_loss.to_numpy(dtype=float)))
    best_epoch = int(epoch.iloc[best_idx])
    best_val_loss = float(val_loss.iloc[best_idx])

    plt.figure(figsize=(10, 5.5))
    plt.plot(epoch, loss, label="Training loss")
    plt.plot(epoch, val_loss, label="Validation loss")
    plt.scatter([best_epoch], [best_val_loss], label=f"Best val_loss (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (training scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(output_path)

    return {
        "rows": int(len(history)),
        "best_epoch_from_history": best_epoch,
        "best_val_loss_from_history": best_val_loss,
        "final_epoch": int(epoch.iloc[-1]),
        "final_loss": float(loss.iloc[-1]),
        "final_val_loss": float(val_loss.iloc[-1]),
    }


def plot_prediction(
    timestamps: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    ts = pd.to_datetime(timestamps, errors="raise")

    plt.figure(figsize=(12, 5.8))
    plt.plot(ts, actual, label="Actual")
    plt.plot(ts, predicted, label="Predicted")
    plt.xlabel("Timestamp")
    plt.ylabel(f"{TARGET_NAME} ({DOCUMENTED_UNIT})")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.gcf().autofmt_xdate()
    save_figure(output_path)


def plot_yy(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    lo = float(min(np.min(actual), np.min(predicted)))
    hi = float(max(np.max(actual), np.max(predicted)))

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(actual, predicted, s=12, alpha=0.5, label="Test observations")
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="Ideal: y = x")
    plt.xlabel(f"Observed {TARGET_NAME} ({DOCUMENTED_UNIT})")
    plt.ylabel(f"Predicted {TARGET_NAME} ({DOCUMENTED_UNIT})")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(output_path)


def plot_residual(
    predicted: np.ndarray,
    residual: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8.5, 5.8))
    plt.scatter(predicted, residual, s=12, alpha=0.5)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(f"Predicted {TARGET_NAME} ({DOCUMENTED_UNIT})")
    plt.ylabel(f"Residual: actual - predicted ({DOCUMENTED_UNIT})")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    save_figure(output_path)


def plot_histogram(
    residual: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8.5, 5.8))
    plt.hist(residual, bins=50)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel(f"Residual: actual - predicted ({DOCUMENTED_UNIT})")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    save_figure(output_path)


def residual_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual: np.ndarray,
) -> dict[str, Any]:
    negative_mask = y_pred < 0
    zero_actual_mask = np.isclose(y_true, 0.0, rtol=0.0, atol=1e-12)

    def optional_mean(values: np.ndarray) -> float | None:
        return float(np.mean(values)) if len(values) else None

    return {
        "n": int(len(y_true)),
        "negative_prediction_count": int(np.sum(negative_mask)),
        "negative_prediction_ratio": float(np.mean(negative_mask)),
        "actual_zero_count": int(np.sum(zero_actual_mask)),
        "prediction_mean_when_actual_zero": optional_mean(y_pred[zero_actual_mask]),
        "prediction_mae_when_actual_zero": optional_mean(
            np.abs(y_pred[zero_actual_mask])
        ),
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "actual_min": float(np.min(y_true)),
        "actual_max": float(np.max(y_true)),
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    paths = {
        "wotl_history": repo_root / WOTL_HISTORY_REL,
        "wotl_predictions": repo_root / WOTL_PRED_REL,
        "wotl_official_metrics": repo_root / WOTL_METRICS_REL,
        "b2_p1_history": repo_root / B2_HISTORY_REL,
        "b2_p1_predictions": repo_root / B2_PRED_REL,
        "b2_p1_official_metrics": repo_root / B2_METRICS_REL,
    }
    output_dir = repo_root / OUTPUT_REL

    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    safe_output_dir(repo_root, output_dir)
    prepare_outputs(output_dir, args.overwrite)

    # ------------------------------------------------------------------
    # 1) Read-only input loading
    # ------------------------------------------------------------------
    w_hist = pd.read_csv(paths["wotl_history"])
    b_hist = pd.read_csv(paths["b2_p1_history"])
    w = pd.read_csv(paths["wotl_predictions"])
    b = pd.read_csv(paths["b2_p1_predictions"])

    require_columns(
        w,
        [
            "timestamp",
            "y_true_original",
            "wotl_pred_original",
            "wotl_residual",
        ],
        "WOTL predictions",
    )
    require_columns(
        b,
        ["timestamp", "y_true_original", "prediction_original", "residual"],
        "B2-P1 predictions",
    )

    # ------------------------------------------------------------------
    # 2) Test Alignment Gate
    # ------------------------------------------------------------------
    if len(w) != EXPECTED_ROWS or len(b) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Test row-count gate FAILED: WOTL={len(w)}, B2-P1={len(b)}, "
            f"expected={EXPECTED_ROWS}"
        )

    if not w["timestamp"].equals(b["timestamp"]):
        raise RuntimeError("Timestamp alignment gate FAILED.")

    w_true = w["y_true_original"].to_numpy(dtype=float)
    b_true = b["y_true_original"].to_numpy(dtype=float)

    # Exact equality is expected from the locked experiment artifacts.
    if not np.array_equal(w_true, b_true):
        raise RuntimeError("y_true alignment gate FAILED.")

    y_true = w_true
    w_pred = w["wotl_pred_original"].to_numpy(dtype=float)
    b_pred = b["prediction_original"].to_numpy(dtype=float)

    w_residual = w["wotl_residual"].to_numpy(dtype=float)
    b_residual = b["residual"].to_numpy(dtype=float)

    if not np.allclose(
        w_residual,
        y_true - w_pred,
        rtol=1e-12,
        atol=1e-8,
    ):
        raise RuntimeError("WOTL residual sign/value verification FAILED.")

    if not np.allclose(
        b_residual,
        y_true - b_pred,
        rtol=1e-12,
        atol=1e-8,
    ):
        raise RuntimeError("B2-P1 residual sign/value verification FAILED.")

    # ------------------------------------------------------------------
    # 3) Metrics Recalculation Gate
    # ------------------------------------------------------------------
    w_calc = calc_metrics(y_true, w_pred)
    b_calc = calc_metrics(y_true, b_pred)

    w_json = read_json(paths["wotl_official_metrics"])
    b_json = read_json(paths["b2_p1_official_metrics"])

    w_official = {
        key: metric_from_json(w_json, key)
        for key in ["mae", "mse", "rmse", "r2"]
    }
    b_official = {
        key: metric_from_json(b_json, key)
        for key in ["mae", "mse", "rmse", "r2"]
    }

    assert_metric_match("WOTL", w_calc, w_official)
    assert_metric_match("B2-P1", b_calc, b_official)

    # ------------------------------------------------------------------
    # 4) Figures
    # ------------------------------------------------------------------
    w_curve_info = plot_learning_curve(
        w_hist,
        output_dir / "01_WOTL_learning_curve.png",
        "WOTL Learning Curve",
    )
    b_curve_info = plot_learning_curve(
        b_hist,
        output_dir / "02_B2_P1_learning_curve.png",
        "Optimized TL (B2-P1) Learning Curve",
    )

    plot_prediction(
        w["timestamp"],
        y_true,
        w_pred,
        output_dir / "03_WOTL_prediction_plot.png",
        "WOTL Actual vs. Predicted — Plant1 Test",
    )
    plot_prediction(
        b["timestamp"],
        y_true,
        b_pred,
        output_dir / "04_B2_P1_prediction_plot.png",
        "Optimized TL (B2-P1) Actual vs. Predicted — Plant1 Test",
    )

    plot_yy(
        y_true,
        w_pred,
        output_dir / "05_WOTL_yy_plot.png",
        "WOTL Observed vs. Predicted — Plant1 Test",
    )
    plot_yy(
        y_true,
        b_pred,
        output_dir / "06_B2_P1_yy_plot.png",
        "Optimized TL (B2-P1) Observed vs. Predicted — Plant1 Test",
    )

    plot_residual(
        w_pred,
        w_residual,
        output_dir / "07_WOTL_residual_plot.png",
        "WOTL Residual Plot — Plant1 Test",
    )
    plot_residual(
        b_pred,
        b_residual,
        output_dir / "08_B2_P1_residual_plot.png",
        "Optimized TL (B2-P1) Residual Plot — Plant1 Test",
    )

    plot_histogram(
        w_residual,
        output_dir / "09_WOTL_error_histogram.png",
        "WOTL Residual Distribution — Plant1 Test",
    )
    plot_histogram(
        b_residual,
        output_dir / "10_B2_P1_error_histogram.png",
        "Optimized TL (B2-P1) Residual Distribution — Plant1 Test",
    )

    # ------------------------------------------------------------------
    # 5) Verification artifacts
    # ------------------------------------------------------------------
    metrics_recheck = {
        "schema_version": 1,
        "purpose": "Chapter 4 figure-generation integrity verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET_NAME,
        "documented_unit": DOCUMENTED_UNIT,
        "scale": "original",
        "test_alignment_gate": {
            "status": "PASS",
            "expected_rows": EXPECTED_ROWS,
            "wotl_rows": int(len(w)),
            "b2_p1_rows": int(len(b)),
            "timestamp_identical": True,
            "y_true_identical": True,
            "residual_definition": "actual - predicted",
            "wotl_residual_verified": True,
            "b2_p1_residual_verified": True,
        },
        "metrics_recalculation_gate": {
            "status": "PASS",
            "wotl": {
                "recalculated": w_calc,
                "official_json": w_official,
            },
            "b2_p1": {
                "recalculated": b_calc,
                "official_json": b_official,
            },
        },
        "diagnostics": {
            "wotl": residual_summary(y_true, w_pred, w_residual),
            "b2_p1": residual_summary(y_true, b_pred, b_residual),
        },
    }

    with (output_dir / "metrics_recheck.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_recheck, f, ensure_ascii=False, indent=2)

    input_hashes = {
        label: {
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
        }
        for label, path in paths.items()
    }

    figure_manifest = {
        "schema_version": 1,
        "title": "Solar Experiment B/B2 — Thesis Figure Generation v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read-only post-processing",
        "experiment_direction": "Plant2 -> Plant1",
        "baseline": "WOTL_lr1e-4",
        "optimized_tl": "B2-P1",
        "target": TARGET_NAME,
        "documented_unit": DOCUMENTED_UNIT,
        "test_rows": EXPECTED_ROWS,
        "raw_unclipped_predictions": True,
        "training_performed": False,
        "model_loaded": False,
        "model_predict_called": False,
        "checkpoint_reselected": False,
        "prediction_clipping_applied": False,
        "formal_experiment_artifacts_modified": False,
        "git": git_info(repo_root),
        "input_files": input_hashes,
        "learning_curve_summary": {
            "wotl": w_curve_info,
            "b2_p1": b_curve_info,
        },
        "figure_files": FIGURE_FILES,
        "verification_files": [
            "metrics_recheck.json",
            "figure_manifest.json",
        ],
    }

    with (output_dir / "figure_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(figure_manifest, f, ensure_ascii=False, indent=2)

    print("PASS: Test Alignment Gate")
    print("PASS: Metrics Recalculation Gate")
    print("PASS: Residual Verification Gate")
    print(f"Generated 10 figures in: {output_dir}")
    print(f"Generated: {output_dir / 'metrics_recheck.json'}")
    print(f"Generated: {output_dir / 'figure_manifest.json'}")
    print()
    print("WOTL metrics:", w_calc)
    print("B2-P1 metrics:", b_calc)
    print()
    print("NO TRAINING / NO MODEL PREDICT / NO CLIPPING / NO FORMAL ARTIFACT OVERWRITE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
