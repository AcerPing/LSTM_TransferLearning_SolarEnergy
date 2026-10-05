"""Generate the thesis-only Experiment A figure pack from locked formal artifacts.

This script is intentionally isolated from every training and inference code path. It
reads the locked A2 histories and final predictions, verifies their provenance and
schema, then exclusively creates ten PNG figures plus one figure manifest outside
the formal run root.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
FORMAL_RUN_REL = Path(
    "reports/Solar Energy Result/Linear_Formal/Experiment_A2/"
    "20260918T051937Z_seed1234"
)
OUTPUT_REL = Path(
    "reports/Solar Energy Result/Thesis_Figures/"
    "Experiment_A_Plant1_to_Plant2"
)

WOTL_HISTORY_REL = FORMAL_RUN_REL / "target_wotl_candidates/WOTL_lr1e-4/history.csv"
PFT_HISTORY_REL = FORMAL_RUN_REL / "target_partial_ft_candidates/PFT_lr3e-5/history.csv"
PREDICTIONS_REL = FORMAL_RUN_REL / "final/predictions.csv"
FINAL_STATE_REL = FORMAL_RUN_REL / "final/final_state.json"
SELECTION_REL = FORMAL_RUN_REL / "selection/selection.json"
PROTOCOL_REL = FORMAL_RUN_REL / "protocol_manifest.json"

EXPECTED_SHA256 = {
    WOTL_HISTORY_REL: "9c3ee9bcf68b6878e03def9c0f86464df7e4844695a4480dbcb3be0dcb1e1731",
    PFT_HISTORY_REL: "627744e4d578df58e740271452ea027b51d8f03c78d6eb4f4797b5f87a6ef00e",
    PREDICTIONS_REL: "12fce7ec5852b428439a09c4fc3b2e3254a6345ef7948c17077b5e38bf08dcaa",
}

EXPECTED_FIGURES = (
    Path("learning_curve/wotl_learning_curve.png"),
    Path("learning_curve/pft_learning_curve.png"),
    Path("prediction/wotl_prediction_plot_original_scale.png"),
    Path("prediction/pft_prediction_plot_original_scale.png"),
    Path("yy_plot/wotl_yy_plot_original_scale.png"),
    Path("yy_plot/pft_yy_plot_original_scale.png"),
    Path("residual/wotl_residual_plot_original_scale.png"),
    Path("residual/pft_residual_plot_original_scale.png"),
    Path("histogram/wotl_error_histogram_original_scale.png"),
    Path("histogram/pft_error_histogram_original_scale.png"),
)

COLORS = {
    "actual": "#263238",
    "wotl": "#1f5a94",
    "pft": "#c66a1b",
    "validation": "#c66a1b",
    "reference": "#3f4650",
    "grid": "#c8cdd2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def relative_posix(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "absolute_path": str(path),
        "relative_path": relative_posix(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "last_modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
    }


def git_value(repo_root: Path, *args: str) -> str | None:
    command = [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        *args,
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"{label} missing required columns: {missing}")


def validate_finite_numeric(
    frame: pd.DataFrame, columns: list[str], label: str
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"{label}.{column} contains NaN or Inf")


def load_and_validate_history(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["epoch", "loss", "val_loss"]
    require_columns(frame, required, label)
    require(len(frame) == 500, f"{label} row count must be 500, got {len(frame)}")
    validate_finite_numeric(frame, list(frame.columns), label)
    epoch = pd.to_numeric(frame["epoch"], errors="raise").to_numpy(dtype=int)
    require(
        np.array_equal(epoch, np.arange(1, 501, dtype=int)),
        f"{label} epoch sequence must be exactly 1..500",
    )
    return frame


def load_and_validate_predictions(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    required = [
        "timestamp",
        "y_true_original",
        "wotl_pred_original",
        "partial_ft_pred_original",
        "wotl_residual",
        "partial_ft_residual",
        "y_true_normalized",
        "wotl_pred_normalized",
        "partial_ft_pred_normalized",
    ]
    require_columns(frame, required, "predictions")
    require(len(frame) == 2607, f"predictions row count must be 2607, got {len(frame)}")
    require(not frame[required].isna().any().any(), "predictions contains missing values")
    validate_finite_numeric(frame, required[1:], "predictions")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    require(timestamps.is_unique, "prediction timestamps must be unique")
    require(timestamps.is_monotonic_increasing, "prediction timestamps must be ascending")
    require(
        (timestamps.diff().dropna() == pd.Timedelta(minutes=15)).all(),
        "prediction timestamps must have a consistent 15-minute interval",
    )
    return frame, timestamps


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#3f4650",
            "axes.linewidth": 0.9,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, path: Path, title: str) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title, "Creator": "Solar Experiment A thesis-only F1"},
    )
    plt.close(fig)


def plot_learning_curve(
    history: pd.DataFrame,
    output_path: Path,
    candidate_id: str,
    color: str,
) -> tuple[str, str, str]:
    title = (
        "Experiment A | Plant1 → Plant2 | Plant2 Target\n"
        f"{candidate_id} Learning Curve"
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        history["epoch"],
        history["loss"],
        color=color,
        linewidth=1.5,
        label="Training loss",
    )
    ax.plot(
        history["epoch"],
        history["val_loss"],
        color=COLORS["validation"],
        linewidth=1.5,
        linestyle="--",
        label="Validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Compiled Loss (training scale)")
    ax.set_title(title)
    ax.set_xlim(1, 500)
    ax.legend(loc="best")
    finish_axes(ax)
    save_figure(fig, output_path, title)
    return title, "Epoch", "Compiled Loss (training scale)"


def plot_prediction(
    timestamps: pd.Series,
    truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    candidate_id: str,
    color: str,
) -> tuple[str, str, str]:
    title = (
        "Experiment A | Plant1 → Plant2 | Plant2 Target\n"
        f"{candidate_id} Actual vs. Predicted (Original Scale)"
    )
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(timestamps, truth, color=COLORS["actual"], linewidth=1.1, label="Actual")
    ax.plot(timestamps, prediction, color=color, linewidth=1.1, label="Predicted")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("DC_POWER (kW, original scale)")
    ax.set_title(title)
    ax.set_xlim(timestamps.iloc[0], timestamps.iloc[-1])
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc="best")
    finish_axes(ax)
    save_figure(fig, output_path, title)
    return title, "Timestamp", "DC_POWER (kW, original scale)"


def plot_yy(
    truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    candidate_id: str,
    color: str,
) -> tuple[str, str, str]:
    title = (
        "Experiment A | Plant1 → Plant2 | Plant2 Target\n"
        f"{candidate_id} Observed vs. Predicted (Original Scale)"
    )
    lower = float(min(np.min(truth), np.min(prediction)))
    upper = float(max(np.max(truth), np.max(prediction)))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(
        truth,
        prediction,
        s=12,
        alpha=0.45,
        color=color,
        edgecolors="none",
        label="Test observations",
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        color=COLORS["reference"],
        linewidth=1.3,
        linestyle="--",
        label="Ideal: y = x",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed DC_POWER (kW, original scale)")
    ax.set_ylabel("Predicted DC_POWER (kW, original scale)")
    ax.set_title(title)
    ax.legend(loc="best")
    finish_axes(ax)
    save_figure(fig, output_path, title)
    return (
        title,
        "Observed DC_POWER (kW, original scale)",
        "Predicted DC_POWER (kW, original scale)",
    )


def plot_residual(
    prediction: np.ndarray,
    residual: np.ndarray,
    output_path: Path,
    candidate_id: str,
    color: str,
) -> tuple[str, str, str]:
    title = (
        "Experiment A | Plant1 → Plant2 | Plant2 Target\n"
        f"{candidate_id} Residual Plot (Original Scale)"
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.scatter(prediction, residual, s=12, alpha=0.45, color=color, edgecolors="none")
    ax.axhline(0.0, color=COLORS["reference"], linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted DC_POWER (kW, original scale)")
    ax.set_ylabel("Residual = True - Predicted (kW, original scale)")
    ax.set_title(title)
    finish_axes(ax)
    save_figure(fig, output_path, title)
    return (
        title,
        "Predicted DC_POWER (kW, original scale)",
        "Residual = True - Predicted (kW, original scale)",
    )


def plot_histogram(
    residual: np.ndarray,
    output_path: Path,
    candidate_id: str,
    color: str,
) -> tuple[str, str, str]:
    title = (
        "Experiment A | Plant1 → Plant2 | Plant2 Target\n"
        f"{candidate_id} Prediction Error Distribution (Original Scale)"
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.hist(residual, bins=50, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color=COLORS["reference"], linewidth=1.2, linestyle="--")
    ax.set_xlabel("Prediction Error = True - Predicted (kW, original scale)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    finish_axes(ax)
    save_figure(fig, output_path, title)
    return (
        title,
        "Prediction Error = True - Predicted (kW, original scale)",
        "Frequency",
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    formal_root = repo_root / FORMAL_RUN_REL
    output_root = repo_root / OUTPUT_REL
    manifest_path = output_root / "figure_manifest.json"

    require(Path(__file__).resolve() == repo_root / "tools/generate_solar_a_thesis_figures.py", "Unexpected script location")
    require(formal_root.is_dir(), f"Formal run root not found: {formal_root}")
    require(not output_root.is_relative_to(formal_root), "Output root must be outside formal run root")
    if output_root.exists():
        require(not any(output_root.iterdir()), f"Output root is not empty: {output_root}")

    input_paths = {
        "wotl_history": repo_root / WOTL_HISTORY_REL,
        "pft_history": repo_root / PFT_HISTORY_REL,
        "predictions": repo_root / PREDICTIONS_REL,
        "final_state": repo_root / FINAL_STATE_REL,
        "selection": repo_root / SELECTION_REL,
        "protocol_manifest": repo_root / PROTOCOL_REL,
    }
    for label, path in input_paths.items():
        require(path.is_file(), f"Required input missing ({label}): {path}")

    guard_details: dict[str, Any] = {}
    for relative_path, expected_hash in EXPECTED_SHA256.items():
        path = repo_root / relative_path
        actual_hash = sha256_file(path)
        require(
            actual_hash == expected_hash,
            f"SHA-256 mismatch for {relative_path}: {actual_hash}",
        )
        guard_details[relative_path.as_posix()] = {
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "passed": True,
        }

    final_state = read_json(input_paths["final_state"])
    selection = read_json(input_paths["selection"])
    protocol = read_json(input_paths["protocol_manifest"])

    final_state_pass = (
        final_state.get("state") == "TEST_COMPLETED"
        and final_state.get("test_access_count") == 1
        and final_state.get("post_test_tuning_allowed") is False
    )
    selection_pass = (
        selection.get("wotl_selected_candidate") == "WOTL_lr1e-4"
        and selection.get("partial_ft_selected_candidate") == "PFT_lr3e-5"
        and selection.get("selection_locked") is True
    )
    protocol_pass = (
        protocol.get("experiment_id") == "A2"
        and protocol.get("direction") == "Plant1_to_Plant2"
        and protocol.get("source_plant") == "Plant1"
        and protocol.get("target_plant") == "Plant2"
    )
    require(final_state_pass, "Final-state identity guard failed")
    require(selection_pass, "Selection identity guard failed")
    require(protocol_pass, "Protocol identity guard failed")

    wotl_history = load_and_validate_history(input_paths["wotl_history"], "WOTL history")
    pft_history = load_and_validate_history(input_paths["pft_history"], "PFT history")
    predictions, timestamps = load_and_validate_predictions(input_paths["predictions"])

    truth = predictions["y_true_original"].to_numpy(dtype=float)
    wotl_prediction = predictions["wotl_pred_original"].to_numpy(dtype=float)
    pft_prediction = predictions["partial_ft_pred_original"].to_numpy(dtype=float)
    wotl_residual = predictions["wotl_residual"].to_numpy(dtype=float)
    pft_residual = predictions["partial_ft_residual"].to_numpy(dtype=float)

    # Every guard and schema check has passed. Writes begin only here.
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("learning_curve", "prediction", "yy_plot", "residual", "histogram"):
        (output_root / directory).mkdir(exist_ok=False)

    configure_style()
    figures: list[dict[str, Any]] = []

    def register_figure(
        relative_path: Path,
        figure_type: str,
        candidate_id: str,
        source_files: list[Path],
        title: str,
        x_axis: str,
        y_axis: str,
        source_scale: str,
    ) -> None:
        path = output_root / relative_path
        require(path.is_file(), f"Expected figure was not created: {relative_path}")
        stat = path.stat()
        record: dict[str, Any] = {
            "output_path": relative_posix(path, repo_root),
            "figure_type": figure_type,
            "candidate_id": candidate_id,
            "target": "Plant2",
            "source_files": [relative_posix(item, repo_root) for item in source_files],
            "figure_title": title,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "source_scale": source_scale,
            "sha256": sha256_file(path),
            "size_bytes": stat.st_size,
            "created_at_utc": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
        }
        if figure_type in {"residual_plot", "error_histogram"}:
            record["residual_definition"] = "y_true_original - y_pred_original"
        figures.append(record)

    figure_specs = [
        (
            EXPECTED_FIGURES[0],
            "learning_curve",
            "WOTL_lr1e-4",
            [input_paths["wotl_history"]],
            plot_learning_curve(
                wotl_history, output_root / EXPECTED_FIGURES[0], "WOTL_lr1e-4", COLORS["wotl"]
            ),
            "training scale",
        ),
        (
            EXPECTED_FIGURES[1],
            "learning_curve",
            "PFT_lr3e-5",
            [input_paths["pft_history"]],
            plot_learning_curve(
                pft_history, output_root / EXPECTED_FIGURES[1], "PFT_lr3e-5", COLORS["pft"]
            ),
            "training scale",
        ),
        (
            EXPECTED_FIGURES[2],
            "prediction_plot",
            "WOTL_lr1e-4",
            [input_paths["predictions"]],
            plot_prediction(
                timestamps,
                truth,
                wotl_prediction,
                output_root / EXPECTED_FIGURES[2],
                "WOTL_lr1e-4",
                COLORS["wotl"],
            ),
            "original",
        ),
        (
            EXPECTED_FIGURES[3],
            "prediction_plot",
            "PFT_lr3e-5",
            [input_paths["predictions"]],
            plot_prediction(
                timestamps,
                truth,
                pft_prediction,
                output_root / EXPECTED_FIGURES[3],
                "PFT_lr3e-5",
                COLORS["pft"],
            ),
            "original",
        ),
        (
            EXPECTED_FIGURES[4],
            "yy_plot",
            "WOTL_lr1e-4",
            [input_paths["predictions"]],
            plot_yy(
                truth,
                wotl_prediction,
                output_root / EXPECTED_FIGURES[4],
                "WOTL_lr1e-4",
                COLORS["wotl"],
            ),
            "original",
        ),
        (
            EXPECTED_FIGURES[5],
            "yy_plot",
            "PFT_lr3e-5",
            [input_paths["predictions"]],
            plot_yy(
                truth,
                pft_prediction,
                output_root / EXPECTED_FIGURES[5],
                "PFT_lr3e-5",
                COLORS["pft"],
            ),
            "original",
        ),
        (
            EXPECTED_FIGURES[6],
            "residual_plot",
            "WOTL_lr1e-4",
            [input_paths["predictions"]],
            plot_residual(
                wotl_prediction,
                wotl_residual,
                output_root / EXPECTED_FIGURES[6],
                "WOTL_lr1e-4",
                COLORS["wotl"],
            ),
            "original",
        ),
        (
            EXPECTED_FIGURES[7],
            "residual_plot",
            "PFT_lr3e-5",
            [input_paths["predictions"]],
            plot_residual(
                pft_prediction,
                pft_residual,
                output_root / EXPECTED_FIGURES[7],
                "PFT_lr3e-5",
                COLORS["pft"],
            ),
            "original",
        ),
        (
            EXPECTED_FIGURES[8],
            "error_histogram",
            "WOTL_lr1e-4",
            [input_paths["predictions"]],
            plot_histogram(
                wotl_residual,
                output_root / EXPECTED_FIGURES[8],
                "WOTL_lr1e-4",
                COLORS["wotl"],
            ),
            "original",
        ),
        (
            EXPECTED_FIGURES[9],
            "error_histogram",
            "PFT_lr3e-5",
            [input_paths["predictions"]],
            plot_histogram(
                pft_residual,
                output_root / EXPECTED_FIGURES[9],
                "PFT_lr3e-5",
                COLORS["pft"],
            ),
            "original",
        ),
    ]

    for relative_path, figure_type, candidate_id, source_files, labels, scale in figure_specs:
        register_figure(
            relative_path,
            figure_type,
            candidate_id,
            source_files,
            labels[0],
            labels[1],
            labels[2],
            scale,
        )

    require(len(figures) == 10, f"Exactly 10 figures are required, got {len(figures)}")
    require(
        {Path(item["output_path"]).relative_to(OUTPUT_REL) for item in figures}
        == set(EXPECTED_FIGURES),
        "Generated figure inventory does not match the locked ten-file contract",
    )

    source_records = {label: file_record(path, repo_root) for label, path in input_paths.items()}
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_timestamp(),
        "mode": "thesis-only controlled post-processing",
        "identity": {
            "experiment": "A",
            "internal_experiment_id": "A2",
            "direction": "Plant1_to_Plant2",
            "source": "Plant1",
            "target": "Plant2",
            "target_name": "DC_POWER",
            "documented_unit": "kW",
            "run_id": "20260918T051937Z_seed1234",
            "selected_wotl": "WOTL_lr1e-4",
            "selected_pft": "PFT_lr3e-5",
        },
        "guard_verification": {
            "sha256_guards": guard_details,
            "final_state_guard": {
                "passed": final_state_pass,
                "state": final_state.get("state"),
                "test_access_count": final_state.get("test_access_count"),
                "post_test_tuning_allowed": final_state.get("post_test_tuning_allowed"),
            },
            "selection_guard": {
                "passed": selection_pass,
                "selected_wotl": selection.get("wotl_selected_candidate"),
                "selected_pft": selection.get("partial_ft_selected_candidate"),
                "selection_locked": selection.get("selection_locked"),
            },
            "protocol_guard": {
                "passed": protocol_pass,
                "direction": protocol.get("direction"),
                "source": protocol.get("source_plant"),
                "target": protocol.get("target_plant"),
            },
            "exclusive_create_guard": True,
        },
        "input_files": source_records,
        "data_schema": {
            "wotl_history_rows": int(len(wotl_history)),
            "pft_history_rows": int(len(pft_history)),
            "history_columns": list(wotl_history.columns),
            "history_epoch_column": "epoch",
            "history_loss_column": "loss",
            "history_validation_loss_column": "val_loss",
            "predictions_rows": int(len(predictions)),
            "predictions_columns": list(predictions.columns),
            "timestamp_column": "timestamp",
            "y_true_original_column": "y_true_original",
            "wotl_prediction_original_column": "wotl_pred_original",
            "pft_prediction_original_column": "partial_ft_pred_original",
            "wotl_residual_column": "wotl_residual",
            "pft_residual_column": "partial_ft_residual",
            "timestamp_first": str(predictions["timestamp"].iloc[0]),
            "timestamp_last": str(predictions["timestamp"].iloc[-1]),
            "timestamp_interval_minutes": 15,
            "learning_curve_scale": "training scale",
            "prediction_figure_scale": "original",
            "residual_definition": "y_true_original - y_pred_original",
            "residual_x_axis": "predicted original-scale DC_POWER",
        },
        "visual_style": {
            "dpi": 300,
            "font_family": "DejaVu Sans",
            "layout": "tight_layout with bbox_inches=tight",
            "learning_curve_figsize_inches": [10, 5.5],
            "prediction_figsize_inches": [12, 5.8],
            "yy_plot_figsize_inches": [6.5, 6.5],
            "residual_figsize_inches": [8.5, 5.8],
            "histogram_figsize_inches": [8.5, 5.8],
            "histogram_bins": 50,
        },
        "figures": figures,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "git_branch": git_value(repo_root, "branch", "--show-current"),
            "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
            "tracked_dirty_status": git_value(
                repo_root, "status", "--porcelain", "--untracked-files=no"
            ),
            "script_path": relative_posix(Path(__file__).resolve(), repo_root),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "provenance": {
            "purpose": "thesis-only figure generation",
            "training_executed": False,
            "model_fit_executed": False,
            "model_predict_executed": False,
            "model_evaluate_executed": False,
            "target_test_rerun": False,
            "formal_metrics_recomputed": False,
            "formal_metrics_overwritten": False,
            "prediction_clipping_applied": False,
            "predictions_modified": False,
            "formal_run_root_modified": False,
        },
    }

    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    actual_files = {
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file()
    }
    require(
        actual_files == set(EXPECTED_FIGURES) | {Path("figure_manifest.json")},
        f"Unexpected output inventory: {sorted(str(path) for path in actual_files)}",
    )

    print("PASS: all provenance, identity, schema, and exclusive-create guards")
    print("PASS: generated exactly 10 thesis-only figures")
    print(f"PASS: manifest={manifest_path}")
    print("NO TRAINING / NO PREDICT / NO EVALUATE / NO TARGET TEST RERUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
