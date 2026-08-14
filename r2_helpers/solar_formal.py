"""R2.4 Experiment-A formal runner and auditable artifact contracts.

Import this module only after the process bootstrap has fixed the CPU policy
and verified PYTHONHASHSEED.  The public helpers are intentionally testable
without executing a training epoch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import keras
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from r2_config.solar_r2 import (  # noqa: E402
    CORRECTED_R1_ROOT,
    EXPECTED_ROW_COUNTS,
    EXPECTED_SEQUENCE_COUNTS,
    EXPERIMENTS,
    FEATURE_COLUMNS,
    HORIZON,
    TARGET_COLUMN,
    WINDOW,
    profile_path,
)
from r2_helpers.solar_data import (  # noqa: E402
    ProfileMetadata,
    ScalerBundle,
    SequenceData,
    SplitData,
    build_sequences,
    load_profile_metadata,
    load_split,
    validate_checksums,
    validate_feature_transform_consistency,
    validate_manifest,
    validate_required_files,
    validate_scalers,
    validate_target_round_trip,
)
from r2_helpers.solar_runtime import (  # noqa: E402
    EXPECTED_LAYER_TOPOLOGY,
    R2_BATCH_SIZE,
    R2_METHOD_NAMES,
    TRANSFERABLE_LAYER_INDICES,
    NamedSequenceSplit,
    ReloadedCheckpointModel,
    build_r2_model,
    checkpoint_paths,
    configure_reproducibility,
    evaluate_test,
    fit_train_validation,
    make_r2_callbacks,
    reload_best_checkpoint,
    validate_transferred_weights,
)


FORMAL_EXPERIMENT = "A"
FORMAL_DEVICE = "cpu"
FORMAL_SEED = 1234
FORMAL_MAX_EPOCHS = 500
FORMAL_METHODS = (
    "source_pretrain",
    "without_tl",
    "tl_freeze",
    "tl_full_finetune",
)
FORMAL_METHOD_LABELS = {
    "source_pretrain": "Source Pre-training",
    "without_tl": "Without TL",
    "tl_freeze": "TL Freeze",
    "tl_full_finetune": "TL Full Fine-tuning",
}
TRAINING_SPLITS = ("training", "validation")
PREDICTION_COLUMNS = (
    "sample_index",
    "target_row_index",
    "target_timestamp",
    "y_true_normalized",
    "y_pred_normalized",
    "y_true_original",
    "y_pred_original",
    "residual_original",
)
HISTORY_COLUMNS = ("epoch", "loss", "val_loss", "learning_rate")
COMPARISON_COLUMNS = (
    "Method",
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "Best_Epoch",
    "Epochs_Completed",
)
REQUIRED_MANIFEST_SECTIONS = (
    "identity",
    "environment",
    "git_provenance",
    "source_file_sha256",
    "data_contract",
    "scaler_provenance",
    "model_contract",
    "protocol",
    "methods",
    "lifecycle",
    "checkpoints",
    "metrics",
    "artifacts",
    "test_isolation_events",
    "residual_definition",
    "transfer_comparison",
    "status",
    "timestamps",
    "failure",
)
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z_seed\d+$")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CRITICAL_PATHS = (
    "r2_solar.py",
    "r2_config/solar_r2.py",
    "r2_helpers/solar_data.py",
    "r2_helpers/solar_runtime.py",
    "r2_helpers/solar_formal.py",
    "utils/model.py",
)
FORMAL_OUTPUT_BASE = (
    REPOSITORY_ROOT
    / "reports"
    / "Solar Energy Result"
    / "R2"
    / "Experiment_A"
)


class FormalContractError(AssertionError):
    """Raised when the approved R2.4 protocol is violated."""


class FormalRunError(RuntimeError):
    """Raised after preserving an in-progress formal run as FAIL."""

    def __init__(self, message: str, run_root: Path | None = None):
        super().__init__(message)
        self.run_root = run_root


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalContractError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_formal_request(experiment: str, device: str, seed: int) -> None:
    _require(experiment == FORMAL_EXPERIMENT, "R2.4 supports Experiment A only")
    _require(device == FORMAL_DEVICE, "R2.4 is CPU-only")
    _require(seed == FORMAL_SEED, "R2.4 seed must be 1234")
    _require(
        os.environ.get("PYTHONHASHSEED") == str(seed),
        "PYTHONHASHSEED must match the formal seed before process start",
    )


def generate_run_id(seed: int = FORMAL_SEED, now: datetime | None = None) -> str:
    instant = now or datetime.now(timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_seed{seed}"


def validate_run_id(run_id: str, seed: int = FORMAL_SEED) -> None:
    _require(bool(RUN_ID_PATTERN.fullmatch(run_id)), f"Invalid run_id: {run_id}")
    _require(run_id.endswith(f"_seed{seed}"), "run_id seed suffix mismatch")


def validate_output_base(output_base: Path | str) -> Path:
    resolved = Path(output_base).resolve()
    approved = FORMAL_OUTPUT_BASE.resolve()
    _require(
        resolved == approved or _is_within(resolved, approved),
        f"Formal output must remain under {approved}",
    )
    protected = (
        CORRECTED_R1_ROOT.resolve(),
        (REPOSITORY_ROOT / "dataset").resolve(),
        (REPOSITORY_ROOT / "preprocess").resolve(),
    )
    for path in protected:
        _require(
            not _is_within(resolved, path) and not _is_within(path, resolved),
            f"Formal output overlaps protected path: {path}",
        )
    return resolved


def validate_artifact_path(path: Path | str, run_root: Path | str) -> Path:
    resolved = Path(path).resolve()
    root = Path(run_root).resolve()
    _require(_is_within(resolved, root), f"Artifact escaped formal run root: {resolved}")
    return resolved


def create_formal_run_root(
    run_id: str,
    *,
    output_base: Path | str = FORMAL_OUTPUT_BASE,
    seed: int = FORMAL_SEED,
) -> Path:
    validate_run_id(run_id, seed)
    base = validate_output_base(output_base)
    root = validate_artifact_path(base / run_id, base)
    _require(not root.exists(), f"Formal run already exists; overwrite forbidden: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return root


def sha256_file(path: Path | str) -> str:
    file_path = Path(path)
    _require(file_path.is_file(), f"Cannot hash missing file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repo.as_posix()}",
        "-c",
        "core.quotePath=false",
        *args,
    ]
    result = subprocess.run(
        command,
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(result.returncode == 0, f"Git command failed: {' '.join(args)}: {result.stderr.strip()}")
    # Porcelain status uses a leading space as the working-tree status column.
    # Remove line terminators only so that semantic status bytes stay intact.
    return result.stdout.rstrip("\r\n")


def _decode_git_quoted_path(value: str) -> str:
    """Decode Git's C-style quoted pathname without changing Git config."""

    value = value.strip()
    if not (len(value) >= 2 and value[0] == value[-1] == '"'):
        return value
    content = value[1:-1]
    decoded = bytearray()
    escapes = {
        "a": 7,
        "b": 8,
        "t": 9,
        "n": 10,
        "v": 11,
        "f": 12,
        "r": 13,
        '"': 34,
        "\\": 92,
    }
    index = 0
    while index < len(content):
        character = content[index]
        if character != "\\":
            decoded.extend(character.encode("utf-8"))
            index += 1
            continue
        index += 1
        _require(index < len(content), f"Malformed Git quoted path: {value}")
        escaped = content[index]
        if escaped in escapes:
            decoded.append(escapes[escaped])
            index += 1
            continue
        if escaped in "01234567":
            digits = escaped
            index += 1
            while index < len(content) and len(digits) < 3 and content[index] in "01234567":
                digits += content[index]
                index += 1
            decoded.append(int(digits, 8))
            continue
        decoded.extend(escaped.encode("utf-8"))
        index += 1
    return decoded.decode("utf-8")


def _split_git_rename(value: str) -> tuple[str, str] | None:
    quoted = False
    escaped = False
    for index in range(len(value) - 3):
        character = value[index]
        if escaped:
            escaped = False
            continue
        if character == "\\" and quoted:
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if not quoted and value[index : index + 4] == " -> ":
            return value[:index], value[index + 4 :]
    return None


def parse_git_status_paths(status_line: str) -> tuple[str, ...]:
    """Return every path represented by one porcelain-v1 status record."""

    _require(len(status_line) >= 4 and status_line[2] == " ", f"Malformed Git status line: {status_line}")
    status = status_line[:2]
    payload = status_line[3:]
    rename = _split_git_rename(payload) if "R" in status or "C" in status else None
    raw_paths = rename if rename is not None else (payload,)
    return tuple(
        _decode_git_quoted_path(path).replace("\\", "/").removeprefix("./")
        for path in raw_paths
    )


def classify_git_status(status_lines: Sequence[str]) -> dict[str, Any]:
    """Classify repository dirtiness without discarding raw porcelain evidence."""

    lines = list(status_lines)
    critical = set(EXPERIMENT_CRITICAL_PATHS)
    critical_paths: set[str] = set()
    unrelated_paths: set[str] = set()
    for line in lines:
        for path in parse_git_status_paths(line):
            if path in critical:
                critical_paths.add(path)
            else:
                unrelated_paths.add(path)
    return {
        "dirty": bool(lines),
        "status_porcelain": lines,
        "critical_dirty": bool(critical_paths),
        "critical_dirty_paths": sorted(critical_paths),
        "unrelated_dirty": bool(unrelated_paths),
        "unrelated_dirty_paths": sorted(unrelated_paths),
    }


def collect_git_provenance(repo: Path | str = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repo).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    classification = classify_git_status(status.splitlines() if status else [])
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        **classification,
    }


def require_clean_git(provenance: Mapping[str, Any]) -> None:
    """Block only uncommitted experiment-critical files."""

    paths = list(provenance.get("critical_dirty_paths", ()))
    _require(
        not provenance.get("critical_dirty"),
        f"Formal execution requires clean experiment-critical files: {paths}",
    )


def collect_source_hashes() -> dict[str, str]:
    paths = tuple(REPOSITORY_ROOT / path for path in EXPERIMENT_CRITICAL_PATHS)
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


@dataclass(frozen=True)
class FormalSequenceSplit:
    named: NamedSequenceSplit
    split_data: SplitData
    sequences: SequenceData


@dataclass(frozen=True)
class FormalRoleData:
    role: str
    metadata: ProfileMetadata
    scalers: ScalerBundle
    checksums: Mapping[str, str]
    training: FormalSequenceSplit
    validation: FormalSequenceSplit


@dataclass(frozen=True)
class LifecycleResult:
    method: str
    model: Any
    method_dir: Path
    checkpoint_path: Path
    history_frame: pd.DataFrame
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    metrics_normalized: Mapping[str, float | int]
    metrics_original: Mapping[str, float | int]
    prediction_frame: pd.DataFrame
    artifacts: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class FormalRunResult:
    run_root: Path
    manifest_path: Path


def _validate_profile(role: str) -> tuple[ProfileMetadata, ScalerBundle, Mapping[str, str]]:
    _require(role in ("source", "target"), f"Unknown formal role: {role}")
    mapping = EXPERIMENTS[FORMAL_EXPERIMENT][role]
    directory = profile_path(FORMAL_EXPERIMENT, role)
    validate_required_files(directory)
    checksums = validate_checksums(directory)
    metadata = load_profile_metadata(directory)
    validate_manifest(
        metadata,
        expected_plant=mapping["manifest_plant"],
        expected_profile=mapping["manifest_profile"],
        expected_profile_dir_name=mapping["profile_dir"],
    )
    scalers = validate_scalers(metadata)
    return metadata, scalers, checksums


def _build_formal_split(
    metadata: ProfileMetadata,
    scalers: ScalerBundle,
    split_name: str,
) -> FormalSequenceSplit:
    split = load_split(metadata, split_name)
    validate_feature_transform_consistency(split, scalers.feature_scaler)
    validate_target_round_trip(split, scalers.target_scaler)
    sequences = build_sequences(
        split.X_normalized,
        split.y_normalized,
        split.timestamps,
    )
    named = NamedSequenceSplit(split_name, sequences.X_seq, sequences.y_seq)
    return FormalSequenceSplit(named=named, split_data=split, sequences=sequences)


def validate_sequence_contract(role: str, split: FormalSequenceSplit) -> None:
    profile_name = EXPERIMENTS[FORMAL_EXPERIMENT][role]["profile_dir"]
    split_name = split.named.name
    expected_rows = EXPECTED_ROW_COUNTS[profile_name][split_name]
    expected_sequences = EXPECTED_SEQUENCE_COUNTS[profile_name][split_name]
    _require(split.split_data.row_count == expected_rows, f"{role}/{split_name} row count mismatch")
    _require(split.sequences.count == expected_sequences, f"{role}/{split_name} sequence count mismatch")
    _require(
        split.sequences.X_seq.shape == (expected_sequences, WINDOW, len(FEATURE_COLUMNS)),
        f"{role}/{split_name} X shape mismatch",
    )
    _require(split.sequences.y_seq.shape == (expected_sequences,), f"{role}/{split_name} y shape mismatch")
    _require(int(split.sequences.target_row_index[0]) == WINDOW + HORIZON - 1, "First target index mismatch")
    _require(int(split.sequences.target_row_index[-1]) == expected_rows - 1, "Last target index mismatch")


def prepare_role_training_validation(role: str) -> FormalRoleData:
    """Load only training and validation values; test values remain unparsed."""

    metadata, scalers, checksums = _validate_profile(role)
    loaded = {
        split_name: _build_formal_split(metadata, scalers, split_name)
        for split_name in TRAINING_SPLITS
    }
    for split in loaded.values():
        validate_sequence_contract(role, split)
    return FormalRoleData(
        role=role,
        metadata=metadata,
        scalers=scalers,
        checksums=checksums,
        training=loaded["training"],
        validation=loaded["validation"],
    )


def load_formal_test(role_data: FormalRoleData) -> FormalSequenceSplit:
    """Parse test values only after all model checkpoints have been fixed."""

    test = _build_formal_split(role_data.metadata, role_data.scalers, "test")
    validate_sequence_contract(role_data.role, test)
    return test


class LearningRateObserver(Callback):
    """Record the LR actually used by each epoch without mutating optimizer state."""

    def __init__(self) -> None:
        super().__init__()
        self.learning_rates: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        learning_rate = getattr(self.model.optimizer, "learning_rate", None)
        if learning_rate is None:
            learning_rate = self.model.optimizer.lr
        self.learning_rates.append(float(K.get_value(learning_rate)))


def build_history_frame(history: Any, learning_rates: Sequence[float]) -> pd.DataFrame:
    loss = np.asarray(history.history.get("loss", ()), dtype=np.float64)
    val_loss = np.asarray(history.history.get("val_loss", ()), dtype=np.float64)
    lr = np.asarray(learning_rates, dtype=np.float64)
    _require(len(loss) > 0, "Training history is empty")
    _require(len(loss) == len(val_loss) == len(lr), "History columns have unequal lengths")
    _require(np.isfinite(loss).all(), "Non-finite training loss")
    _require(np.isfinite(val_loss).all(), "Non-finite validation loss")
    _require(np.isfinite(lr).all(), "Non-finite learning rate")
    return pd.DataFrame(
        {
            "epoch": np.arange(1, len(loss) + 1, dtype=np.int64),
            "loss": loss,
            "val_loss": val_loss,
            "learning_rate": lr,
        },
        columns=HISTORY_COLUMNS,
    )


def summarize_history(
    history_frame: pd.DataFrame,
    early_stopping: Any,
    *,
    max_epochs: int = FORMAL_MAX_EPOCHS,
) -> dict[str, Any]:
    _require(tuple(history_frame.columns) == HISTORY_COLUMNS, "History schema mismatch")
    values = history_frame["val_loss"].to_numpy(dtype=np.float64)
    best_index = int(np.argmin(values))
    epochs_completed = len(history_frame)
    stopped_epoch = int(getattr(early_stopping, "stopped_epoch", 0))
    if stopped_epoch > 0 or epochs_completed < max_epochs:
        stop_reason = "early_stopping"
    else:
        stop_reason = "max_epochs"
    return {
        "best_epoch": best_index + 1,
        "best_val_loss": float(values[best_index]),
        "epochs_completed": epochs_completed,
        "stopped_epoch": stopped_epoch + 1 if stopped_epoch > 0 else None,
        "stop_reason": stop_reason,
    }


def _write_history_and_plot(frame: pd.DataFrame, method_dir: Path) -> dict[str, Mapping[str, Any]]:
    history_path = validate_artifact_path(method_dir / "history.csv", method_dir)
    frame.to_csv(history_path, index=False)
    plot_path = validate_artifact_path(method_dir / "learning_curve.png", method_dir)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(frame["epoch"], frame["loss"], label="Training loss")
    axis.plot(frame["epoch"], frame["val_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("Training and validation loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return artifact_records((history_path, plot_path), method_dir)


def train_lifecycle(
    *,
    method: str,
    model: Any,
    role_data: FormalRoleData,
    method_dir: Path,
    max_epochs: int = FORMAL_MAX_EPOCHS,
) -> LifecycleResult:
    _require(method in FORMAL_METHODS, f"Unknown formal method: {method}")
    _require(max_epochs == FORMAL_MAX_EPOCHS, "Formal max_epochs must remain 500")
    _require(not method_dir.exists(), f"Method output already exists: {method_dir}")
    method_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = method_dir / "best_model.hdf5"
    callbacks = list(make_r2_callbacks(checkpoint_path))
    observer = LearningRateObserver()
    callbacks.append(observer)
    history = fit_train_validation(
        model,
        training=role_data.training.named,
        validation=role_data.validation.named,
        callbacks=callbacks,
        epochs=max_epochs,
        verbose=2,
    )
    history_frame = build_history_frame(history, observer.learning_rates)
    summary = summarize_history(history_frame, callbacks[2], max_epochs=max_epochs)
    _require(checkpoint_path.is_file(), f"Best checkpoint not produced: {checkpoint_path}")
    _require(checkpoint_path.stat().st_size > 0, f"Best checkpoint is empty: {checkpoint_path}")
    artifacts = _write_history_and_plot(history_frame, method_dir)
    summary.update(
        {
            "status": "PASS",
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_size": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "history_artifacts": artifacts,
        }
    )
    return LifecycleResult(
        method=method,
        model=model,
        method_dir=method_dir,
        checkpoint_path=checkpoint_path,
        history_frame=history_frame,
        summary=summary,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    _require(len(true) == len(pred) and len(true) > 0, "Metric input count mismatch")
    _require(np.isfinite(true).all() and np.isfinite(pred).all(), "Non-finite metric input")
    mse = float(mean_squared_error(true, pred))
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(true, pred)),
        "n_samples": len(true),
    }


def build_prediction_frame(
    test: FormalSequenceSplit,
    y_pred_normalized: np.ndarray,
    target_scaler: Any,
) -> pd.DataFrame:
    pred_normalized = np.asarray(y_pred_normalized, dtype=np.float64).reshape(-1)
    true_normalized = np.asarray(test.sequences.y_seq, dtype=np.float64).reshape(-1)
    _require(len(pred_normalized) == test.sequences.count, "Prediction count mismatch")
    _require(np.isfinite(pred_normalized).all(), "Non-finite normalized prediction")
    _require(int(target_scaler.n_features_in_) == 1, "Target scaler dimension mismatch")
    pred_original = target_scaler.inverse_transform(pred_normalized.reshape(-1, 1)).reshape(-1)
    inverted_true = target_scaler.inverse_transform(true_normalized.reshape(-1, 1)).reshape(-1)
    true_original = test.split_data.y_original[test.sequences.target_row_index]
    _require(
        np.allclose(inverted_true, true_original, rtol=1e-10, atol=1e-6),
        "Normalized/original test target mismatch",
    )
    _require(np.isfinite(pred_original).all(), "Non-finite original prediction")
    frame = pd.DataFrame(
        {
            "sample_index": test.sequences.sample_index,
            "target_row_index": test.sequences.target_row_index,
            "target_timestamp": test.sequences.target_timestamp,
            "y_true_normalized": true_normalized,
            "y_pred_normalized": pred_normalized,
            "y_true_original": true_original,
            "y_pred_original": pred_original,
            "residual_original": true_original - pred_original,
        },
        columns=PREDICTION_COLUMNS,
    )
    _require(tuple(frame.columns) == PREDICTION_COLUMNS, "Prediction CSV schema mismatch")
    return frame


def _write_metrics(metrics: Mapping[str, float | int], scale: str, method_dir: Path) -> tuple[Path, Path]:
    json_path = method_dir / f"metrics_{scale}.json"
    csv_path = method_dir / f"metrics_{scale}.csv"
    json_path.write_text(json.dumps(dict(metrics), indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    return json_path, csv_path


def _write_prediction_plot(frame: pd.DataFrame, method: str, method_dir: Path) -> Path:
    path = method_dir / "prediction_test.png"
    timestamps = pd.to_datetime(frame["target_timestamp"])
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(timestamps, frame["y_true_original"], label="Actual", linewidth=1)
    axis.plot(timestamps, frame["y_pred_original"], label="Predicted", linewidth=1)
    axis.set_xlabel("Target timestamp")
    axis.set_ylabel(TARGET_COLUMN)
    axis.set_title(f"Experiment A — {method} — ordered test prediction")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def artifact_records(paths: Sequence[Path], root: Path) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        validate_artifact_path(path, root)
        _require(path.is_file(), f"Artifact not found: {path}")
        records[path.name] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def evaluate_lifecycle(
    *,
    method: str,
    checkpoint_model: ReloadedCheckpointModel,
    test: FormalSequenceSplit,
    scalers: ScalerBundle,
    method_dir: Path,
) -> EvaluationResult:
    evaluation = evaluate_test(checkpoint_model, test=test.named, verbose=0)
    frame = build_prediction_frame(test, evaluation.predictions, scalers.target_scaler)
    normalized = compute_metrics(frame["y_true_normalized"], frame["y_pred_normalized"])
    original = compute_metrics(frame["y_true_original"], frame["y_pred_original"])
    prediction_path = method_dir / "predictions_test.csv"
    frame.to_csv(prediction_path, index=False)
    normalized_paths = _write_metrics(normalized, "normalized", method_dir)
    original_paths = _write_metrics(original, "original", method_dir)
    plot_path = _write_prediction_plot(frame, method, method_dir)
    artifacts = artifact_records(
        (prediction_path, *normalized_paths, *original_paths, plot_path),
        method_dir,
    )
    return EvaluationResult(
        method=method,
        metrics_normalized=normalized,
        metrics_original=original,
        prediction_frame=frame,
        artifacts=artifacts,
    )


def transfer_interpretation(
    baseline_rmse: float,
    method_rmse: float,
) -> dict[str, float | str]:
    _require(baseline_rmse > 0, "Without-TL RMSE must be positive")
    delta = float(method_rmse - baseline_rmse)
    tolerance = 1e-12 * max(1.0, abs(baseline_rmse))
    if delta < -tolerance:
        label = "observed_positive"
    elif delta > tolerance:
        label = "observed_negative"
    else:
        label = "neutral"
    return {
        "delta_rmse": delta,
        "improvement_percent": float(100.0 * (baseline_rmse - method_rmse) / baseline_rmse),
        "label": label,
        "criterion": "original_scale_RMSE",
        "statistical_significance_claimed": False,
    }


def build_target_comparison(
    evaluations: Mapping[str, EvaluationResult],
    lifecycles: Mapping[str, LifecycleResult],
) -> pd.DataFrame:
    rows = []
    for method in R2_METHOD_NAMES:
        metrics = evaluations[method].metrics_original
        summary = lifecycles[method].summary
        rows.append(
            {
                "Method": method,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Best_Epoch": summary["best_epoch"],
                "Epochs_Completed": summary["epochs_completed"],
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def validate_manifest_schema(manifest: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_MANIFEST_SECTIONS if name not in manifest]
    _require(not missing, f"Manifest missing sections: {missing}")
    _require(manifest["residual_definition"] == "y_true_original - y_pred_original", "Residual definition mismatch")
    _require(manifest["status"] in ("RUNNING", "PASS", "FAIL"), "Invalid manifest status")


def write_manifest_atomic(path: Path, manifest: Mapping[str, Any]) -> None:
    validate_artifact_path(path, path.parent)
    validate_manifest_schema(manifest)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def mark_manifest_failed(
    manifest: dict[str, Any],
    *,
    failed_stage: str,
    exc: BaseException,
    message: str | None = None,
) -> None:
    failure_type = type(exc).__name__
    manifest["status"] = "FAIL"
    manifest["failure"] = {
        "failed_stage": failed_stage,
        "type": failure_type,
        "exception_type": failure_type,
        "message": str(exc) if message is None else message,
        "traceback": traceback.format_exc(),
    }
    manifest["timestamps"]["end"] = _utc_now()


def _persist_failed_run(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    failed_stage: str,
    exc: BaseException,
    started: float,
    message: str | None = None,
) -> None:
    """Preserve available run evidence without deleting or synthesizing artifacts."""

    mark_manifest_failed(
        manifest,
        failed_stage=failed_stage,
        exc=exc,
        message=message,
    )
    manifest["timestamps"]["elapsed_seconds"] = round(
        time.perf_counter() - started,
        6,
    )
    write_manifest_atomic(manifest_path, manifest)
    (run_root / "failure_traceback.txt").write_text(
        traceback.format_exc(),
        encoding="utf-8",
    )


def _read_result_manifest(result: FormalRunResult) -> dict[str, Any]:
    _require(result.manifest_path.is_file(), f"Formal manifest not found: {result.manifest_path}")
    return json.loads(result.manifest_path.read_text(encoding="utf-8"))


def format_formal_success_summary(result: FormalRunResult) -> str:
    """Format a PASS summary solely from finalized formal artifacts."""

    manifest = _read_result_manifest(result)
    _require(manifest.get("status") == "PASS", "Formal PASS summary requires a PASS manifest")
    comparison_path = result.run_root / "target_comparison.csv"
    _require(comparison_path.is_file(), f"Target comparison not found: {comparison_path}")
    identity = manifest["identity"]
    lines = [
        "=" * 60,
        "R2 Formal Experiment A Completed",
        "=" * 60,
        "",
        "Run ID:",
        str(identity["run_id"]),
        "",
        "Source:",
        str(identity["source"]).split("/", 1)[0],
        "",
        "Target:",
        str(identity["target"]).split("/", 1)[0],
        "",
        "Status:",
        "PASS",
        "",
        "Target Test Metrics - Original Scale",
        "",
        f"{'Method':<24}{'MAE':>12}{'MSE':>14}{'RMSE':>12}{'R²':>12}",
        "-" * 74,
    ]
    for method in R2_METHOD_NAMES:
        metrics = manifest["metrics"][method]["original"]
        lines.append(
            f"{FORMAL_METHOD_LABELS[method]:<24}"
            f"{float(metrics['MAE']):>12.6f}"
            f"{float(metrics['MSE']):>14.6f}"
            f"{float(metrics['RMSE']):>12.6f}"
            f"{float(metrics['R2']):>12.6f}"
        )
    lines.extend(
        [
            "",
            "Transfer Comparison",
            "Primary criterion: Original-scale RMSE",
        ]
    )
    for method in ("tl_freeze", "tl_full_finetune"):
        comparison = manifest["transfer_comparison"][method]
        lines.extend(
            [
                "",
                f"{FORMAL_METHOD_LABELS[method]}:",
                f"delta_rmse = {float(comparison['delta_rmse']):.6f}",
                f"improvement_percent = {float(comparison['improvement_percent']):.6f}",
                f"label = {comparison['label']}",
            ]
        )
    lines.extend(
        [
            "",
            "Training Summary",
            "",
            f"{'Method':<24}{'Best Epoch':>16}{'Epochs Completed':>20}",
            "-" * 60,
        ]
    )
    for method in FORMAL_METHODS:
        lifecycle = manifest["lifecycle"][method]
        lines.append(
            f"{FORMAL_METHOD_LABELS[method]:<24}"
            f"{int(lifecycle['best_epoch']):>16d}"
            f"{int(lifecycle['epochs_completed']):>20d}"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "Run manifest:",
            str(result.manifest_path),
            "",
            "Target comparison:",
            str(comparison_path),
            "",
            "Formal output:",
            str(result.run_root),
            "",
            "=" * 60,
            "Formal Experiment A: PASS",
            "=" * 60,
        ]
    )
    return "\n".join(lines)


def format_formal_interrupted_summary(
    run_root: Path | None,
    *,
    manifest_path: Path | None = None,
    failed_stage: str | None = None,
) -> str:
    """Format interruption evidence without invoking data or model operations."""

    manifest: Mapping[str, Any] = {}
    candidate = manifest_path or (run_root / "run_manifest.json" if run_root else None)
    if candidate is not None and candidate.is_file():
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
    run_id = manifest.get("identity", {}).get(
        "run_id",
        run_root.name if run_root is not None else "not-created",
    )
    stage = manifest.get("failure", {}).get("failed_stage", failed_stage or "preflight")
    return "\n".join(
        [
            "=" * 60,
            "R2 Formal Experiment A INTERRUPTED",
            "=" * 60,
            "",
            "Run ID:",
            str(run_id),
            "",
            "Failed Stage:",
            str(stage),
            "",
            "Manifest:",
            str(candidate) if candidate is not None else "not-created",
            "",
            "Preserved Output:",
            str(run_root) if run_root is not None else "not-created",
            "",
            "=" * 60,
        ]
    )


def validate_test_isolation_events(events: Sequence[str]) -> None:
    training_methods = FORMAL_METHODS
    for method in training_methods:
        _require(f"fit_end:{method}" in events, f"Missing fit completion: {method}")
    first_test_event = min(
        (events.index(event) for event in events if event.startswith("load_test:")),
        default=len(events),
    )
    last_fit_event = max(events.index(f"fit_end:{method}") for method in training_methods)
    _require(first_test_event > last_fit_event, "Test values loaded before all training completed")
    for method in FORMAL_METHODS:
        reload_event = f"reload:{method}"
        load_event = "load_test:source" if method == "source_pretrain" else "load_test:target"
        _require(reload_event in events and load_event in events, f"Missing reload/test event for {method}")
        _require(events.index(reload_event) < events.index(load_event), f"Test loaded before reload for {method}")


def audit_git_writes(
    status_lines: Sequence[str],
    run_root: Path,
    repo_root: Path = REPOSITORY_ROOT,
    *,
    baseline_status_lines: Sequence[str] = (),
) -> None:
    root = run_root.resolve()
    repo = repo_root.resolve()
    baseline = set(baseline_status_lines)
    for line in status_lines:
        if line in baseline:
            continue
        for value in parse_git_status_paths(line):
            candidate = (repo / value.replace("/", os.sep)).resolve()
            _require(_is_within(candidate, root), f"Unexpected repository write outside formal root: {candidate}")


def _sequence_evidence(split: FormalSequenceSplit) -> dict[str, Any]:
    seq = split.sequences
    return {
        "rows": split.split_data.row_count,
        "sequences": seq.count,
        "X_shape": list(seq.X_seq.shape),
        "y_shape": list(seq.y_seq.shape),
        "first_target_row_index": int(seq.target_row_index[0]),
        "last_target_row_index": int(seq.target_row_index[-1]),
        "first_target_timestamp": str(seq.target_timestamp[0]),
        "last_target_timestamp": str(seq.target_timestamp[-1]),
    }


def _initial_manifest(
    run_id: str,
    seed: int,
    git_provenance: Mapping[str, Any],
    source: FormalRoleData,
    target: FormalRoleData,
) -> dict[str, Any]:
    return {
        "identity": {
            "run_id": run_id,
            "run_type": "formal",
            "formal_result": True,
            "experiment": "A",
            "source": "Plant1/source_profile",
            "target": "Plant2/target_profile",
        },
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "seed": seed,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "deterministic_ops": True,
            "training_shuffle": True,
            "validation_ordered": True,
            "test_ordered": True,
        },
        "git_provenance": dict(git_provenance),
        "source_file_sha256": collect_source_hashes(),
        "data_contract": {
            "corrected_r1_root": str(CORRECTED_R1_ROOT),
            "window": WINDOW,
            "horizon": HORIZON,
            "features": list(FEATURE_COLUMNS),
            "target": TARGET_COLUMN,
            "integrity_hash_access_may_read_test_bytes": True,
            "test_values_loaded_during_training": False,
            "source": {
                "training": _sequence_evidence(source.training),
                "validation": _sequence_evidence(source.validation),
            },
            "target": {
                "training": _sequence_evidence(target.training),
                "validation": _sequence_evidence(target.validation),
            },
            "file_checksums": {
                "source": dict(source.checksums),
                "target": dict(target.checksums),
            },
        },
        "scaler_provenance": {
            "source": dict(source.metadata.scaler_manifest),
            "target": dict(target.metadata.scaler_manifest),
        },
        "model_contract": {
            "input_shape": [WINDOW, len(FEATURE_COLUMNS)],
            "topology": list(EXPECTED_LAYER_TOPOLOGY),
            "output_activation": "sigmoid",
            "transferable_layer_indices": list(TRANSFERABLE_LAYER_INDICES),
        },
        "protocol": {
            "max_epochs": FORMAL_MAX_EPOCHS,
            "batch_size": R2_BATCH_SIZE,
            "optimizer": "Adam",
            "source_without_tl_learning_rate": 1e-4,
            "tl_learning_rate": 1e-5,
            "loss": "mse",
            "reduce_lr": {"monitor": "val_loss", "factor": 0.5, "patience": 4, "min_lr": 1e-7},
            "early_stopping": {"monitor": "val_loss", "patience": 10, "restore_best_weights": True},
            "model_checkpoint": {"monitor": "val_loss", "save_best_only": True, "save_weights_only": False},
        },
        "methods": list(FORMAL_METHODS),
        "lifecycle": {method: {"status": "PENDING"} for method in FORMAL_METHODS},
        "checkpoints": {},
        "metrics": {},
        "artifacts": {},
        "test_isolation_events": [],
        "residual_definition": "y_true_original - y_pred_original",
        "transfer_comparison": {},
        "status": "RUNNING",
        "timestamps": {"start": _utc_now(), "end": None, "elapsed_seconds": None},
        "failure": None,
    }


def _record_event(manifest: dict[str, Any], event: str) -> None:
    manifest["test_isolation_events"].append(event)


def run_formal(
    *,
    experiment: str,
    device: str,
    seed: int,
    run_id: str | None = None,
) -> FormalRunResult:
    """Execute the approved formal protocol.  Never call from contract tests."""

    run_root: Path | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    stage = "preflight"
    started = time.perf_counter()
    try:
        validate_formal_request(experiment, device, seed)
        _require(not tf.config.list_physical_devices("GPU"), "GPU visible in CPU-only formal process")
        provenance = collect_git_provenance()
        require_clean_git(provenance)
        configure_reproducibility(seed)
        source = prepare_role_training_validation("source")
        target = prepare_role_training_validation("target")
        actual_run_id = run_id or generate_run_id(seed)
        run_root = create_formal_run_root(actual_run_id)
        manifest = _initial_manifest(actual_run_id, seed, provenance, source, target)
        manifest_path = run_root / "run_manifest.json"
        write_manifest_atomic(manifest_path, manifest)

        paths = checkpoint_paths(run_root)
        lifecycle: dict[str, LifecycleResult] = {}

        stage = "source_pretrain"
        configure_reproducibility(seed)
        source_model = build_r2_model(output_dir=run_root)
        _record_event(manifest, "fit_start:source_pretrain")
        lifecycle["source_pretrain"] = train_lifecycle(
            method="source_pretrain",
            model=source_model,
            role_data=source,
            method_dir=paths.source_pretrain.parent,
        )
        _record_event(manifest, "fit_end:source_pretrain")
        manifest["lifecycle"]["source_pretrain"] = dict(lifecycle["source_pretrain"].summary)
        write_manifest_atomic(manifest_path, manifest)

        stage = "without_tl"
        configure_reproducibility(seed)
        without_model = build_r2_model(output_dir=run_root, pretrained_model=None)
        _record_event(manifest, "fit_start:without_tl")
        lifecycle["without_tl"] = train_lifecycle(
            method="without_tl",
            model=without_model,
            role_data=target,
            method_dir=paths.without_tl.parent,
        )
        _record_event(manifest, "fit_end:without_tl")
        manifest["lifecycle"]["without_tl"] = dict(lifecycle["without_tl"].summary)
        write_manifest_atomic(manifest_path, manifest)

        stage = "tl_freeze"
        source_best = reload_best_checkpoint(paths.source_pretrain)
        configure_reproducibility(seed)
        freeze_initial = build_r2_model(output_dir=run_root)
        configure_reproducibility(seed)
        freeze_model = build_r2_model(
            output_dir=run_root,
            pretrained_model=source_best.model,
            freeze=True,
        )
        validate_transferred_weights(source_best.model, freeze_model, freeze_initial, freeze=True)
        frozen_before = {
            index: [weight.copy() for weight in freeze_model.layers[index].get_weights()]
            for index in TRANSFERABLE_LAYER_INDICES
        }
        _record_event(manifest, "fit_start:tl_freeze")
        lifecycle["tl_freeze"] = train_lifecycle(
            method="tl_freeze",
            model=freeze_model,
            role_data=target,
            method_dir=paths.tl_freeze.parent,
        )
        _record_event(manifest, "fit_end:tl_freeze")
        for index in TRANSFERABLE_LAYER_INDICES:
            _require(
                all(
                    np.array_equal(before, after)
                    for before, after in zip(frozen_before[index], freeze_model.layers[index].get_weights())
                ),
                f"Frozen layer {index} changed during formal training",
            )
        manifest["lifecycle"]["tl_freeze"] = dict(lifecycle["tl_freeze"].summary)
        manifest["lifecycle"]["tl_freeze"]["transferable_weights_unchanged"] = True
        write_manifest_atomic(manifest_path, manifest)

        stage = "tl_full_finetune"
        source_best = reload_best_checkpoint(paths.source_pretrain)
        configure_reproducibility(seed)
        full_initial = build_r2_model(output_dir=run_root)
        configure_reproducibility(seed)
        full_model = build_r2_model(
            output_dir=run_root,
            pretrained_model=source_best.model,
            freeze=False,
        )
        validate_transferred_weights(source_best.model, full_model, full_initial, freeze=False)
        _record_event(manifest, "fit_start:tl_full_finetune")
        lifecycle["tl_full_finetune"] = train_lifecycle(
            method="tl_full_finetune",
            model=full_model,
            role_data=target,
            method_dir=paths.tl_full_finetune.parent,
        )
        _record_event(manifest, "fit_end:tl_full_finetune")
        manifest["lifecycle"]["tl_full_finetune"] = dict(lifecycle["tl_full_finetune"].summary)
        write_manifest_atomic(manifest_path, manifest)

        stage = "reload_all_checkpoints"
        reloaded = {
            "source_pretrain": reload_best_checkpoint(paths.source_pretrain),
            "without_tl": reload_best_checkpoint(paths.without_tl),
            "tl_freeze": reload_best_checkpoint(paths.tl_freeze),
            "tl_full_finetune": reload_best_checkpoint(paths.tl_full_finetune),
        }
        for method in FORMAL_METHODS:
            _record_event(manifest, f"reload:{method}")

        stage = "load_test_values"
        source_test = load_formal_test(source)
        _record_event(manifest, "load_test:source")
        target_test = load_formal_test(target)
        _record_event(manifest, "load_test:target")
        manifest["data_contract"]["source"]["test"] = _sequence_evidence(source_test)
        manifest["data_contract"]["target"]["test"] = _sequence_evidence(target_test)

        stage = "formal_evaluation"
        evaluations: dict[str, EvaluationResult] = {}
        evaluations["source_pretrain"] = evaluate_lifecycle(
            method="source_pretrain",
            checkpoint_model=reloaded["source_pretrain"],
            test=source_test,
            scalers=source.scalers,
            method_dir=lifecycle["source_pretrain"].method_dir,
        )
        _record_event(manifest, "evaluate_test:source_pretrain")
        for method in R2_METHOD_NAMES:
            evaluations[method] = evaluate_lifecycle(
                method=method,
                checkpoint_model=reloaded[method],
                test=target_test,
                scalers=target.scalers,
                method_dir=lifecycle[method].method_dir,
            )
            _record_event(manifest, f"evaluate_test:{method}")

        validate_test_isolation_events(manifest["test_isolation_events"])
        manifest["metrics"] = {
            method: {
                "normalized": dict(result.metrics_normalized),
                "original": dict(result.metrics_original),
            }
            for method, result in evaluations.items()
        }
        manifest["artifacts"] = {
            method: {
                **dict(lifecycle[method].summary["history_artifacts"]),
                **dict(result.artifacts),
            }
            for method, result in evaluations.items()
        }
        for method, result in evaluations.items():
            manifest["lifecycle"][method].update(
                {
                    "checkpoint_reloaded": True,
                    "test_evaluated_after_reload": True,
                    "prediction_count": len(result.prediction_frame),
                    "predictions_finite": bool(
                        np.isfinite(
                            result.prediction_frame["y_pred_normalized"].to_numpy()
                        ).all()
                        and np.isfinite(
                            result.prediction_frame["y_pred_original"].to_numpy()
                        ).all()
                    ),
                }
            )
        manifest["checkpoints"] = {
            method: {
                "path": result.checkpoint_path.relative_to(run_root).as_posix(),
                "size_bytes": result.checkpoint_path.stat().st_size,
                "sha256": sha256_file(result.checkpoint_path),
            }
            for method, result in lifecycle.items()
        }

        comparison = build_target_comparison(evaluations, lifecycle)
        comparison_path = run_root / "target_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        baseline = float(evaluations["without_tl"].metrics_original["RMSE"])
        manifest["transfer_comparison"] = {
            method: transfer_interpretation(
                baseline,
                float(evaluations[method].metrics_original["RMSE"]),
            )
            for method in ("tl_freeze", "tl_full_finetune")
        }
        manifest["artifacts"]["target_comparison.csv"] = artifact_records(
            (comparison_path,), run_root
        )["target_comparison.csv"]

        stage = "post_run_integrity"
        _require(validate_checksums(profile_path("A", "source")) == source.checksums, "Source data changed during run")
        _require(validate_checksums(profile_path("A", "target")) == target.checksums, "Target data changed during run")
        _require(
            collect_source_hashes() == manifest["source_file_sha256"],
            "Experiment-critical source files changed during run",
        )
        final_status = collect_git_provenance()["status_porcelain"]
        audit_git_writes(
            final_status,
            run_root,
            baseline_status_lines=provenance["status_porcelain"],
        )

        manifest["status"] = "PASS"
        manifest["timestamps"]["end"] = _utc_now()
        manifest["timestamps"]["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        write_manifest_atomic(manifest_path, manifest)
        return FormalRunResult(run_root=run_root, manifest_path=manifest_path)
    except KeyboardInterrupt as exc:
        if run_root is not None and manifest is not None and manifest_path is not None:
            try:
                _persist_failed_run(
                    run_root=run_root,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    failed_stage=stage,
                    exc=exc,
                    started=started,
                    message="Formal run interrupted by user",
                )
            except Exception:
                pass
        exc.run_root = run_root
        exc.manifest_path = manifest_path
        exc.failed_stage = stage
        raise
    except Exception as exc:
        if run_root is not None and manifest is not None and manifest_path is not None:
            try:
                _persist_failed_run(
                    run_root=run_root,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    failed_stage=stage,
                    exc=exc,
                    started=started,
                )
            except Exception:
                pass
        raise FormalRunError(str(exc), run_root) from exc
