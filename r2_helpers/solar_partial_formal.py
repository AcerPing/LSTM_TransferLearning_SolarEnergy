"""R2.5 Phase-D Candidate-B formal Train + Validation lifecycle.

Target Test is deliberately absent from this module's data contract and API.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from r2_config.solar_r25 import (
    PRIMARY_STRATEGY_ID,
    R25_BATCH_SIZE,
    R25_LEARNING_RATE,
    R25_MAX_EPOCHS,
    R25_OUTPUT_BASE,
    R25_SEED,
)
from r2_helpers.solar_formal import LearningRateObserver, build_history_frame, compute_metrics
from r2_helpers.solar_partial_ft import (
    ValidationScore,
    batch_normalization_state,
    build_partial_candidate,
    inverse_transform_original,
    lock_selection,
    reload_best_checkpoint,
    select_validation_winner,
    sha256_file,
    validate_baseline_source_checkpoint,
    validate_partial_transfer,
    validate_r25_output_root,
)
from r2_helpers.solar_partial_smoke import (
    prepare_target_training_validation,
    snapshot_weight_bearing_layers,
    verify_bn_unchanged,
    verify_layer_changes,
)
from r2_helpers.solar_runtime import configure_reproducibility, fit_train_validation


FORMAL_RUN_ID_PATTERN = r"^\d{8}T\d{6}Z_seed1234$"
FORMAL_CHECKPOINT_TEMPLATE = "checkpoint_epoch_{epoch:04d}.hdf5"


class PartialFormalError(RuntimeError):
    """Raised after preserving Phase-D failure evidence when possible."""

    def __init__(self, message: str, *, run_root: Path | None = None):
        super().__init__(message)
        self.run_root = run_root


@dataclass(frozen=True)
class PartialFormalResult:
    run_root: Path
    manifest_path: Path
    selection_path: Path
    summary: Mapping[str, Any]


class EpochProgress(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.epochs_completed = 0

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        self.epochs_completed = epoch + 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PartialFormalError(message)


def generate_formal_run_id(now: datetime | None = None) -> str:
    instant = now or datetime.now(timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ_seed1234")


def create_partial_formal_root(
    run_id: str | None = None,
    *,
    output_base: Path | str = R25_OUTPUT_BASE,
) -> Path:
    import re

    identifier = run_id or generate_formal_run_id()
    _require(bool(re.fullmatch(FORMAL_RUN_ID_PATTERN, identifier)), "Invalid formal run ID")
    base = Path(output_base).resolve()
    candidate = (base / identifier).resolve()
    if base == R25_OUTPUT_BASE.resolve():
        validate_r25_output_root(candidate)
    else:
        _require(candidate.parent == base, "Formal root escapes its output base")
        _require(not candidate.exists(), f"Formal run already exists: {candidate}")
    base.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(exist_ok=False)
    return candidate


def formal_candidate_directory(run_root: Path | str) -> Path:
    return Path(run_root) / "candidate" / PRIMARY_STRATEGY_ID


def formal_checkpoint_pattern(run_root: Path | str) -> Path:
    return formal_candidate_directory(run_root) / FORMAL_CHECKPOINT_TEMPLATE


def prepare_formal_checkpoint_directory(run_root: Path | str) -> Path:
    pattern = formal_checkpoint_pattern(run_root)
    _require(not pattern.parent.exists(), f"Candidate directory exists: {pattern.parent}")
    pattern.parent.mkdir(parents=True, exist_ok=False)
    return pattern


def make_formal_partial_callbacks(checkpoint_pattern: Path | str) -> tuple[Any, ...]:
    destination = Path(checkpoint_pattern)
    _require(destination.name == FORMAL_CHECKPOINT_TEMPLATE, "Invalid checkpoint template")
    return (
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            filepath=str(destination),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact_record(path: Path, run_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(run_root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_history_and_curve(frame: pd.DataFrame, candidate_dir: Path) -> tuple[Path, Path]:
    history_path = candidate_dir / "history.csv"
    frame.to_csv(history_path, index=False)
    curve_path = candidate_dir / "training_curve.png"
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(frame["epoch"], frame["loss"], label="Training loss")
    axis.plot(frame["epoch"], frame["val_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("R2.5 Candidate B Training and Validation Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)
    return history_path, curve_path


def build_validation_outputs(
    role_data: Any,
    prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    split = role_data.validation
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    true_normalized = np.asarray(split.named.y_seq, dtype=np.float64).reshape(-1)
    _require(len(predicted) == len(true_normalized) == 126, "Validation count mismatch")
    true_original_expected = split.split_data.y_original[split.sequences.target_row_index]
    true_original = inverse_transform_original(
        true_normalized,
        role_data.scalers.target_scaler,
        expected_original=true_original_expected,
    )
    predicted_original = inverse_transform_original(predicted, role_data.scalers.target_scaler)
    frame = pd.DataFrame(
        {
            "sample_index": split.sequences.sample_index,
            "target_row_index": split.sequences.target_row_index,
            "target_timestamp": split.sequences.target_timestamp,
            "y_true_normalized": true_normalized,
            "y_pred_normalized": predicted,
            "y_true_original": true_original,
            "y_pred_original": predicted_original,
            "residual_original": true_original - predicted_original,
        }
    )
    return (
        frame,
        compute_metrics(true_normalized, predicted),
        compute_metrics(true_original, predicted_original),
    )


def _latest_checkpoint(run_root: Path) -> str | None:
    candidates = sorted(formal_candidate_directory(run_root).glob("checkpoint_epoch_*.hdf5"))
    return candidates[-1].relative_to(run_root).as_posix() if candidates else None


def run_partial_formal_validation(run_id: str | None = None) -> PartialFormalResult:
    """Train Candidate B and lock its best Validation checkpoint; never access Test."""

    run_root: Path | None = None
    progress = EpochProgress()
    stage = "create_run_root"
    manifest: dict[str, Any] = {
        "phase": "R2.5 Phase D",
        "status": "INITIALIZING",
        "strategy_id": PRIMARY_STRATEGY_ID,
        "seed": R25_SEED,
        "target_test_accessed": False,
        "test_metrics_used_for_selection": False,
        "positive_transfer_evaluated": False,
        "splits_loaded": ["training", "validation"],
        "epochs_completed": 0,
        "failure": None,
    }
    try:
        run_root = create_partial_formal_root(run_id)
        manifest_path = run_root / "run_manifest.json"
        manifest.update({"run_id": run_root.name, "status": "RUNNING"})
        _write_json_atomic(manifest_path, manifest)

        stage = "source_provenance"
        baseline = validate_baseline_source_checkpoint()
        manifest["source_checkpoint"] = {
            "path": str(baseline.checkpoint_path),
            "sha256": baseline.sha256,
            "best_epoch": baseline.best_epoch,
        }

        stage = "target_train_validation_load"
        role_data = prepare_target_training_validation()
        _require(len(role_data.training.named.X_seq) == 516, "Training count mismatch")
        _require(len(role_data.validation.named.X_seq) == 126, "Validation count mismatch")
        manifest["target_sequences"] = {"training": 516, "validation": 126}
        manifest["target_checksum_files_read"] = sorted(role_data.checksums)

        stage = "model_build"
        configure_reproducibility(R25_SEED)
        source = reload_best_checkpoint(baseline.checkpoint_path).model
        candidate = build_partial_candidate(
            source,
            strategy_id=PRIMARY_STRATEGY_ID,
            output_dir=run_root,
            seed=R25_SEED,
        )
        validate_partial_transfer(source, candidate.model)
        _require(candidate.parameter_counts.trainable_params == 29161, "Trainable params mismatch")
        before = snapshot_weight_bearing_layers(candidate.model)
        bn_before = batch_normalization_state(candidate.model)
        pattern = prepare_formal_checkpoint_directory(run_root)
        callbacks = list(make_formal_partial_callbacks(pattern))
        lr_observer = LearningRateObserver()
        callbacks.extend((lr_observer, progress))
        manifest["parameter_counts"] = candidate.parameter_counts.__dict__
        manifest["layer_manifest_before_fit"] = [
            dict(layer) for layer in candidate.layer_manifest
        ]
        _write_json_atomic(manifest_path, manifest)

        stage = "formal_fit"
        history = fit_train_validation(
            candidate.model,
            training=role_data.training.named,
            validation=role_data.validation.named,
            callbacks=callbacks,
            epochs=R25_MAX_EPOCHS,
            verbose=2,
        )
        history_frame = build_history_frame(history, lr_observer.learning_rates)
        manifest["epochs_completed"] = len(history_frame)
        best_index = int(np.argmin(history_frame["val_loss"].to_numpy()))
        best_epoch = best_index + 1
        selected = pattern.with_name(pattern.name.format(epoch=best_epoch))
        _require(selected.is_file() and selected.stat().st_size > 0, "Best checkpoint missing")

        stage = "layer_integrity"
        layer_verification = verify_layer_changes(candidate.model, before, PRIMARY_STRATEGY_ID)
        bn_verification = verify_bn_unchanged(candidate.model, bn_before)

        stage = "checkpoint_reload_validation"
        reloaded = reload_best_checkpoint(selected)
        prediction = reloaded.model.predict(
            role_data.validation.named.X_seq,
            batch_size=R25_BATCH_SIZE,
            verbose=0,
        )
        prediction_frame, metrics_normalized, metrics_original = build_validation_outputs(
            role_data, prediction
        )
        candidate_dir = formal_candidate_directory(run_root)
        history_path, curve_path = _write_history_and_curve(history_frame, candidate_dir)
        prediction_path = candidate_dir / "validation_predictions.csv"
        prediction_frame.to_csv(prediction_path, index=False)
        metrics_json_path = candidate_dir / "validation_metrics_original.json"
        _write_json_atomic(metrics_json_path, metrics_original)
        metrics_csv_path = candidate_dir / "validation_metrics_original.csv"
        pd.DataFrame([metrics_original]).to_csv(metrics_csv_path, index=False)

        stage = "selection_lock"
        score = ValidationScore(
            strategy_id=PRIMARY_STRATEGY_ID,
            split_name="validation",
            scale="original",
            rmse=float(metrics_original["RMSE"]),
            mae=float(metrics_original["MAE"]),
            r2=float(metrics_original["R2"]),
            best_epoch=best_epoch,
            checkpoint_path=selected,
        )
        decision = select_validation_winner((score,))
        selection_path = lock_selection(
            run_root / "selection.json",
            decision,
            source_checkpoint_sha256=baseline.sha256,
        )

        manifest.update(
            {
                "status": "PASS",
                "epochs_completed": len(history_frame),
                "best_epoch": best_epoch,
                "best_validation_loss": float(history_frame.iloc[best_index]["val_loss"]),
                "final_learning_rate": float(
                    candidate.model.optimizer.learning_rate.numpy()
                ),
                "layer_verification": layer_verification,
                "bn_verification": bn_verification,
                "validation_metrics_normalized": metrics_normalized,
                "validation_metrics_original": metrics_original,
                "selected_checkpoint": _artifact_record(selected, run_root),
                "selection_locked": True,
                "artifacts": {
                    path.name: _artifact_record(path, run_root)
                    for path in (
                        history_path,
                        curve_path,
                        prediction_path,
                        metrics_json_path,
                        metrics_csv_path,
                        selection_path,
                    )
                },
            }
        )
        _write_json_atomic(manifest_path, manifest)
        return PartialFormalResult(run_root, manifest_path, selection_path, manifest)
    except BaseException as exc:
        if run_root is not None:
            manifest.update(
                {
                    "status": "FAIL",
                    "epochs_completed": progress.epochs_completed,
                    "latest_valid_checkpoint": _latest_checkpoint(run_root),
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "stage": stage,
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            try:
                _write_json_atomic(run_root / "run_manifest.json", manifest)
            except Exception:
                pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, PartialFormalError):
            exc.run_root = run_root
            raise
        raise PartialFormalError(str(exc), run_root=run_root) from exc
