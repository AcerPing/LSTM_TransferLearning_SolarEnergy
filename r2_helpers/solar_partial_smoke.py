"""R2.5 Phase-C two-epoch Partial FT integration smoke.

This diagnostic lifecycle uses Target training and validation only.  It never
loads Target Test, selects a formal strategy, or classifies transfer quality.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from keras.callbacks import Callback

from r2_config.solar_r25 import (
    BATCH_NORMALIZATION_INDICES,
    PARTIAL_STRATEGY_ORDER,
    R25_BATCH_SIZE,
    R25_LEARNING_RATE,
    R25_LOSS,
    R25_SEED,
    R25_SMOKE_OUTPUT_BASE,
    WEIGHT_BEARING_LAYER_INDICES,
)
from r2_config.solar_r2 import (
    EXPERIMENTS,
    EXPECTED_SEQUENCE_COUNTS,
    FEATURE_COLUMNS,
    WINDOW,
    profile_path,
)
from r2_helpers.solar_data import (
    build_sequences,
    load_profile_metadata,
    load_split,
    validate_feature_transform_consistency,
    validate_manifest,
    validate_scalers,
    validate_target_round_trip,
)
from r2_helpers.solar_formal import compute_metrics
from r2_helpers.solar_partial_ft import (
    STRATEGY_REGISTRY,
    batch_normalization_state,
    build_partial_candidate,
    candidate_checkpoint_pattern,
    inverse_transform_original,
    make_partial_callbacks,
    parameter_counts,
    reload_best_checkpoint,
    sha256_file,
    validate_baseline_source_checkpoint,
    validate_partial_transfer,
)
from r2_helpers.solar_runtime import (
    NamedSequenceSplit,
    configure_reproducibility,
    fit_train_validation,
)


PARTIAL_SMOKE_EPOCHS = 2
SMOKE_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z_seed1234$")
BN_STATE_NAMES = ("gamma", "beta", "moving_mean", "moving_variance")
SMOKE_TARGET_CHECKSUM_FILES = (
    "original_scale_training.csv",
    "normalized_scale_training.csv",
    "original_scale_validation.csv",
    "normalized_scale_validation.csv",
    "feature_scaler.joblib",
    "target_scaler.joblib",
    "split_manifest.json",
    "scaler_manifest.json",
)


class PartialSmokeError(RuntimeError):
    """Raised after preserving a failed R2.5 smoke manifest."""

    def __init__(self, message: str, *, smoke_root: Path | None = None):
        super().__init__(message)
        self.smoke_root = smoke_root


@dataclass(frozen=True)
class PartialSmokeResult:
    smoke_root: Path
    manifest_path: Path
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class SmokeSequenceSplit:
    named: NamedSequenceSplit
    split_data: Any
    sequences: Any


@dataclass(frozen=True)
class SmokeTargetData:
    metadata: Any
    scalers: Any
    checksums: Mapping[str, str]
    training: SmokeSequenceSplit
    validation: SmokeSequenceSplit


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PartialSmokeError(message)


def generate_smoke_run_id(now: datetime | None = None) -> str:
    instant = now or datetime.now(timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ_seed1234")


def create_smoke_root(
    run_id: str | None = None,
    *,
    output_base: Path | str = R25_SMOKE_OUTPUT_BASE,
) -> Path:
    """Create a unique diagnostic root outside the formal Partial_FT namespace."""

    identifier = run_id or generate_smoke_run_id()
    _require(bool(SMOKE_RUN_ID_PATTERN.fullmatch(identifier)), "Invalid smoke run ID")
    base = Path(output_base).resolve()
    root = (base / identifier).resolve()
    _require(root.parent == base, "Smoke root escapes its dedicated output base")
    _require(not root.exists(), f"Smoke run already exists: {root}")
    base.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    return root


def prepare_candidate_checkpoint_directory(root: Path, strategy_id: str) -> Path:
    """Explicitly create the ModelCheckpoint parent before fitting."""

    pattern = candidate_checkpoint_pattern(root, strategy_id)
    _require(not pattern.parent.exists(), f"Candidate directory exists: {pattern.parent}")
    pattern.parent.mkdir(parents=True, exist_ok=False)
    _require(pattern.parent.is_dir(), "Checkpoint parent was not created")
    return pattern


def validate_smoke_target_checksums(profile_dir: Path | str) -> Mapping[str, str]:
    """Hash only artifacts needed by Target training/validation smoke."""

    directory = Path(profile_dir)
    checksum_path = directory / "file_checksums.csv"
    try:
        with checksum_path.open("r", encoding="utf-8-sig", newline="") as stream:
            records = {row["file_name"]: row["sha256"].lower() for row in csv.DictReader(stream)}
    except (OSError, KeyError) as exc:
        raise PartialSmokeError(f"Cannot read Target checksum manifest: {exc}") from exc
    results: dict[str, str] = {}
    for name in SMOKE_TARGET_CHECKSUM_FILES:
        _require(name in records, f"Target checksum is missing: {name}")
        path = directory / name
        _require(path.is_file(), f"Target smoke artifact is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == records[name], f"Target smoke checksum mismatch: {name}")
        results[name] = actual
    _require(
        all("test" not in name for name in results),
        "Target Test artifact entered smoke checksum scope",
    )
    return results


def prepare_target_training_validation() -> SmokeTargetData:
    """Load Target training/validation CSVs without opening Target Test CSVs."""

    mapping = EXPERIMENTS["A"]["target"]
    directory = profile_path("A", "target")
    checksums = validate_smoke_target_checksums(directory)
    metadata = load_profile_metadata(directory)
    validate_manifest(
        metadata,
        expected_plant=mapping["manifest_plant"],
        expected_profile=mapping["manifest_profile"],
        expected_profile_dir_name=mapping["profile_dir"],
    )
    scalers = validate_scalers(metadata)
    loaded: dict[str, SmokeSequenceSplit] = {}
    expected = EXPECTED_SEQUENCE_COUNTS[mapping["profile_dir"]]
    for split_name in ("training", "validation"):
        split = load_split(metadata, split_name)
        validate_feature_transform_consistency(split, scalers.feature_scaler)
        validate_target_round_trip(split, scalers.target_scaler)
        sequences = build_sequences(
            split.X_normalized,
            split.y_normalized,
            split.timestamps,
        )
        _require(sequences.count == expected[split_name], f"{split_name} count mismatch")
        _require(
            sequences.X_seq.shape == (expected[split_name], WINDOW, len(FEATURE_COLUMNS)),
            f"{split_name} sequence shape mismatch",
        )
        loaded[split_name] = SmokeSequenceSplit(
            named=NamedSequenceSplit(split_name, sequences.X_seq, sequences.y_seq),
            split_data=split,
            sequences=sequences,
        )
    return SmokeTargetData(
        metadata=metadata,
        scalers=scalers,
        checksums=checksums,
        training=loaded["training"],
        validation=loaded["validation"],
    )


def snapshot_weight_bearing_layers(model: Any) -> dict[int, tuple[np.ndarray, ...]]:
    return {
        index: tuple(weight.copy() for weight in model.layers[index].get_weights())
        for index in WEIGHT_BEARING_LAYER_INDICES
    }


def verify_layer_changes(
    model: Any,
    before: Mapping[int, tuple[np.ndarray, ...]],
    strategy_id: str,
) -> dict[str, Any]:
    strategy = STRATEGY_REGISTRY[strategy_id]
    rows: dict[str, Any] = {}
    for index in WEIGHT_BEARING_LAYER_INDICES:
        initial = before[index]
        current = tuple(model.layers[index].get_weights())
        _require(len(initial) == len(current), f"Layer {index} tensor count changed")
        changed = [not np.array_equal(left, right) for left, right in zip(initial, current)]
        max_delta = max(
            (float(np.max(np.abs(left - right))) for left, right in zip(initial, current)),
            default=0.0,
        )
        if index in strategy.trainable_indices:
            _require(any(changed), f"Trainable layer {index} did not update")
        else:
            _require(not any(changed), f"Frozen layer {index} changed")
        rows[str(index)] = {
            "runtime_name": model.layers[index].name,
            "class_name": type(model.layers[index]).__name__,
            "expected_trainable": index in strategy.trainable_indices,
            "tensor_changed": changed,
            "any_tensor_changed": any(changed),
            "max_abs_delta": max_delta,
            "verified": True,
        }
    return rows


def verify_bn_unchanged(
    model: Any,
    before: Mapping[int, tuple[np.ndarray, ...]],
) -> dict[str, Any]:
    after = batch_normalization_state(model)
    result: dict[str, Any] = {}
    for index in BATCH_NORMALIZATION_INDICES:
        _require(not model.layers[index].trainable, f"BN layer {index} became trainable")
        _require(len(before[index]) == 4 == len(after[index]), "Unexpected BN state count")
        states = {
            name: bool(np.array_equal(left, right))
            for name, left, right in zip(BN_STATE_NAMES, before[index], after[index])
        }
        _require(all(states.values()), f"Frozen BN layer {index} state changed")
        result[str(index)] = {
            "runtime_name": model.layers[index].name,
            "trainable": False,
            "states_unchanged": states,
            "all_states_unchanged": True,
        }
    return result


class BestValidationPredictionRecorder(Callback):
    """Capture the in-memory prediction corresponding to each saved best epoch."""

    def __init__(self, validation_X: np.ndarray):
        super().__init__()
        self.validation_X = validation_X
        self.best = np.inf
        self.best_epoch: int | None = None
        self.best_prediction: np.ndarray | None = None

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        values = dict(logs or {})
        value = values.get("val_loss")
        _require(value is not None and np.isfinite(value), "val_loss missing or non-finite")
        if float(value) < self.best:
            self.best = float(value)
            self.best_epoch = epoch + 1
            self.best_prediction = np.asarray(
                self.model(self.validation_X, training=False)
            ).copy()


def validation_smoke_metrics(
    role_data: Any,
    predictions: np.ndarray,
) -> dict[str, Any]:
    validation = role_data.validation
    predicted = np.asarray(predictions).reshape(-1)
    y_normalized = np.asarray(validation.named.y_seq).reshape(-1)
    _require(len(predicted) == len(y_normalized), "Validation prediction count mismatch")
    y_original_expected = validation.split_data.y_original[
        validation.sequences.target_row_index
    ]
    y_original = inverse_transform_original(
        y_normalized,
        role_data.scalers.target_scaler,
        expected_original=y_original_expected,
    )
    prediction_original = inverse_transform_original(
        predicted,
        role_data.scalers.target_scaler,
    )
    return {
        "SMOKE_ONLY": True,
        "split": "validation",
        "normalized": compute_metrics(y_normalized, predicted),
        "original": compute_metrics(y_original, prediction_original),
        "prediction_count": len(predicted),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _run_candidate(
    root: Path,
    strategy_id: str,
    role_data: Any,
    baseline: Any,
) -> dict[str, Any]:
    configure_reproducibility(R25_SEED)
    source_wrapper = reload_best_checkpoint(baseline.checkpoint_path)
    source_model = source_wrapper.model
    candidate = build_partial_candidate(
        source_model,
        strategy_id=strategy_id,
        output_dir=root,
        seed=R25_SEED,
    )
    validate_partial_transfer(source_model, candidate.model)

    before = snapshot_weight_bearing_layers(candidate.model)
    bn_before = batch_normalization_state(candidate.model)
    checkpoint_pattern = prepare_candidate_checkpoint_directory(root, strategy_id)
    callbacks = list(make_partial_callbacks(root, strategy_id))
    recorder = BestValidationPredictionRecorder(role_data.validation.named.X_seq)
    callbacks.append(recorder)

    history = fit_train_validation(
        candidate.model,
        training=role_data.training.named,
        validation=role_data.validation.named,
        callbacks=callbacks,
        epochs=PARTIAL_SMOKE_EPOCHS,
        verbose=2,
    )
    losses = [float(value) for value in history.history.get("loss", ())]
    val_losses = [float(value) for value in history.history.get("val_loss", ())]
    _require(len(losses) == PARTIAL_SMOKE_EPOCHS, "Unexpected smoke epoch count")
    _require(len(val_losses) == PARTIAL_SMOKE_EPOCHS, "Missing Validation loss")
    _require(np.isfinite(losses).all() and np.isfinite(val_losses).all(), "Non-finite loss")

    layer_verification = verify_layer_changes(candidate.model, before, strategy_id)
    bn_verification = verify_bn_unchanged(candidate.model, bn_before)
    _require(recorder.best_epoch is not None, "No best Validation epoch recorded")
    _require(recorder.best_prediction is not None, "No best Validation prediction recorded")

    checkpoints = sorted(checkpoint_pattern.parent.glob("checkpoint_epoch_*.hdf5"))
    _require(bool(checkpoints), "ModelCheckpoint did not write a file")
    _require(all(path.stat().st_size > 0 for path in checkpoints), "Empty checkpoint")
    selected = checkpoint_pattern.with_name(
        checkpoint_pattern.name.format(epoch=recorder.best_epoch)
    )
    _require(selected.is_file(), "Best-Validation checkpoint is missing")
    reloaded = reload_best_checkpoint(selected)
    reloaded_prediction = np.asarray(
        reloaded.model.predict(
            role_data.validation.named.X_seq,
            batch_size=R25_BATCH_SIZE,
            verbose=0,
        )
    )
    _require(
        np.allclose(
            reloaded_prediction,
            recorder.best_prediction,
            rtol=1e-6,
            atol=1e-7,
        ),
        "Reloaded prediction differs from the saved best model state",
    )
    metrics = validation_smoke_metrics(role_data, reloaded_prediction)

    record = {
        "SMOKE_ONLY": True,
        "strategy_id": strategy_id,
        "epochs_executed": PARTIAL_SMOKE_EPOCHS,
        "trainable_indices": list(candidate.strategy.trainable_indices),
        "frozen_indices": list(candidate.strategy.frozen_indices),
        "optimizer_identity": candidate.optimizer_identity,
        "optimizer": "Adam",
        "learning_rate": R25_LEARNING_RATE,
        "loss": R25_LOSS,
        "batch_size": R25_BATCH_SIZE,
        "parameter_counts": candidate.parameter_counts.__dict__,
        "layer_verification": layer_verification,
        "bn_verification": bn_verification,
        "history": {"loss": losses, "val_loss": val_losses},
        "best_validation_epoch": recorder.best_epoch,
        "best_validation_loss": recorder.best,
        "checkpoint_pattern": checkpoint_pattern.relative_to(root).as_posix(),
        "checkpoints": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in checkpoints
        ],
        "selected_checkpoint": {
            "path": selected.relative_to(root).as_posix(),
            "size_bytes": selected.stat().st_size,
            "sha256": sha256_file(selected),
            "reloaded": True,
            "validation_prediction_consistent": True,
        },
        "validation_metrics": metrics,
        "target_test_accessed": False,
        "positive_transfer_evaluated": False,
        "status": "PASS",
    }
    _write_json(checkpoint_pattern.parent / "smoke_result.json", record)
    return record


def run_partial_smoke(run_id: str | None = None) -> PartialSmokeResult:
    """Run exactly two Target train/validation epochs for each candidate."""

    root: Path | None = None
    manifest: dict[str, Any] = {
        "SMOKE_ONLY": True,
        "phase": "R2.5 Phase C",
        "status": "INITIALIZING",
        "seed": R25_SEED,
        "epochs_per_candidate": PARTIAL_SMOKE_EPOCHS,
        "splits_loaded": ["training", "validation"],
        "target_test_accessed": False,
        "positive_transfer_evaluated": False,
        "strategies": {},
        "failure": None,
    }
    try:
        root = create_smoke_root(run_id)
        manifest_path = root / "smoke_manifest.json"
        manifest["run_id"] = root.name
        manifest["status"] = "RUNNING"
        _write_json(manifest_path, manifest)

        baseline = validate_baseline_source_checkpoint()
        manifest["source_checkpoint"] = {
            "path": str(baseline.checkpoint_path),
            "sha256": baseline.sha256,
            "best_epoch": baseline.best_epoch,
        }
        role_data = prepare_target_training_validation()
        manifest["target_checksum_files_read"] = sorted(role_data.checksums)
        manifest["target_sequences"] = {
            "training": len(role_data.training.named.X_seq),
            "validation": len(role_data.validation.named.X_seq),
        }

        for strategy_id in PARTIAL_STRATEGY_ORDER:
            manifest["strategies"][strategy_id] = _run_candidate(
                root,
                strategy_id,
                role_data,
                baseline,
            )

        _require(
            all(
                result["target_test_accessed"] is False
                for result in manifest["strategies"].values()
            ),
            "Target Test isolation failed",
        )
        manifest["status"] = "PASS"
        _write_json(manifest_path, manifest)
        return PartialSmokeResult(
            smoke_root=root,
            manifest_path=manifest_path,
            summary=manifest,
        )
    except BaseException as exc:
        if root is not None:
            manifest["status"] = "FAIL"
            manifest["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                _write_json(root / "smoke_manifest.json", manifest)
            except Exception:
                pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, PartialSmokeError):
            exc.smoke_root = root
            raise
        raise PartialSmokeError(str(exc), smoke_root=root) from exc
