"""Disposable real-training smoke gate for the Solar R2 lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import keras
import numpy as np
import tensorflow as tf

from r2_config.solar_r2 import (
    CORRECTED_R1_ROOT,
    EXPERIMENTS,
    FEATURE_COLUMNS,
    HORIZON,
    WINDOW,
    profile_path,
)
from r2_helpers.solar_data import (
    ProfileMetadata,
    ScalerBundle,
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
from r2_helpers.solar_runtime import (
    R2_BATCH_SIZE,
    R2_METHOD_NAMES,
    TRANSFERABLE_LAYER_INDICES,
    NamedSequenceSplit,
    RuntimeContractError,
    build_r2_model,
    checkpoint_paths,
    configure_reproducibility,
    evaluate_test,
    fit_train_validation,
    make_r2_callbacks,
    reload_best_checkpoint,
    validate_transferred_weights,
)


SMOKE_EXPERIMENT = "A"
SMOKE_EPOCHS = 2
SMOKE_COUNTS = {"training": 256, "validation": 64, "test": 32}


class SmokeTrainingError(RuntimeError):
    """Raised after preserving evidence for a failed smoke run."""

    def __init__(self, message: str, smoke_root: Path):
        super().__init__(message)
        self.smoke_root = smoke_root


@dataclass(frozen=True)
class PreparedRole:
    role: str
    metadata: ProfileMetadata
    scalers: ScalerBundle
    training: NamedSequenceSplit
    validation: NamedSequenceSplit


@dataclass(frozen=True)
class SmokeRunResult:
    smoke_root: Path
    manifest_path: Path | None
    summary: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_smoke_root(root: Path | str) -> Path:
    """Require a dedicated location below the operating-system temp root."""

    resolved = Path(root).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    corrected_root = CORRECTED_R1_ROOT.resolve()
    if resolved == temp_root or not _is_within(resolved, temp_root):
        raise ValueError(f"Smoke root must be below OS temp: {resolved}")
    for forbidden in (repository_root, corrected_root):
        if _is_within(resolved, forbidden) or _is_within(forbidden, resolved):
            raise ValueError(f"Unsafe smoke root overlaps protected data: {resolved}")
    return resolved


def create_smoke_root(requested_root: Path | str | None = None) -> Path:
    if requested_root is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        requested_root = (
            Path(tempfile.gettempdir())
            / "solar_r2_smoke"
            / f"{stamp}_{os.getpid()}"
        )
    root = validate_smoke_root(requested_root)
    if root.exists():
        raise FileExistsError(f"Smoke root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return root


def finalize_smoke_root(
    root: Path | str,
    *,
    success: bool,
    keep_smoke: bool,
) -> bool:
    """Remove only successful, non-kept smoke output; preserve all failures."""

    resolved = validate_smoke_root(root)
    if success and not keep_smoke and resolved.exists():
        shutil.rmtree(resolved)
        return True
    return False


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    path = root / "smoke_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event(manifest: dict[str, Any], name: str) -> None:
    manifest["events"].append({"order": len(manifest["events"]), "name": name})


def _profile_contract(role: str) -> tuple[ProfileMetadata, ScalerBundle]:
    mapping = EXPERIMENTS[SMOKE_EXPERIMENT][role]
    directory = profile_path(SMOKE_EXPERIMENT, role)
    validate_required_files(directory)
    validate_checksums(directory)
    metadata = load_profile_metadata(directory)
    validate_manifest(
        metadata,
        expected_plant=mapping["manifest_plant"],
        expected_profile=mapping["manifest_profile"],
        expected_profile_dir_name=mapping["profile_dir"],
    )
    return metadata, validate_scalers(metadata)


def _named_prefix(
    metadata: ProfileMetadata,
    scalers: ScalerBundle,
    split_name: str,
) -> tuple[NamedSequenceSplit, dict[str, Any]]:
    split = load_split(metadata, split_name)
    validate_feature_transform_consistency(split, scalers.feature_scaler)
    validate_target_round_trip(split, scalers.target_scaler)
    sequences = build_sequences(
        split.X_normalized,
        split.y_normalized,
        split.timestamps,
    )
    count = SMOKE_COUNTS[split_name]
    if sequences.count < count:
        raise AssertionError(f"Insufficient {split_name} sequences for smoke")
    X = sequences.X_seq[:count].copy()
    y = sequences.y_seq[:count].copy()
    if X.shape != (count, WINDOW, len(FEATURE_COLUMNS)):
        raise AssertionError(f"Unexpected smoke sequence shape: {X.shape}")
    named = NamedSequenceSplit(name=split_name, X_seq=X, y_seq=y)
    evidence = {
        "count": count,
        "shape": list(X.shape),
        "first_sample_index": int(sequences.sample_index[0]),
        "last_sample_index": int(sequences.sample_index[count - 1]),
        "first_target_row_index": int(sequences.target_row_index[0]),
        "last_target_row_index": int(sequences.target_row_index[count - 1]),
        "first_target_timestamp": str(sequences.target_timestamp[0]),
        "last_target_timestamp": str(sequences.target_timestamp[count - 1]),
    }
    return named, evidence


def _prepare_role(
    role: str,
    manifest: dict[str, Any],
) -> PreparedRole:
    metadata, scalers = _profile_contract(role)
    training, training_evidence = _named_prefix(
        metadata, scalers, "training"
    )
    validation, validation_evidence = _named_prefix(
        metadata, scalers, "validation"
    )
    manifest["subsets"][role] = {
        "training": training_evidence,
        "validation": validation_evidence,
    }
    _event(manifest, f"load_training_validation:{role}")
    return PreparedRole(
        role=role,
        metadata=metadata,
        scalers=scalers,
        training=training,
        validation=validation,
    )


def _load_delayed_test(
    prepared: PreparedRole,
    manifest: dict[str, Any],
    method: str,
) -> NamedSequenceSplit:
    test, evidence = _named_prefix(prepared.metadata, prepared.scalers, "test")
    manifest["subsets"][prepared.role]["test"] = evidence
    _event(manifest, f"load_test:{method}")
    return test


def _assert_history(history: Any, method: str) -> dict[str, Any]:
    loss = np.asarray(history.history.get("loss", ()), dtype=np.float64)
    val_loss = np.asarray(history.history.get("val_loss", ()), dtype=np.float64)
    if len(loss) != SMOKE_EPOCHS or not np.isfinite(loss).all():
        raise AssertionError(f"{method} loss history is invalid")
    if len(val_loss) != SMOKE_EPOCHS or not np.isfinite(val_loss).all():
        raise AssertionError(f"{method} val_loss history is invalid")
    return {
        "epochs_completed": SMOKE_EPOCHS,
        "loss_finite": True,
        "val_loss_finite": True,
        "val_loss_received": True,
    }


def _assert_checkpoint_path(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if not _is_within(resolved, root.resolve()):
        raise AssertionError(f"Checkpoint escaped smoke root: {resolved}")
    if path.name != "best_model.hdf5":
        raise AssertionError(f"Unexpected checkpoint name: {path.name}")


def _fit_method(
    *,
    method: str,
    model: Any,
    prepared: PreparedRole,
    checkpoint: Path,
    root: Path,
    manifest: dict[str, Any],
) -> None:
    _assert_checkpoint_path(checkpoint, root)
    if checkpoint.exists():
        raise AssertionError(f"Checkpoint exists before fit: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    callbacks = make_r2_callbacks(checkpoint)
    _event(manifest, f"fit_start:{method}")
    history = fit_train_validation(
        model,
        training=prepared.training,
        validation=prepared.validation,
        callbacks=callbacks,
        epochs=SMOKE_EPOCHS,
        verbose=2,
    )
    _event(manifest, f"fit_end:{method}")
    manifest["lifecycles"][method].update(_assert_history(history, method))
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise AssertionError(f"Checkpoint not produced for {method}")
    manifest["checkpoints"][method] = {
        "path": checkpoint.relative_to(root).as_posix(),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }


def _reload_and_evaluate(
    *,
    method: str,
    checkpoint: Path,
    prepared: PreparedRole,
    manifest: dict[str, Any],
) -> None:
    reloaded = reload_best_checkpoint(checkpoint)
    if reloaded.checkpoint_path.resolve() != checkpoint.resolve():
        raise AssertionError(f"Reload path mismatch for {method}")
    _event(manifest, f"reload:{method}")
    test = _load_delayed_test(prepared, manifest, method)
    evaluation = evaluate_test(reloaded, test=test, verbose=0)
    _event(manifest, f"evaluate_test:{method}")
    if len(evaluation.predictions) != SMOKE_COUNTS["test"]:
        raise AssertionError(f"Prediction count mismatch for {method}")
    if not np.isfinite(evaluation.predictions).all():
        raise AssertionError(f"Non-finite predictions for {method}")
    manifest["lifecycles"][method].update(
        {
            "checkpoint_reloaded": True,
            "test_loaded_after_reload": True,
            "prediction_count": len(evaluation.predictions),
            "predictions_finite": True,
            "status": "PASS",
        }
    )


def _all_weights_snapshot(model: Any) -> dict[int, list[np.ndarray]]:
    return {
        index: [weight.copy() for weight in model.layers[index].get_weights()]
        for index in TRANSFERABLE_LAYER_INDICES
    }


def _trainable_snapshot(model: Any) -> list[np.ndarray]:
    return [
        np.asarray(variable.numpy()).copy()
        for index in TRANSFERABLE_LAYER_INDICES
        for variable in model.layers[index].trainable_weights
    ]


def _run_lifecycles(
    root: Path,
    seed: int,
    manifest: dict[str, Any],
) -> None:
    source = _prepare_role("source", manifest)
    target = _prepare_role("target", manifest)
    paths = checkpoint_paths(root)

    configure_reproducibility(seed)
    source_model = build_r2_model(output_dir=root)
    _fit_method(
        method="source_pretrain",
        model=source_model,
        prepared=source,
        checkpoint=paths.source_pretrain,
        root=root,
        manifest=manifest,
    )
    _reload_and_evaluate(
        method="source_pretrain",
        checkpoint=paths.source_pretrain,
        prepared=source,
        manifest=manifest,
    )

    source_reloads_before_without = sum(
        event["name"].startswith("reload:source_pretrain")
        for event in manifest["events"]
    )
    configure_reproducibility(seed)
    without_model = build_r2_model(output_dir=root, pretrained_model=None)
    _fit_method(
        method="without_tl",
        model=without_model,
        prepared=target,
        checkpoint=paths.without_tl,
        root=root,
        manifest=manifest,
    )
    source_reloads_after_without_fit = sum(
        event["name"].startswith("reload:source_pretrain")
        for event in manifest["events"]
    )
    if source_reloads_after_without_fit != source_reloads_before_without:
        raise AssertionError("Without-TL read the Source checkpoint before training")
    manifest["lifecycles"]["without_tl"][
        "source_checkpoint_reads_before_training"
    ] = 0
    _reload_and_evaluate(
        method="without_tl",
        checkpoint=paths.without_tl,
        prepared=target,
        manifest=manifest,
    )

    source_best_for_freeze = reload_best_checkpoint(paths.source_pretrain)
    _event(manifest, "reload_source_for:tl_freeze")
    configure_reproducibility(seed)
    freeze_initial = build_r2_model(output_dir=root)
    configure_reproducibility(seed)
    freeze_model = build_r2_model(
        output_dir=root,
        pretrained_model=source_best_for_freeze.model,
        freeze=True,
    )
    validate_transferred_weights(
        source_best_for_freeze.model,
        freeze_model,
        freeze_initial,
        freeze=True,
    )
    freeze_before = _all_weights_snapshot(freeze_model)
    _fit_method(
        method="tl_freeze",
        model=freeze_model,
        prepared=target,
        checkpoint=paths.tl_freeze,
        root=root,
        manifest=manifest,
    )
    for index in TRANSFERABLE_LAYER_INDICES:
        after = freeze_model.layers[index].get_weights()
        if len(after) != len(freeze_before[index]) or any(
            not np.array_equal(before, current)
            for before, current in zip(freeze_before[index], after)
        ):
            raise AssertionError(f"Frozen layer {index} changed during smoke fit")
    manifest["lifecycles"]["tl_freeze"].update(
        {
            "transferable_weights_unchanged": True,
            "checked_layer_indices": list(TRANSFERABLE_LAYER_INDICES),
        }
    )
    _reload_and_evaluate(
        method="tl_freeze",
        checkpoint=paths.tl_freeze,
        prepared=target,
        manifest=manifest,
    )

    source_best_for_full = reload_best_checkpoint(paths.source_pretrain)
    _event(manifest, "reload_source_for:tl_full_finetune")
    configure_reproducibility(seed)
    full_initial = build_r2_model(output_dir=root)
    configure_reproducibility(seed)
    full_model = build_r2_model(
        output_dir=root,
        pretrained_model=source_best_for_full.model,
        freeze=False,
    )
    validate_transferred_weights(
        source_best_for_full.model,
        full_model,
        full_initial,
        freeze=False,
    )
    full_before = _trainable_snapshot(full_model)
    iterations_before = int(full_model.optimizer.iterations.numpy())
    _fit_method(
        method="tl_full_finetune",
        model=full_model,
        prepared=target,
        checkpoint=paths.tl_full_finetune,
        root=root,
        manifest=manifest,
    )
    full_after = _trainable_snapshot(full_model)
    iterations_after = int(full_model.optimizer.iterations.numpy())
    if len(full_before) != len(full_after) or not any(
        not np.array_equal(before, after)
        for before, after in zip(full_before, full_after)
    ):
        raise AssertionError("Full fine-tuning changed no transferred trainable tensor")
    if iterations_after <= iterations_before:
        raise AssertionError("Full fine-tuning optimizer iterations did not increase")
    manifest["lifecycles"]["tl_full_finetune"].update(
        {
            "transferred_trainable_weight_changed": True,
            "optimizer_iterations_before": iterations_before,
            "optimizer_iterations_after": iterations_after,
            "all_intended_layers_trainable": all(
                layer.trainable for layer in full_model.layers
            ),
        }
    )
    _reload_and_evaluate(
        method="tl_full_finetune",
        checkpoint=paths.tl_full_finetune,
        prepared=target,
        manifest=manifest,
    )

    path_values = [entry["path"] for entry in manifest["checkpoints"].values()]
    if len(path_values) != 4 or len(set(path_values)) != 4:
        raise AssertionError("Method checkpoint paths are not unique")


def run_smoke(
    *,
    seed: int = 1234,
    device: str = "cpu",
    keep_smoke: bool = False,
    requested_root: Path | str | None = None,
) -> SmokeRunResult:
    """Run exactly one Experiment-A, eight-epoch, disposable lifecycle gate."""

    root = create_smoke_root(requested_root)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "run_type": "smoke_only",
        "formal_result": False,
        "performance_comparison_allowed": False,
        "status": "RUNNING",
        "experiment": SMOKE_EXPERIMENT,
        "experiment_name": EXPERIMENTS[SMOKE_EXPERIMENT]["name"],
        "mapping": {
            "source": "Plant1/source_profile",
            "target": "Plant2/target_profile",
        },
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "device": device,
        "visible_gpu_count": None,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "numpy_version": np.__version__,
        "batch_size": R2_BATCH_SIZE,
        "window": WINDOW,
        "horizon": HORIZON,
        "epochs_per_lifecycle": SMOKE_EPOCHS,
        "total_authorized_epochs": 4 * SMOKE_EPOCHS,
        "method_names": list(R2_METHOD_NAMES),
        "subsets": {},
        "lifecycles": {
            name: {"status": "PENDING"}
            for name in (
                "source_pretrain",
                "without_tl",
                "tl_freeze",
                "tl_full_finetune",
            )
        },
        "checkpoints": {},
        "events": [],
        "smoke_start_timestamp": _utc_now(),
        "smoke_end_timestamp": None,
        "elapsed_seconds": None,
        "error": None,
    }
    manifest_path = _write_manifest(root, manifest)

    try:
        if device != "cpu":
            raise AssertionError("R2.3 supports CPU-only smoke")
        visible_gpus = tf.config.list_physical_devices("GPU")
        manifest["visible_gpu_count"] = len(visible_gpus)
        if visible_gpus:
            raise AssertionError(f"GPU visible in CPU-only smoke: {visible_gpus}")
        reproducibility = configure_reproducibility(seed)
        if not reproducibility.strict_hash_reproducibility:
            raise AssertionError("PYTHONHASHSEED was not set before process start")
        _event(manifest, "device_and_reproducibility_gate:PASS")
        _run_lifecycles(root, seed, manifest)
        manifest["status"] = "PASS"
        manifest["smoke_end_timestamp"] = _utc_now()
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_manifest(root, manifest)
    except Exception as exc:
        manifest["status"] = "FAIL"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["smoke_end_timestamp"] = _utc_now()
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_manifest(root, manifest)
        raise SmokeTrainingError(str(exc), root) from exc

    summary = {
        "status": manifest["status"],
        "experiment": manifest["experiment"],
        "device": manifest["device"],
        "visible_gpu_count": manifest["visible_gpu_count"],
        "total_epochs": manifest["total_authorized_epochs"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "smoke_root": str(root),
        "manifest_path": str(manifest_path),
        "cleaned_up": False,
    }
    cleaned = finalize_smoke_root(root, success=True, keep_smoke=keep_smoke)
    summary["cleaned_up"] = cleaned
    return SmokeRunResult(
        smoke_root=root,
        manifest_path=None if cleaned else manifest_path,
        summary=summary,
    )
