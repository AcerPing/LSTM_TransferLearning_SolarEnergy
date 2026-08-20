"""Explicit, disposable two-epoch integration smoke for Solar Linear A2/B.

Importing this module disables CUDA before TensorFlow/Keras is imported.  The
runner is intentionally separate from formal lifecycles and writes only under
``reports/Solar Energy Result/_smoke_linear``.
"""

from __future__ import annotations

import os

# Phase II is CPU-only.  This assignment must precede every TensorFlow/Keras
# import, including transitive imports through the Phase-I runtime.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import contextlib
import csv
import hashlib
import json
import platform
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping
from unittest.mock import patch

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.models import load_model

from r2_config.solar_linear import (
    LINEAR_BATCH_SIZE,
    LINEAR_EXPERIMENTS,
    LINEAR_FORMAL_BASE,
    LINEAR_SEED,
    PROTECTED_LEGACY_ROOTS,
    REPOSITORY_ROOT,
    LinearExperimentSpec,
)
from r2_helpers import solar_data
from r2_helpers.solar_data import TrainingValidationProfile
from r2_helpers.solar_linear_runtime import (
    EXPECTED_LAYER_CLASSES,
    LinearPartialCandidate,
    build_linear_partial_ft_candidate,
    build_linear_source_model,
    build_linear_without_tl_model,
    parameter_counts,
    rmse,
    validate_linear_model,
)


SMOKE_PHASE = "Phase II"
SMOKE_EPOCHS = 2
SMOKE_SHUFFLE = False
SOURCE_LEARNING_RATE = 1e-4
WOTL_LEARNING_RATE = 1e-4
PARTIAL_FT_LEARNING_RATE = 1e-5
LINEAR_SMOKE_BASE = (
    REPOSITORY_ROOT / "reports" / "Solar Energy Result" / "_smoke_linear"
)
FORBIDDEN_TEST_FILENAMES = {
    "normalized_scale_test.csv",
    "original_scale_test.csv",
}


class LinearSmokeContractError(AssertionError):
    """Raised immediately when the disposable Phase-II contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LinearSmokeContractError(message)


@dataclass(frozen=True)
class SmokeEnvironment:
    python: str
    tensorflow: str
    keras: str
    numpy: str
    device: str
    pythonhashseed: str
    seed: int


@dataclass(frozen=True)
class PartialStateAudit:
    layer2_unchanged: bool
    bn3_all_states_unchanged: bool
    layer4_changed: bool
    bn5_all_states_unchanged: bool
    layer1_changed: bool
    layer6_changed: bool


@dataclass(frozen=True)
class LifecycleSmokeResult:
    experiment_id: str
    method: str
    role: str
    epochs: int
    learning_rate: float
    training_rows: int
    validation_rows: int
    training_sequences: int
    validation_sequences: int
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str
    history_path: Path
    manifest_path: Path
    access_log_path: Path
    bn_audit_path: Path | None
    reload_passed: bool
    reload_mask_preserved: bool | None
    test_accessed: bool
    reloaded_model: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class ExperimentSmokeResult:
    source: LifecycleSmokeResult
    wotl: LifecycleSmokeResult
    partial_ft: LifecycleSmokeResult
    partial_state_audit: PartialStateAudit


@dataclass(frozen=True)
class PhaseIISmokeResult:
    run_root: Path
    environment: SmokeEnvironment
    experiments: Mapping[str, ExperimentSmokeResult]
    accessed_data_files: tuple[Path, ...]
    protected_roots_unchanged: bool
    linear_formal_created: bool


def _set_reproducibility() -> SmokeEnvironment:
    hash_seed = os.environ.get("PYTHONHASHSEED")
    _require(
        hash_seed == str(LINEAR_SEED),
        "Phase II requires PYTHONHASHSEED=1234 before process startup",
    )
    random.seed(LINEAR_SEED)
    np.random.seed(LINEAR_SEED)
    tf.random.set_seed(LINEAR_SEED)
    enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
    _require(callable(enable_determinism), "TensorFlow deterministic ops API unavailable")
    enable_determinism()
    _require(
        not tf.config.list_physical_devices("GPU"),
        "Phase II CPU policy failed: TensorFlow still exposes a GPU",
    )
    return SmokeEnvironment(
        python=platform.python_version(),
        tensorflow=tf.__version__,
        keras=keras.__version__,
        numpy=np.__version__,
        device="CPU",
        pythonhashseed=hash_seed,
        seed=LINEAR_SEED,
    )


def _reset_seed() -> None:
    random.seed(LINEAR_SEED)
    np.random.seed(LINEAR_SEED)
    tf.random.set_seed(LINEAR_SEED)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity() -> Mapping[str, Any]:
    safe_arg = f"safe.directory={REPOSITORY_ROOT.as_posix()}"

    def git_bytes(*args: str) -> bytes:
        completed = subprocess.run(
            ["git", "-c", safe_arg, *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
        )
        _require(
            completed.returncode == 0,
            f"Git provenance command failed: {' '.join(args)}",
        )
        return completed.stdout

    head = git_bytes("rev-parse", "HEAD").strip().decode("ascii")
    branch = git_bytes("branch", "--show-current").strip().decode("utf-8")
    dirty = bool(git_bytes("status", "--porcelain"))
    return MappingProxyType({"head": head, "branch": branch, "dirty": dirty})


def _protected_snapshot() -> Mapping[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for root in PROTECTED_LEGACY_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
    return MappingProxyType(snapshot)


def create_unique_smoke_run_root() -> Path:
    """Create one collision-protected disposable run namespace."""

    LINEAR_SMOKE_BASE.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_seed1234")
    run_root = LINEAR_SMOKE_BASE / run_id
    resolved = run_root.resolve()
    _require(resolved.is_relative_to(LINEAR_SMOKE_BASE.resolve()), "Smoke root escaped")
    _require(
        not resolved.is_relative_to(LINEAR_FORMAL_BASE.resolve()),
        "Smoke root points into Linear_Formal",
    )
    _require(
        not any(resolved.is_relative_to(root.resolve()) for root in PROTECTED_LEGACY_ROOTS),
        "Smoke root points into a protected Legacy root",
    )
    run_root.mkdir(parents=False, exist_ok=False)
    return run_root


@contextlib.contextmanager
def _record_data_access(accessed: list[Path]) -> Iterator[None]:
    """Record and pre-emptively block every Test CSV open attempt."""

    original_path_open = Path.open
    original_read_csv = solar_data.pd.read_csv
    original_joblib_load = solar_data.joblib.load

    def record(path: Path | str) -> None:
        resolved = Path(path).resolve()
        if resolved.name in FORBIDDEN_TEST_FILENAMES:
            raise LinearSmokeContractError(f"Forbidden Test CSV access: {resolved}")
        accessed.append(resolved)

    def path_open(path: Path, *args: Any, **kwargs: Any):
        record(path)
        return original_path_open(path, *args, **kwargs)

    def read_csv(path: Path | str, *args: Any, **kwargs: Any):
        record(path)
        return original_read_csv(path, *args, **kwargs)

    def joblib_load(path: Path | str, *args: Any, **kwargs: Any):
        record(path)
        return original_joblib_load(path, *args, **kwargs)

    with patch.object(Path, "open", path_open), patch.object(
        solar_data.pd, "read_csv", read_csv
    ), patch.object(solar_data.joblib, "load", joblib_load):
        yield


def _load_role(
    spec: LinearExperimentSpec,
    role: str,
    accessed: list[Path],
) -> TrainingValidationProfile:
    _require(role in ("source", "target"), f"Unknown smoke role: {role}")
    profile_path = spec.source_profile_path if role == "source" else spec.target_profile_path
    plant = spec.source_plant if role == "source" else spec.target_plant
    with _record_data_access(accessed):
        profile = solar_data.load_training_validation_profile(
            profile_path,
            expected_plant=plant,
            expected_profile=role,
            expected_profile_dir_name=f"{role}_profile",
        )
    _require(
        FORBIDDEN_TEST_FILENAMES.isdisjoint(path.name for path in accessed),
        "A Test CSV appeared in the smoke access log",
    )
    return profile


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_history(path: Path, history: Mapping[str, list[float]]) -> None:
    keys = tuple(history)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch",) + keys)
        for epoch in range(SMOKE_EPOCHS):
            writer.writerow((epoch + 1,) + tuple(history[key][epoch] for key in keys))


def _snapshot_weights(model: Any, indices: tuple[int, ...]) -> Mapping[int, tuple[np.ndarray, ...]]:
    return MappingProxyType(
        {
            index: tuple(weight.copy() for weight in model.layers[index].get_weights())
            for index in indices
        }
    )


def _all_equal(before: tuple[np.ndarray, ...], after: tuple[np.ndarray, ...]) -> bool:
    return len(before) == len(after) and all(
        np.array_equal(left, right) for left, right in zip(before, after)
    )


def _any_changed(before: tuple[np.ndarray, ...], after: tuple[np.ndarray, ...]) -> bool:
    return len(before) == len(after) and any(
        not np.array_equal(left, right) for left, right in zip(before, after)
    )


def _fit_lifecycle(
    *,
    spec: LinearExperimentSpec,
    method: str,
    role: str,
    model: Any,
    profile: TrainingValidationProfile,
    learning_rate: float,
    lifecycle_root: Path,
    accessed_data_files: tuple[Path, ...],
    environment: SmokeEnvironment,
    git_identity: Mapping[str, Any],
    partial_before: Mapping[int, tuple[np.ndarray, ...]] | None = None,
) -> tuple[LifecycleSmokeResult, PartialStateAudit | None]:
    lifecycle_root.mkdir(parents=True, exist_ok=False)
    checkpoint_pattern = lifecycle_root / "checkpoint_epoch_{epoch:04d}.hdf5"
    callbacks = (
        ModelCheckpoint(
            filepath=str(checkpoint_pattern),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        TerminateOnNaN(),
    )
    training = profile.training
    validation = profile.validation
    history_object = model.fit(
        training.sequences.X_seq,
        training.sequences.y_seq,
        validation_data=(validation.sequences.X_seq, validation.sequences.y_seq),
        epochs=SMOKE_EPOCHS,
        batch_size=LINEAR_BATCH_SIZE,
        shuffle=SMOKE_SHUFFLE,
        callbacks=list(callbacks),
        verbose=2,
    )
    history = {key: [float(value) for value in values] for key, values in history_object.history.items()}
    _require(history, f"{spec.experiment_id}/{method} returned empty history")
    _require(
        all(len(values) == SMOKE_EPOCHS for values in history.values()),
        f"{spec.experiment_id}/{method} did not execute exactly two epochs",
    )
    _require(
        all(np.isfinite(value) for values in history.values() for value in values),
        f"{spec.experiment_id}/{method} produced NaN/Inf diagnostics",
    )
    _require("val_loss" in history, "ModelCheckpoint selection requires val_loss")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = history["val_loss"][best_epoch - 1]
    checkpoint_path = lifecycle_root / f"checkpoint_epoch_{best_epoch:04d}.hdf5"
    _require(checkpoint_path.is_file(), f"Selected smoke checkpoint missing: {checkpoint_path}")
    _require(checkpoint_path.stat().st_size > 0, "Selected smoke checkpoint is empty")
    checkpoint_sha256 = _sha256(checkpoint_path)
    reloaded = load_model(str(checkpoint_path), custom_objects={"rmse": rmse})
    validate_linear_model(reloaded, expected_learning_rate=learning_rate)

    partial_audit: PartialStateAudit | None = None
    reload_mask_preserved: bool | None = None
    bn_audit_path: Path | None = None
    if partial_before is not None:
        after = _snapshot_weights(model, (1, 2, 3, 4, 5, 6))
        partial_audit = PartialStateAudit(
            layer2_unchanged=_all_equal(partial_before[2], after[2]),
            bn3_all_states_unchanged=_all_equal(partial_before[3], after[3]),
            layer4_changed=_any_changed(partial_before[4], after[4]),
            bn5_all_states_unchanged=_all_equal(partial_before[5], after[5]),
            layer1_changed=_any_changed(partial_before[1], after[1]),
            layer6_changed=_any_changed(partial_before[6], after[6]),
        )
        _require(all(partial_audit.__dict__.values()), "Partial FT state audit failed")
        reload_mask_preserved = (
            tuple(index for index in range(1, 7) if reloaded.layers[index].trainable)
            == (1, 4, 6)
            and tuple(index for index in range(1, 7) if not reloaded.layers[index].trainable)
            == (2, 3, 5)
        )
        _require(reload_mask_preserved, "Keras reload did not preserve the Partial FT mask")
        counts = parameter_counts(reloaded)
        _require(
            (counts.total, counts.trainable, counts.non_trainable)
            == (46681, 29161, 17520),
            "Reloaded Partial FT parameter mask mismatch",
        )
        bn_audit_path = lifecycle_root / "bn_state_audit.json"
        _write_json_exclusive(
            bn_audit_path,
            {
                **partial_audit.__dict__,
                "bn3_state_names": ["gamma", "beta", "moving_mean", "moving_variance"],
                "bn5_state_names": ["gamma", "beta", "moving_mean", "moving_variance"],
            },
        )

    history_path = lifecycle_root / "history.csv"
    _write_history(history_path, history)
    unique_accessed = tuple(dict.fromkeys(accessed_data_files))
    access_log_path = lifecycle_root / "access_log.json"
    _write_json_exclusive(
        access_log_path,
        {
            "opened_data_files": [str(path) for path in unique_accessed],
            "forbidden_test_csv_opened": False,
        },
    )
    manifest_path = lifecycle_root / "smoke_manifest.json"
    source_dependency = method == "partial_ft"
    payload = {
        "phase": SMOKE_PHASE,
        "smoke_only": True,
        "formal_eligible": False,
        "experiment_id": spec.experiment_id,
        "direction": spec.direction,
        "role": role,
        "method": method,
        "source": spec.source_plant,
        "target": spec.target_plant,
        "source_checkpoint_dependency": source_dependency,
        "activation": "linear",
        "learning_rate": learning_rate,
        "epochs": SMOKE_EPOCHS,
        "batch_size": LINEAR_BATCH_SIZE,
        "seed": LINEAR_SEED,
        "device": environment.device,
        "shuffle": SMOKE_SHUFFLE,
        "window": spec.window,
        "horizon": spec.horizon,
        "training_rows": training.split_data.row_count,
        "validation_rows": validation.split_data.row_count,
        "training_sequences": training.sequences.count,
        "validation_sequences": validation.sequences.count,
        "test_accessed": False,
        "test_metrics_computed": False,
        "positive_transfer_evaluated": False,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "reload_passed": True,
        "reload_mask_preserved": reload_mask_preserved,
        "git": dict(git_identity),
        "environment": environment.__dict__,
        "protected_root_written": False,
        "history_keys": list(history),
    }
    _write_json_exclusive(manifest_path, payload)
    return (
        LifecycleSmokeResult(
            experiment_id=spec.experiment_id,
            method=method,
            role=role,
            epochs=SMOKE_EPOCHS,
            learning_rate=learning_rate,
            training_rows=training.split_data.row_count,
            validation_rows=validation.split_data.row_count,
            training_sequences=training.sequences.count,
            validation_sequences=validation.sequences.count,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            history_path=history_path,
            manifest_path=manifest_path,
            access_log_path=access_log_path,
            bn_audit_path=bn_audit_path,
            reload_passed=True,
            reload_mask_preserved=reload_mask_preserved,
            test_accessed=False,
            reloaded_model=reloaded,
        ),
        partial_audit,
    )


def _run_experiment(
    spec: LinearExperimentSpec,
    *,
    run_root: Path,
    environment: SmokeEnvironment,
    git_identity: Mapping[str, Any],
    all_accessed: list[Path],
) -> ExperimentSmokeResult:
    experiment_root = run_root / spec.experiment_id
    experiment_root.mkdir(parents=False, exist_ok=False)

    source_accessed: list[Path] = []
    source_profile = _load_role(spec, "source", source_accessed)
    all_accessed.extend(source_accessed)
    _reset_seed()
    source_model = build_linear_source_model(
        output_dir=experiment_root,
        learning_rate=SOURCE_LEARNING_RATE,
    )
    source_result, _ = _fit_lifecycle(
        spec=spec,
        method="source",
        role="source",
        model=source_model,
        profile=source_profile,
        learning_rate=SOURCE_LEARNING_RATE,
        lifecycle_root=experiment_root / "source",
        accessed_data_files=tuple(source_accessed),
        environment=environment,
        git_identity=git_identity,
    )

    target_accessed: list[Path] = []
    target_profile = _load_role(spec, "target", target_accessed)
    all_accessed.extend(target_accessed)
    _reset_seed()
    wotl_model = build_linear_without_tl_model(
        output_dir=experiment_root,
        learning_rate=WOTL_LEARNING_RATE,
    )
    wotl_result, _ = _fit_lifecycle(
        spec=spec,
        method="wotl",
        role="target",
        model=wotl_model,
        profile=target_profile,
        learning_rate=WOTL_LEARNING_RATE,
        lifecycle_root=experiment_root / "wotl",
        accessed_data_files=tuple(target_accessed),
        environment=environment,
        git_identity=git_identity,
    )

    _reset_seed()
    partial: LinearPartialCandidate = build_linear_partial_ft_candidate(
        source_result.reloaded_model,
        output_dir=experiment_root,
        learning_rate=PARTIAL_FT_LEARNING_RATE,
    )
    before = _snapshot_weights(partial.model, (1, 2, 3, 4, 5, 6))
    partial_result, partial_audit = _fit_lifecycle(
        spec=spec,
        method="partial_ft",
        role="target",
        model=partial.model,
        profile=target_profile,
        learning_rate=PARTIAL_FT_LEARNING_RATE,
        lifecycle_root=experiment_root / "partial_ft",
        accessed_data_files=tuple(target_accessed),
        environment=environment,
        git_identity=git_identity,
        partial_before=before,
    )
    _require(partial_audit is not None, "Partial FT audit was not produced")
    return ExperimentSmokeResult(
        source=source_result,
        wotl=wotl_result,
        partial_ft=partial_result,
        partial_state_audit=partial_audit,
    )


def run_phase_ii_smoke() -> PhaseIISmokeResult:
    """Run A2 then B, six lifecycles total, exactly two epochs each."""

    environment = _set_reproducibility()
    git_identity = _git_identity()
    protected_before = _protected_snapshot()
    _require(not LINEAR_FORMAL_BASE.exists(), "Linear_Formal must not exist before smoke")
    run_root = create_unique_smoke_run_root()
    all_accessed: list[Path] = []
    experiments: dict[str, ExperimentSmokeResult] = {}
    for experiment_id in ("A2", "B"):
        experiments[experiment_id] = _run_experiment(
            LINEAR_EXPERIMENTS[experiment_id],
            run_root=run_root,
            environment=environment,
            git_identity=git_identity,
            all_accessed=all_accessed,
        )
    _require(
        FORBIDDEN_TEST_FILENAMES.isdisjoint(path.name for path in all_accessed),
        "Test CSV access detected after smoke",
    )
    protected_unchanged = protected_before == _protected_snapshot()
    _require(protected_unchanged, "A protected R2/R2.5 root changed during smoke")
    linear_formal_created = LINEAR_FORMAL_BASE.exists()
    _require(not linear_formal_created, "Linear_Formal was created during smoke")
    return PhaseIISmokeResult(
        run_root=run_root,
        environment=environment,
        experiments=MappingProxyType(experiments),
        accessed_data_files=tuple(dict.fromkeys(all_accessed)),
        protected_roots_unchanged=protected_unchanged,
        linear_formal_created=linear_formal_created,
    )
