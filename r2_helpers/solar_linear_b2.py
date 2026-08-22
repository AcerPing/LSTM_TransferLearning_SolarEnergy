"""Isolated Phase B2-1 post-Test supplementary tuning contracts.

The module is inert on import.  Real training and the already-revealed Plant1
Test are reachable only through explicit function calls; unit tests inject
fake models, arrays, loaders, callbacks, and temporary output roots.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from r2_config.solar_linear_b2 import (
    B2_AUTOMATIC_NEXT_ROUND,
    B2_BATCH_NORMALIZATION_LAYER_INDICES,
    B2_BATCH_SIZE,
    B2_CALLBACK_POLICY,
    B2_CANDIDATES,
    B2_DEVICE_POLICY,
    B2_DIRECTION,
    B2_DISCLOSURE,
    B2_EXPECTED_TEST_SEQUENCES,
    B2_EXPECTED_TARGET_ROW_COUNTS,
    B2_EXPECTED_TARGET_SEQUENCE_COUNTS,
    B2_EXPECTED_X_TEST_SHAPE,
    B2_EXPECTED_Y_TEST_SHAPE,
    B2_EXPERIMENT_ID,
    B2_FEATURES,
    B2_FIXED_WOTL_CANDIDATE_ID,
    B2_FIXED_WOTL_METRICS,
    B2_FORMAL_CLASSIFICATION,
    B2_FORMAL_COMPARISON_PATH,
    B2_FORMAL_EXPERIMENT_ID,
    B2_FORMAL_FINAL_TEST_GIT_HEAD,
    B2_FORMAL_RUN_ID,
    B2_FORMAL_RUN_ROOT,
    B2_FREQUENCY,
    B2_HORIZON,
    B2_LOCKED_SOURCE_CANDIDATE_ID,
    B2_LOCKED_SOURCE_CHECKPOINT,
    B2_LOCKED_SOURCE_SHA256,
    B2_LOSS,
    B2_MAXIMUM_EPOCHS,
    B2_OUTPUT_BASE,
    B2_PROTOCOL_VERSION,
    B2_RUN_ID_PATTERN,
    B2_SCALER_FIT_SPLIT,
    B2_SEED,
    B2_SELECTION_GIT_HEAD,
    B2_SHUFFLE,
    B2_SOURCE_PLANT,
    B2_TARGET_COLUMN,
    B2_TARGET_PROFILE_PATH,
    B2_TARGET_PLANT,
    B2_TOTAL_PARAMS,
    B2_TRANSFERRED_LAYER_INDICES,
    B2_VALIDATION_TEST_TRANSFORM_ONLY,
    B2_WINDOW,
    B2_WOTL_BASELINE_PATH,
    B2CandidateSpec,
    b2_candidate,
    validate_b2_config,
)


class B2ContractError(AssertionError):
    """Raised when supplementary B2 protocol evidence drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B2ContractError(message)


class B2Classification(str, Enum):
    POSITIVE = "Supplementary Positive Transfer"
    PARTIAL_POSITIVE = "Supplementary Partial Positive Transfer"
    MIXED = "Supplementary Mixed Result"
    NEGATIVE = "Supplementary Negative Transfer"


@dataclass(frozen=True)
class B2PathContract:
    run_id: str
    experiment_root: Path
    run_root: Path
    run_manifest: Path
    test_reuse_disclosure: Path
    baseline_root: Path
    wotl_baseline: Path
    candidates_root: Path
    candidate_roots: Mapping[str, Path]
    summary_root: Path
    comparison: Path
    results_summary: Path


@dataclass(frozen=True)
class B2Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float
    prediction_count: int
    negative_prediction_count: int
    target_name: str = "DC_POWER"
    scale: str = "original"
    unit: str = "DC_POWER (raw dataset scale; documented unit: kW)"


@dataclass(frozen=True)
class B2ModelContract:
    candidate: B2CandidateSpec
    model: Any
    transferred_layer_names: tuple[str, ...]
    trainable_layer_names: tuple[str, ...]
    frozen_layer_names: tuple[str, ...]
    trainable_params: int
    total_params: int
    trainable_ratio: float
    source_output_head_transferred: bool = False


@dataclass(frozen=True)
class B2FitEvidence:
    candidate_id: str
    history: Mapping[str, tuple[float, ...]]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_sha256_verified: bool
    selection_split: str = "validation"
    test_used_for_epoch_selection: bool = False


@dataclass(frozen=True)
class B2ExecutionGitProvenance:
    head: str
    branch: str
    tracked_dirty: bool
    untracked_paths: tuple[str, ...]
    critical_files_committed: bool


@dataclass(frozen=True)
class B2CheckpointLock:
    candidate_id: str
    strategy: str
    learning_rate: float
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_sha256_verified: bool
    selection_split: str
    test_used_for_epoch_selection: bool
    transferred_layer_indices: tuple[int, ...]
    trainable_layer_indices: tuple[int, ...]
    frozen_layer_indices: tuple[int, ...]
    transferred_layer_names: tuple[str, ...]
    trainable_layer_names: tuple[str, ...]
    frozen_layer_names: tuple[str, ...]
    trainable_params: int
    total_params: int
    trainable_ratio: float
    execution_git_head: str
    run_root: Path


@dataclass(frozen=True)
class B2LockedCandidate(B2CheckpointLock):
    validation_metrics_normalized: Mapping[str, float | int]
    validation_metrics_original_scale: Mapping[str, float | int | str]


@dataclass(frozen=True)
class B2ReloadedCandidate:
    locked_candidate: B2CheckpointLock
    model: Any
    checkpoint_sha256_verified: bool


@dataclass(frozen=True)
class B2ValidationCheckpointEvaluation:
    normalized_metrics: Mapping[str, float | int]
    original_scale_metrics: B2Metrics


@dataclass(frozen=True)
class B2TrainingValidationResult:
    paths: B2PathContract
    locked_candidates: Mapping[str, B2LockedCandidate]
    validation_registry_path: Path
    execution_git: B2ExecutionGitProvenance
    accessed_files: tuple[Path, ...]
    total_training_epochs_executed: int


@dataclass(frozen=True)
class B2RunnerDependencies:
    git_collector: Callable[[], B2ExecutionGitProvenance]
    runtime_gate: Callable[[], Mapping[str, Any]]
    path_factory: Callable[[str, Path | str], B2PathContract]
    baseline_loader: Callable[[], B2Metrics]
    source_checkpoint_validator: Callable[[], Path]
    source_model_loader: Callable[[], Any]
    profile_loader: Callable[[list[Path]], Any]
    run_initializer: Callable[[B2PathContract, Mapping[str, Any], B2Metrics], None]
    candidate_builder: Callable[..., B2ModelContract]
    candidate_fitter: Callable[..., B2FitEvidence]
    checkpoint_reloader: Callable[[B2CheckpointLock], B2ReloadedCandidate]
    validation_evaluator: Callable[[B2ReloadedCandidate, Any, Any], B2ValidationCheckpointEvaluation]
    candidate_writer: Callable[[B2PathContract, B2LockedCandidate, Mapping[str, Sequence[float]]], None]


@dataclass(frozen=True)
class B2TestData:
    X_test: np.ndarray
    y_test: np.ndarray
    target_timestamp: np.ndarray
    target_scaler: Any


@dataclass(frozen=True)
class B2PredictionEvaluation:
    prediction_normalized: np.ndarray
    prediction_original: np.ndarray
    y_true_original: np.ndarray
    metrics: B2Metrics
    normalized_mae: float
    normalized_mse: float
    normalized_rmse: float


@dataclass(frozen=True)
class B2CandidateComparison:
    candidate_id: str
    validation_best_val_loss: float
    metrics: B2Metrics
    delta_mae: float
    delta_mse: float
    delta_rmse: float
    delta_r2: float
    relative_mae_change_percent: float
    relative_mse_change_percent: float
    relative_rmse_change_percent: float
    r2_difference: float
    mae_better: bool
    mse_better: bool
    rmse_better: bool
    r2_better: bool
    success: bool
    classification: B2Classification


@dataclass(frozen=True)
class B2SupplementaryBatchResult:
    evaluations: Mapping[str, B2PredictionEvaluation]
    comparisons: Mapping[str, B2CandidateComparison]
    selected_candidate_id: str | None
    b2_goal_achieved: bool
    test_disk_load_count: int
    shared_x_test_object: bool


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"Required evidence file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"Required JSON missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B2ContractError(f"Cannot read B2 evidence JSON: {path}") from exc
    _require(isinstance(payload, Mapping), f"B2 evidence is not an object: {path}")
    return payload


_B2_CRITICAL_TRACKED_PATHS = (
    "r2_config/solar_linear_b2.py",
    "r2_helpers/solar_linear_b2.py",
    "tests/test_solar_linear_b2.py",
)


def _git_bytes(*arguments: str) -> bytes:
    from r2_config.solar_linear import REPOSITORY_ROOT

    safe_arg = f"safe.directory={REPOSITORY_ROOT.as_posix()}"
    completed = subprocess.run(
        ["git", "-c", safe_arg, *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    _require(completed.returncode == 0, f"B2 Git command failed: {' '.join(arguments)}")
    _require(completed.stdout is not None, "B2 Git stdout is unavailable")
    _require(completed.stderr is not None, "B2 Git stderr is unavailable")
    return completed.stdout


def _git_path_is_committed(path: str) -> bool:
    from r2_config.solar_linear import REPOSITORY_ROOT

    safe_arg = f"safe.directory={REPOSITORY_ROOT.as_posix()}"
    completed = subprocess.run(
        ["git", "-c", safe_arg, "cat-file", "-e", f"HEAD:{path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    _require(completed.stdout is not None, "B2 Git stdout is unavailable")
    _require(completed.stderr is not None, "B2 Git stderr is unavailable")
    return completed.returncode == 0


def collect_b2_execution_git_provenance() -> B2ExecutionGitProvenance:
    """Collect byte-safe Git state without changing user or repository config."""

    head = _git_bytes("rev-parse", "HEAD").strip().decode("ascii")
    branch = _git_bytes("branch", "--show-current").strip().decode("utf-8")
    tracked_dirty = bool(_git_bytes("status", "--porcelain=v1", "--untracked-files=no"))
    entries = _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all")
    untracked = tuple(
        entry[3:].replace("\\", "/")
        for entry in entries.decode("utf-8").split("\0")
        if entry.startswith("?? ")
    )
    committed = all(_git_path_is_committed(path) for path in _B2_CRITICAL_TRACKED_PATHS)
    return B2ExecutionGitProvenance(
        head=head,
        branch=branch,
        tracked_dirty=tracked_dirty,
        untracked_paths=untracked,
        critical_files_committed=committed,
    )


def validate_b2_execution_git_provenance(
    provenance: B2ExecutionGitProvenance,
    *,
    expected_git_head: str,
) -> None:
    _require(provenance.head == expected_git_head, "B2 execution Git HEAD mismatch")
    _require(bool(provenance.branch), "B2 execution Git branch is detached/unknown")
    _require(not provenance.tracked_dirty, "B2 tracked working tree is dirty")
    _require(provenance.critical_files_committed, "B2 experiment-critical files are not committed")
    critical_untracked = tuple(
        path for path in provenance.untracked_paths if path in _B2_CRITICAL_TRACKED_PATHS
    )
    _require(not critical_untracked, "B2 experiment-critical files are untracked")


def enforce_b2_cpu_seed_determinism() -> Mapping[str, Any]:
    """Apply the locked runtime gate before any TensorFlow/Keras import."""

    _require(
        "tensorflow" not in sys.modules and "keras" not in sys.modules,
        "B2 CPU visibility must be fixed before TensorFlow/Keras import",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = B2_DEVICE_POLICY.cuda_visible_devices
    _require(os.environ.get("PYTHONHASHSEED") == str(B2_SEED), "B2 PYTHONHASHSEED was not pre-set")
    random.seed(B2_SEED)
    np.random.seed(B2_SEED)

    import keras
    import tensorflow as tf

    tf.random.set_seed(B2_SEED)
    enable = getattr(tf.config.experimental, "enable_op_determinism", None)
    _require(callable(enable), "B2 deterministic ops API is unavailable")
    enable()
    _require(not tf.config.list_physical_devices("GPU"), "B2 runtime exposes a GPU")
    return MappingProxyType(
        {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "device": B2_DEVICE_POLICY.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "seed": B2_SEED,
            "deterministic_ops": True,
        }
    )


def b2_path_contract(
    run_id: str,
    *,
    output_base: Path | str = B2_OUTPUT_BASE,
    _allow_test_output: bool = False,
) -> B2PathContract:
    """Resolve a new isolated B2 namespace without creating it."""

    _require(bool(B2_RUN_ID_PATTERN.fullmatch(run_id)), "Invalid B2 run_id")
    base = Path(output_base)
    if not _allow_test_output:
        _require(base.resolve() == B2_OUTPUT_BASE.resolve(), "Unexpected B2 output base")
    run_root = base / run_id
    resolved = run_root.resolve()
    _require(resolved.is_relative_to(base.resolve()), "B2 run root escaped output base")
    _require(
        not resolved.is_relative_to(B2_FORMAL_RUN_ROOT.resolve()),
        "B2 output overlaps sealed Experiment B",
    )
    candidate_roots = MappingProxyType(
        {candidate_id: run_root / "candidates" / candidate_id for candidate_id in B2_CANDIDATES}
    )
    return B2PathContract(
        run_id=run_id,
        experiment_root=base,
        run_root=run_root,
        run_manifest=run_root / "run_manifest.json",
        test_reuse_disclosure=run_root / "test_reuse_disclosure.json",
        baseline_root=run_root / "baseline",
        wotl_baseline=run_root / "baseline" / "wotl_fixed_baseline.json",
        candidates_root=run_root / "candidates",
        candidate_roots=candidate_roots,
        summary_root=run_root / "summary",
        comparison=run_root / "summary" / "comparison.json",
        results_summary=run_root / "summary" / "results_summary.json",
    )


def validate_new_b2_destination(paths: B2PathContract) -> None:
    _require(not paths.run_root.exists(), f"B2 run already exists: {paths.run_root}")


def load_fixed_wotl_baseline(
    *,
    metrics_path: Path | str = B2_WOTL_BASELINE_PATH,
    comparison_path: Path | str = B2_FORMAL_COMPARISON_PATH,
    _allow_test_evidence: bool = False,
) -> B2Metrics:
    """Re-read and exactly validate the sealed Formal WOTL baseline."""

    metrics_file = Path(metrics_path)
    comparison_file = Path(comparison_path)
    if not _allow_test_evidence:
        _require(
            metrics_file.resolve() == B2_WOTL_BASELINE_PATH.resolve(),
            "Unexpected WOTL baseline evidence",
        )
        _require(
            comparison_file.resolve() == B2_FORMAL_COMPARISON_PATH.resolve(),
            "Unexpected Formal comparison evidence",
        )
    payload = _load_json(metrics_file)
    for key, expected in B2_FIXED_WOTL_METRICS.items():
        _require(payload.get(key) == expected, f"Fixed WOTL baseline mismatch: {key}")
    comparison = _load_json(comparison_file)
    _require(
        comparison.get("classification") == B2_FORMAL_CLASSIFICATION,
        "Formal Experiment B classification changed",
    )
    _require(
        comparison.get("execution_git_head") == B2_FORMAL_FINAL_TEST_GIT_HEAD,
        "Formal Final Test execution provenance changed",
    )
    _require(
        comparison.get("selection_git_head") == B2_SELECTION_GIT_HEAD,
        "Formal selection provenance changed",
    )
    _require(
        comparison.get("wotl_candidate_id") == B2_FIXED_WOTL_CANDIDATE_ID,
        "Formal WOTL identity changed",
    )
    return B2Metrics(
        mae=float(payload["mae"]),
        mse=float(payload["mse"]),
        rmse=float(payload["rmse"]),
        r2=float(payload["r2"]),
        prediction_count=int(payload["prediction_count"]),
        negative_prediction_count=int(payload["negative_prediction_count"]),
        target_name=str(payload["target_name"]),
        scale=str(payload["scale"]),
        unit=str(payload["unit"]),
    )


def validate_locked_source_checkpoint(
    path: Path | str = B2_LOCKED_SOURCE_CHECKPOINT,
    *,
    sha256_func: Callable[[Path], str] = _sha256,
    _allow_test_checkpoint: bool = False,
) -> Path:
    checkpoint = Path(path)
    if not _allow_test_checkpoint:
        _require(
            checkpoint.resolve() == B2_LOCKED_SOURCE_CHECKPOINT.resolve(),
            "Unexpected B2 Source checkpoint",
        )
        _require(sha256_func is _sha256, "Production requires real Source SHA-256")
    _require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Source checkpoint missing/empty")
    _require(
        sha256_func(checkpoint) == B2_LOCKED_SOURCE_SHA256,
        "Locked Source checkpoint SHA mismatch",
    )
    return checkpoint


def load_locked_source_model() -> Any:
    """Reload the sealed Source model after revalidating its SHA-256."""

    checkpoint = validate_locked_source_checkpoint()
    from keras.models import load_model

    from r2_helpers.solar_linear_runtime import rmse, validate_linear_model

    model = load_model(str(checkpoint), custom_objects={"rmse": rmse})
    validate_linear_model(model)
    return model


def load_b2_target_training_validation(accessed_files: list[Path]) -> Any:
    """Load only corrected Plant1 Training/Validation through the Formal gate."""

    from r2_config.solar_linear import LINEAR_EXPERIMENTS
    from r2_helpers.solar_linear_formal_train import load_role_training_validation

    return load_role_training_validation(
        LINEAR_EXPERIMENTS[B2_FORMAL_EXPERIMENT_ID],
        "target",
        accessed_files,
    )


def load_revealed_plant1_test() -> B2TestData:
    """Explicit B2-1C loader for the already-revealed Plant1 Test."""

    from r2_helpers.solar_linear_final_test import load_locked_target_test

    data = load_locked_target_test()
    return B2TestData(
        X_test=data.X_test,
        y_test=data.y_test,
        target_timestamp=data.target_timestamp,
        target_scaler=data.target_scaler,
    )


def build_run_manifest(
    *,
    run_id: str,
    execution_git_head: str,
    tracked_dirty: bool,
    untracked_paths: Sequence[str],
) -> Mapping[str, Any]:
    _require(bool(B2_RUN_ID_PATTERN.fullmatch(run_id)), "Invalid B2 run_id")
    _require(len(execution_git_head) == 40, "B2 execution Git HEAD must be explicit")
    return MappingProxyType(
        {
            "protocol_version": B2_PROTOCOL_VERSION,
            "experiment": B2_EXPERIMENT_ID,
            "direction": B2_DIRECTION,
            "source_plant": B2_SOURCE_PLANT,
            "target_plant": B2_TARGET_PLANT,
            "run_id": run_id,
            "execution_git_head": execution_git_head,
            "tracked_dirty": bool(tracked_dirty),
            "untracked_paths": tuple(str(path) for path in untracked_paths),
            "formal_run_id": B2_FORMAL_RUN_ID,
            "formal_selection_git_head": B2_SELECTION_GIT_HEAD,
            "formal_final_test_execution_git_head": B2_FORMAL_FINAL_TEST_GIT_HEAD,
            "formal_classification_preserved": B2_FORMAL_CLASSIFICATION,
            **dict(B2_DISCLOSURE),
            "features": B2_FEATURES,
            "target": B2_TARGET_COLUMN,
            "target_profile_path": B2_TARGET_PROFILE_PATH,
            "target_row_counts": dict(B2_EXPECTED_TARGET_ROW_COUNTS),
            "target_sequence_counts": dict(B2_EXPECTED_TARGET_SEQUENCE_COUNTS),
            "target_split_reused_without_resplitting": True,
            "scaler_fit_split": B2_SCALER_FIT_SPLIT,
            "validation_test_transform_only": B2_VALIDATION_TEST_TRANSFORM_ONLY,
            "window": B2_WINDOW,
            "horizon": B2_HORIZON,
            "frequency": B2_FREQUENCY,
            "seed": B2_SEED,
            "device": B2_DEVICE_POLICY.device,
            "cuda_visible_devices": B2_DEVICE_POLICY.cuda_visible_devices,
            "batch_size": B2_BATCH_SIZE,
            "loss": B2_LOSS,
            "maximum_epochs": B2_MAXIMUM_EPOCHS,
            "shuffle": B2_SHUFFLE,
            "callback_policy": asdict(B2_CALLBACK_POLICY),
            "candidate_ids": tuple(B2_CANDIDATES),
            "automatic_next_round": B2_AUTOMATIC_NEXT_ROUND,
            "formal_result_replaced": False,
        }
    )


def test_reuse_disclosure() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "experiment": B2_EXPERIMENT_ID,
            **dict(B2_DISCLOSURE),
            "test_disk_load_count": 1,
        }
    )


def _layer_weights(model: Any, index: int) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(weight).copy() for weight in model.layers[index].get_weights())


def _weights_equal(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(first, second) for first, second in zip(left, right)
    )


def build_b2_candidate_model(
    source_model: Any,
    candidate_id: str,
    *,
    output_dir: Path | str,
) -> B2ModelContract:
    """Build one approved target model, transfer layers 1--5, and apply its mask."""

    from keras.optimizers import Adam

    from r2_helpers.solar_linear_runtime import (
        build_linear_without_tl_model,
        parameter_counts,
        rmse,
        validate_linear_model,
    )

    validate_b2_config()
    candidate = b2_candidate(candidate_id)
    validate_linear_model(source_model)
    target = build_linear_without_tl_model(
        output_dir=output_dir,
        learning_rate=candidate.learning_rate,
    )
    target_output_initial = _layer_weights(target, 6)
    for index in candidate.transferred_indices:
        _require(
            type(source_model.layers[index]).__name__ == type(target.layers[index]).__name__,
            f"B2 transfer layer {index} class mismatch",
        )
        target.layers[index].set_weights(source_model.layers[index].get_weights())
    for index in range(1, 7):
        target.layers[index].trainable = index in candidate.trainable_indices
    target.compile(
        optimizer=Adam(learning_rate=candidate.learning_rate),
        loss="mse",
        metrics=["mse", rmse, "mae", "mape", "msle"],
    )
    validate_linear_model(target, expected_learning_rate=candidate.learning_rate)
    for index in candidate.transferred_indices:
        _require(
            _weights_equal(_layer_weights(source_model, index), _layer_weights(target, index)),
            f"B2 transferred layer {index} weights mismatch",
        )
    _require(
        _weights_equal(target_output_initial, _layer_weights(target, 6)),
        "B2 Source output head was transferred",
    )
    actual_trainable = tuple(index for index in range(1, 7) if target.layers[index].trainable)
    actual_frozen = tuple(index for index in range(1, 7) if not target.layers[index].trainable)
    _require(actual_trainable == candidate.trainable_indices, "B2 trainable mask mismatch")
    _require(actual_frozen == candidate.frozen_indices, "B2 frozen mask mismatch")
    _require(
        all(not target.layers[index].trainable for index in B2_BATCH_NORMALIZATION_LAYER_INDICES),
        "B2 BatchNormalization layers must remain frozen",
    )
    counts = parameter_counts(target)
    _require(counts.total == B2_TOTAL_PARAMS, "B2 total parameter count changed")
    _require(
        counts.trainable == candidate.expected_trainable_params,
        "B2 trainable parameter count changed",
    )
    return B2ModelContract(
        candidate=candidate,
        model=target,
        transferred_layer_names=tuple(target.layers[index].name for index in candidate.transferred_indices),
        trainable_layer_names=tuple(target.layers[index].name for index in candidate.trainable_indices),
        frozen_layer_names=tuple(target.layers[index].name for index in candidate.frozen_indices),
        trainable_params=counts.trainable,
        total_params=counts.total,
        trainable_ratio=counts.trainable / counts.total,
        source_output_head_transferred=False,
    )


def _history_mapping(history_object: Any) -> Mapping[str, tuple[float, ...]]:
    raw = history_object.history if hasattr(history_object, "history") else history_object
    _require(isinstance(raw, Mapping) and raw, "B2 fit returned no history")
    history = {str(key): tuple(float(value) for value in values) for key, values in raw.items()}
    _require(
        not any("test" in key.lower() for key in history),
        "B2 fit history must not contain Test metrics",
    )
    lengths = {len(values) for values in history.values()}
    _require(len(lengths) == 1 and next(iter(lengths)) > 0, "B2 history lengths differ")
    _require("val_loss" in history, "B2 history lacks val_loss")
    _require(
        all(math.isfinite(value) for values in history.values() for value in values),
        "B2 history contains NaN/Inf",
    )
    return MappingProxyType(history)


def fit_b2_candidate_validation_only(
    contract: B2ModelContract,
    training: Any,
    validation: Any,
    *,
    checkpoint_directory: Path | str,
    callback_factory: Callable[[Path | str], tuple[Any, ...]] | None = None,
    sha256_func: Callable[[Path], str] = _sha256,
) -> B2FitEvidence:
    """Fit one candidate using Training/Validation only; the API has no Test input."""

    if callback_factory is None:
        from r2_helpers.solar_linear_formal import make_formal_callbacks

        callback_factory = make_formal_callbacks
    checkpoint_root = Path(checkpoint_directory)
    _require(not checkpoint_root.exists(), "B2 checkpoint directory already exists")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    checkpoint_pattern = checkpoint_root / "checkpoint_epoch_{epoch:04d}.hdf5"
    callbacks = callback_factory(checkpoint_pattern)
    _require(len(callbacks) == 4, "B2 must reuse the four Formal callbacks")
    history_object = contract.model.fit(
        training.X_seq,
        training.y_seq,
        validation_data=(validation.X_seq, validation.y_seq),
        epochs=B2_MAXIMUM_EPOCHS,
        batch_size=B2_BATCH_SIZE,
        shuffle=B2_SHUFFLE,
        callbacks=list(callbacks),
        verbose=2,
    )
    history = _history_mapping(history_object)
    values = np.asarray(history["val_loss"], dtype=np.float64)
    best_epoch = int(np.argmin(values)) + 1
    checkpoint_path = checkpoint_root / f"checkpoint_epoch_{best_epoch:04d}.hdf5"
    _require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, "B2 best checkpoint missing")
    first = sha256_func(checkpoint_path)
    second = sha256_func(checkpoint_path)
    _require(first == second, "B2 checkpoint SHA verification failed")
    return B2FitEvidence(
        candidate_id=contract.candidate.candidate_id,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=float(values[best_epoch - 1]),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=first,
        checkpoint_sha256_verified=True,
    )


def lock_b2_best_checkpoint(
    contract: B2ModelContract,
    fit: B2FitEvidence,
    *,
    execution_git_head: str,
    run_root: Path | str,
) -> B2CheckpointLock:
    """Bind Validation selection evidence to one exact candidate checkpoint."""

    candidate = contract.candidate
    _require(fit.candidate_id == candidate.candidate_id, "B2 fit/model candidate mismatch")
    _require(fit.selection_split == "validation", "B2 checkpoint was not Validation-selected")
    _require(not fit.test_used_for_epoch_selection, "B2 Test selected the checkpoint")
    _require(fit.checkpoint_sha256_verified, "B2 checkpoint SHA was not verified")
    _require(len(execution_git_head) == 40, "B2 checkpoint lock lacks execution Git HEAD")
    root = Path(run_root)
    expected = (
        root
        / "candidates"
        / candidate.candidate_id
        / "checkpoints"
        / f"checkpoint_epoch_{fit.best_epoch:04d}.hdf5"
    )
    _require(
        fit.checkpoint_path.resolve() == expected.resolve(),
        "B2 best checkpoint path is not candidate/run bound",
    )
    _require(contract.total_params == B2_TOTAL_PARAMS, "B2 checkpoint total parameters changed")
    _require(
        contract.trainable_params == candidate.expected_trainable_params,
        "B2 checkpoint trainable parameters changed",
    )
    return B2CheckpointLock(
        candidate_id=candidate.candidate_id,
        strategy=candidate.strategy,
        learning_rate=candidate.learning_rate,
        best_epoch=fit.best_epoch,
        best_val_loss=fit.best_val_loss,
        checkpoint_path=fit.checkpoint_path,
        checkpoint_sha256=fit.checkpoint_sha256,
        checkpoint_sha256_verified=fit.checkpoint_sha256_verified,
        selection_split=fit.selection_split,
        test_used_for_epoch_selection=fit.test_used_for_epoch_selection,
        transferred_layer_indices=candidate.transferred_indices,
        trainable_layer_indices=candidate.trainable_indices,
        frozen_layer_indices=candidate.frozen_indices,
        transferred_layer_names=contract.transferred_layer_names,
        trainable_layer_names=contract.trainable_layer_names,
        frozen_layer_names=contract.frozen_layer_names,
        trainable_params=contract.trainable_params,
        total_params=contract.total_params,
        trainable_ratio=contract.trainable_ratio,
        execution_git_head=execution_git_head,
        run_root=root,
    )


def _validate_b2_checkpoint_lock(locked: B2CheckpointLock) -> B2CandidateSpec:
    _require(isinstance(locked, B2CheckpointLock), "B2 checkpoint evidence type is invalid")
    _require(locked.candidate_id in B2_CANDIDATES, "Unknown locked B2 candidate")
    candidate = b2_candidate(locked.candidate_id)
    _require(locked.strategy == candidate.strategy, "Locked B2 strategy mismatch")
    _require(locked.learning_rate == candidate.learning_rate, "Locked B2 learning rate mismatch")
    _require(
        locked.transferred_layer_indices == candidate.transferred_indices,
        "Locked B2 transfer indices mismatch",
    )
    _require(
        locked.trainable_layer_indices == candidate.trainable_indices,
        "Locked B2 trainable indices mismatch",
    )
    _require(
        locked.frozen_layer_indices == candidate.frozen_indices,
        "Locked B2 frozen indices mismatch",
    )
    _require(locked.selection_split == "validation", "Locked B2 selection split changed")
    _require(not locked.test_used_for_epoch_selection, "Locked B2 evidence used Test selection")
    _require(locked.checkpoint_sha256_verified, "Locked B2 SHA verification flag is false")
    _require(len(locked.checkpoint_sha256) == 64, "Locked B2 checkpoint SHA is malformed")
    _require(len(locked.execution_git_head) == 40, "Locked B2 execution Git HEAD is malformed")
    _require(locked.best_epoch > 0, "Locked B2 best epoch is invalid")
    _require(math.isfinite(locked.best_val_loss), "Locked B2 best val_loss is invalid")
    expected = (
        locked.run_root
        / "candidates"
        / locked.candidate_id
        / "checkpoints"
        / f"checkpoint_epoch_{locked.best_epoch:04d}.hdf5"
    )
    _require(
        locked.checkpoint_path.resolve() == expected.resolve(),
        "Locked B2 checkpoint path/epoch/run identity mismatch",
    )
    _require(locked.total_params == B2_TOTAL_PARAMS, "Locked B2 total parameter count mismatch")
    _require(
        locked.trainable_params == candidate.expected_trainable_params,
        "Locked B2 trainable parameter count mismatch",
    )
    _require(
        np.isclose(
            locked.trainable_ratio,
            candidate.expected_trainable_params / B2_TOTAL_PARAMS,
            rtol=0.0,
            atol=1e-15,
        ),
        "Locked B2 trainable ratio mismatch",
    )
    return candidate


def _load_and_validate_b2_model(path: Path, candidate: B2CandidateSpec, locked: B2CheckpointLock) -> Any:
    from keras.models import load_model
    from keras.optimizers import Adam

    from r2_helpers.solar_linear_runtime import parameter_counts, rmse, validate_linear_model

    model = load_model(str(path), custom_objects={"rmse": rmse}, compile=False)
    for index in range(1, 7):
        model.layers[index].trainable = index in candidate.trainable_indices
    model.compile(
        optimizer=Adam(learning_rate=candidate.learning_rate),
        loss=B2_LOSS,
        metrics=["mse", rmse, "mae", "mape", "msle"],
    )
    validate_linear_model(model, expected_learning_rate=candidate.learning_rate)
    actual_trainable = tuple(index for index in range(1, 7) if model.layers[index].trainable)
    actual_frozen = tuple(index for index in range(1, 7) if not model.layers[index].trainable)
    _require(actual_trainable == candidate.trainable_indices, "Reloaded B2 trainable mask mismatch")
    _require(actual_frozen == candidate.frozen_indices, "Reloaded B2 frozen mask mismatch")
    _require(
        all(not model.layers[index].trainable for index in B2_BATCH_NORMALIZATION_LAYER_INDICES),
        "Reloaded B2 BatchNormalization layer is trainable",
    )
    _require(
        tuple(model.layers[index].name for index in candidate.transferred_indices)
        == locked.transferred_layer_names,
        "Reloaded B2 transferred layer names mismatch",
    )
    _require(
        tuple(model.layers[index].name for index in candidate.trainable_indices)
        == locked.trainable_layer_names,
        "Reloaded B2 trainable layer names mismatch",
    )
    _require(
        tuple(model.layers[index].name for index in candidate.frozen_indices)
        == locked.frozen_layer_names,
        "Reloaded B2 frozen layer names mismatch",
    )
    counts = parameter_counts(model)
    _require(counts.total == locked.total_params, "Reloaded B2 total parameters mismatch")
    _require(counts.trainable == locked.trainable_params, "Reloaded B2 trainable parameters mismatch")
    return model


def reload_b2_best_checkpoint(
    locked_candidate: B2CheckpointLock,
    *,
    sha256_func: Callable[[Path], str] = _sha256,
    model_loader_validator: Callable[[Path, B2CandidateSpec, B2CheckpointLock], Any]
    | None = None,
) -> B2ReloadedCandidate:
    """Reload and validate the exact Validation-selected checkpoint, never the fit model."""

    candidate = _validate_b2_checkpoint_lock(locked_candidate)
    checkpoint = locked_candidate.checkpoint_path
    _require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Locked B2 checkpoint is missing/empty")
    actual_first = sha256_func(checkpoint)
    actual_second = sha256_func(checkpoint)
    _require(actual_first == actual_second, "Reloaded B2 checkpoint SHA is unstable")
    _require(
        actual_first == locked_candidate.checkpoint_sha256,
        "Reloaded B2 checkpoint SHA differs from locked evidence",
    )
    loader = model_loader_validator or _load_and_validate_b2_model
    model = loader(checkpoint, candidate, locked_candidate)
    _require(model is not None, "B2 checkpoint loader returned no model")
    return B2ReloadedCandidate(
        locked_candidate=locked_candidate,
        model=model,
        checkpoint_sha256_verified=True,
    )


def evaluate_b2_validation_checkpoint(
    reloaded: B2ReloadedCandidate,
    validation_sequences: Any,
    target_scaler: Any,
    *,
    expected_count: int = B2_EXPECTED_TARGET_SEQUENCE_COUNTS["validation"],
) -> B2ValidationCheckpointEvaluation:
    """Evaluate Validation only from a SHA-verified reloaded best checkpoint."""

    _require(isinstance(reloaded, B2ReloadedCandidate), "B2 Validation requires a reload wrapper")
    _require(reloaded.checkpoint_sha256_verified, "B2 Validation checkpoint SHA is unverified")
    _validate_b2_checkpoint_lock(reloaded.locked_candidate)
    truth = np.asarray(validation_sequences.y_seq, dtype=np.float64).reshape(-1)
    _require(len(truth) == expected_count, "B2 Validation target count mismatch")
    prediction = np.asarray(
        reloaded.model.predict(
            validation_sequences.X_seq,
            batch_size=B2_BATCH_SIZE,
            verbose=0,
        ),
        dtype=np.float64,
    ).reshape(-1)
    _require(len(prediction) == expected_count, "B2 Validation prediction count mismatch")
    evaluation = evaluate_raw_prediction(truth, prediction, target_scaler)
    normalized_total = float(np.sum(np.square(truth - np.mean(truth))))
    _require(normalized_total > 0, "B2 normalized Validation target variance is zero")
    normalized_r2 = 1.0 - float(np.sum(np.square(truth - prediction))) / normalized_total
    normalized = MappingProxyType(
        {
            "mae": evaluation.normalized_mae,
            "mse": evaluation.normalized_mse,
            "rmse": evaluation.normalized_rmse,
            "r2": normalized_r2,
            "prediction_count": len(prediction),
            "negative_prediction_count": int(np.count_nonzero(prediction < 0)),
            "target_name": B2_TARGET_COLUMN,
            "scale": "normalized",
            "unit": "MinMaxScaler normalized scale",
        }
    )
    return B2ValidationCheckpointEvaluation(
        normalized_metrics=normalized,
        original_scale_metrics=evaluation.metrics,
    )


def finalize_b2_locked_candidate(
    checkpoint_lock: B2CheckpointLock,
    evaluation: B2ValidationCheckpointEvaluation,
) -> B2LockedCandidate:
    _validate_b2_checkpoint_lock(checkpoint_lock)
    _require(
        evaluation.original_scale_metrics.prediction_count
        == int(evaluation.normalized_metrics["prediction_count"]),
        "B2 Validation metric counts differ",
    )
    return B2LockedCandidate(
        **asdict(checkpoint_lock),
        validation_metrics_normalized=MappingProxyType(dict(evaluation.normalized_metrics)),
        validation_metrics_original_scale=MappingProxyType(asdict(evaluation.original_scale_metrics)),
    )


def compute_original_scale_metrics(
    y_true_original: np.ndarray,
    prediction_original: np.ndarray,
) -> B2Metrics:
    truth = np.asarray(y_true_original, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_original, dtype=np.float64).reshape(-1)
    _require(truth.shape == prediction.shape and len(truth) > 0, "B2 prediction alignment mismatch")
    _require(np.isfinite(truth).all() and np.isfinite(prediction).all(), "B2 metrics contain NaN/Inf")
    error = truth - prediction
    mse = float(np.mean(np.square(error)))
    total = float(np.sum(np.square(truth - np.mean(truth))))
    _require(total > 0, "B2 target variance is zero")
    r2 = 1.0 - float(np.sum(np.square(error))) / total
    return B2Metrics(
        mae=float(np.mean(np.abs(error))),
        mse=mse,
        rmse=math.sqrt(mse),
        r2=r2,
        prediction_count=len(prediction),
        negative_prediction_count=int(np.count_nonzero(prediction < 0)),
    )


def evaluate_raw_prediction(
    y_true_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
    target_scaler: Any,
) -> B2PredictionEvaluation:
    """Inverse-transform raw Linear outputs without clipping or post-processing."""

    truth = np.asarray(y_true_normalized, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_normalized, dtype=np.float64).reshape(-1)
    _require(truth.shape == prediction.shape and len(truth) > 0, "B2 normalized alignment mismatch")
    normalized_error = truth - prediction
    normalized_mse = float(np.mean(np.square(normalized_error)))
    truth_original = np.asarray(
        target_scaler.inverse_transform(truth.reshape(-1, 1)), dtype=np.float64
    ).reshape(-1)
    prediction_original = np.asarray(
        target_scaler.inverse_transform(prediction.reshape(-1, 1)), dtype=np.float64
    ).reshape(-1)
    return B2PredictionEvaluation(
        prediction_normalized=prediction,
        prediction_original=prediction_original,
        y_true_original=truth_original,
        metrics=compute_original_scale_metrics(truth_original, prediction_original),
        normalized_mae=float(np.mean(np.abs(normalized_error))),
        normalized_mse=normalized_mse,
        normalized_rmse=math.sqrt(normalized_mse),
    )


def compare_with_fixed_wotl(
    candidate_id: str,
    metrics: B2Metrics,
    baseline: B2Metrics,
    *,
    validation_best_val_loss: float,
) -> B2CandidateComparison:
    _require(candidate_id in B2_CANDIDATES, "Unknown B2 comparison candidate")
    _require(math.isfinite(validation_best_val_loss), "B2 validation best loss is invalid")
    mae_better = metrics.mae < baseline.mae
    mse_better = metrics.mse < baseline.mse
    rmse_better = metrics.rmse < baseline.rmse
    r2_better = metrics.r2 > baseline.r2
    errors_better = mae_better and mse_better and rmse_better
    success = errors_better and r2_better
    if success:
        classification = B2Classification.POSITIVE
    elif errors_better:
        classification = B2Classification.PARTIAL_POSITIVE
    elif (
        metrics.mae >= baseline.mae
        and metrics.mse >= baseline.mse
        and metrics.rmse >= baseline.rmse
        and metrics.r2 <= baseline.r2
    ):
        classification = B2Classification.NEGATIVE
    else:
        classification = B2Classification.MIXED
    delta_mae = metrics.mae - baseline.mae
    delta_mse = metrics.mse - baseline.mse
    delta_rmse = metrics.rmse - baseline.rmse
    delta_r2 = metrics.r2 - baseline.r2
    return B2CandidateComparison(
        candidate_id=candidate_id,
        validation_best_val_loss=float(validation_best_val_loss),
        metrics=metrics,
        delta_mae=delta_mae,
        delta_mse=delta_mse,
        delta_rmse=delta_rmse,
        delta_r2=delta_r2,
        relative_mae_change_percent=delta_mae / baseline.mae * 100.0,
        relative_mse_change_percent=delta_mse / baseline.mse * 100.0,
        relative_rmse_change_percent=delta_rmse / baseline.rmse * 100.0,
        r2_difference=delta_r2,
        mae_better=mae_better,
        mse_better=mse_better,
        rmse_better=rmse_better,
        r2_better=r2_better,
        success=success,
        classification=classification,
    )


def validate_fixed_wotl_metrics(baseline: B2Metrics) -> None:
    """Require a runtime baseline to equal the sealed Formal JSON contract."""

    actual = asdict(baseline)
    for key, expected in B2_FIXED_WOTL_METRICS.items():
        _require(actual.get(key) == expected, f"Runtime WOTL baseline mismatch: {key}")


def select_supplementary_candidate(
    comparisons: Mapping[str, B2CandidateComparison],
) -> tuple[str | None, bool]:
    _require(tuple(comparisons) == tuple(B2_CANDIDATES), "B2 comparison matrix incomplete")
    successful = [item for item in comparisons.values() if item.success]
    if not successful:
        return None, False
    selected = min(
        successful,
        key=lambda item: (item.validation_best_val_loss, item.candidate_id),
    )
    return selected.candidate_id, True


def validate_b2_test_data(data: B2TestData) -> None:
    _require(np.asarray(data.X_test).shape == B2_EXPECTED_X_TEST_SHAPE, "B2 Test X shape mismatch")
    _require(np.asarray(data.y_test).shape == B2_EXPECTED_Y_TEST_SHAPE, "B2 Test y shape mismatch")
    _require(
        np.asarray(data.target_timestamp).shape == B2_EXPECTED_Y_TEST_SHAPE,
        "B2 Test timestamp shape mismatch",
    )
    _require(hasattr(data.target_scaler, "inverse_transform"), "B2 target scaler is invalid")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _evaluate_supplementary_test_batch_unlocked(
    models: Mapping[str, Any],
    validation_best_losses: Mapping[str, float],
    baseline: B2Metrics,
    *,
    test_loader: Callable[[], B2TestData],
    disclosure_path: Path | str,
) -> B2SupplementaryBatchResult:
    """Internal raw-model evaluator; callers must first enforce the lock gate."""

    _require(tuple(models) == tuple(B2_CANDIDATES), "B2 model batch must contain only P1/P2/P3")
    validate_fixed_wotl_metrics(baseline)
    _require(
        tuple(validation_best_losses) == tuple(B2_CANDIDATES),
        "B2 validation-loss registry must contain only P1/P2/P3",
    )
    disclosure = Path(disclosure_path)
    _require(disclosure.parent.is_dir(), "B2 disclosure parent does not exist")
    _write_json_exclusive(disclosure, test_reuse_disclosure())
    load_count = 0

    def load_once() -> B2TestData:
        nonlocal load_count
        _require(load_count == 0, "B2 Test disk loader may run only once")
        value = test_loader()
        validate_b2_test_data(value)
        load_count = 1
        return value

    data = load_once()
    evaluations: dict[str, B2PredictionEvaluation] = {}
    comparisons: dict[str, B2CandidateComparison] = {}
    shared_x = True
    for candidate_id, model in models.items():
        prediction = np.asarray(
            model.predict(data.X_test, batch_size=B2_BATCH_SIZE, verbose=0),
            dtype=np.float64,
        ).reshape(-1)
        _require(len(prediction) == B2_EXPECTED_TEST_SEQUENCES, "B2 Test prediction count mismatch")
        shared_x = shared_x and bool(model.predict_calls[-1][0] is data.X_test) if hasattr(model, "predict_calls") else shared_x
        evaluation = evaluate_raw_prediction(data.y_test, prediction, data.target_scaler)
        evaluations[candidate_id] = evaluation
        comparisons[candidate_id] = compare_with_fixed_wotl(
            candidate_id,
            evaluation.metrics,
            baseline,
            validation_best_val_loss=float(validation_best_losses[candidate_id]),
        )
    selected, achieved = select_supplementary_candidate(comparisons)
    return B2SupplementaryBatchResult(
        evaluations=MappingProxyType(evaluations),
        comparisons=MappingProxyType(comparisons),
        selected_candidate_id=selected,
        b2_goal_achieved=achieved,
        test_disk_load_count=load_count,
        shared_x_test_object=shared_x,
    )


def evaluate_supplementary_test_batch(
    models: Mapping[str, Any],
    validation_best_losses: Mapping[str, float],
    baseline: B2Metrics,
    *,
    test_loader: Callable[[], B2TestData],
    disclosure_path: Path | str,
    _allow_unlocked_test_models: bool = False,
) -> B2SupplementaryBatchResult:
    """Unsafe compatibility/test helper; production must use the locked entry."""

    _require(
        _allow_unlocked_test_models,
        "Arbitrary B2 models are rejected; use evaluate_locked_supplementary_test_batch",
    )
    return _evaluate_supplementary_test_batch_unlocked(
        models,
        validation_best_losses,
        baseline,
        test_loader=test_loader,
        disclosure_path=disclosure_path,
    )


def _validate_complete_locked_candidate(locked: B2LockedCandidate) -> None:
    _require(isinstance(locked, B2LockedCandidate), "B2-1C requires finalized locked evidence")
    _validate_b2_checkpoint_lock(locked)
    normalized_required = {
        "mae",
        "mse",
        "rmse",
        "r2",
        "prediction_count",
        "negative_prediction_count",
        "target_name",
        "scale",
        "unit",
    }
    original_required = normalized_required
    _require(
        set(locked.validation_metrics_normalized) == normalized_required,
        "Locked B2 normalized Validation evidence is incomplete",
    )
    _require(
        set(locked.validation_metrics_original_scale) == original_required,
        "Locked B2 original Validation evidence is incomplete",
    )
    _require(
        locked.validation_metrics_normalized["scale"] == "normalized",
        "Locked B2 normalized metric scale changed",
    )
    _require(
        locked.validation_metrics_original_scale["scale"] == "original",
        "Locked B2 original metric scale changed",
    )


def evaluate_locked_supplementary_test_batch(
    locked_candidates: Mapping[str, B2LockedCandidate],
    baseline: B2Metrics,
    *,
    test_loader: Callable[[], B2TestData],
    disclosure_path: Path | str,
) -> B2SupplementaryBatchResult:
    """B2-1C safe entry: validate/reload all locks before one revealed-Test load."""

    _require(
        tuple(locked_candidates) == tuple(B2_CANDIDATES),
        "Locked B2 registry must contain exactly P1/P2/P3 in protocol order",
    )
    validate_fixed_wotl_metrics(baseline)
    reloaded_models: dict[str, Any] = {}
    validation_losses: dict[str, float] = {}
    for candidate_id in B2_CANDIDATES:
        locked = locked_candidates[candidate_id]
        _validate_complete_locked_candidate(locked)
        _require(locked.candidate_id == candidate_id, "Locked B2 registry key/identity mismatch")
        reloaded = reload_b2_best_checkpoint(locked)
        _require(
            isinstance(reloaded, B2ReloadedCandidate)
            and reloaded.checkpoint_sha256_verified,
            "B2-1C checkpoint reload evidence is invalid",
        )
        _require(
            reloaded.locked_candidate == locked,
            "B2-1C reloader returned a different candidate lock",
        )
        reloaded_models[candidate_id] = reloaded.model
        validation_losses[candidate_id] = locked.best_val_loss
    return _evaluate_supplementary_test_batch_unlocked(
        MappingProxyType(reloaded_models),
        MappingProxyType(validation_losses),
        baseline,
        test_loader=test_loader,
        disclosure_path=disclosure_path,
    )


def initialize_b2_run(
    paths: B2PathContract,
    manifest: Mapping[str, Any],
    baseline: B2Metrics,
) -> None:
    """Create a new B2 namespace only; collisions are never overwritten."""

    validate_new_b2_destination(paths)
    paths.run_root.mkdir(parents=True, exist_ok=False)
    paths.baseline_root.mkdir()
    paths.candidates_root.mkdir()
    paths.summary_root.mkdir()
    _write_json_exclusive(paths.run_manifest, manifest)
    _write_json_exclusive(paths.wotl_baseline, asdict(baseline))
    for candidate_id, candidate_root in paths.candidate_roots.items():
        candidate_root.mkdir()
        candidate = B2_CANDIDATES[candidate_id]
        _write_json_exclusive(candidate_root / "config.json", asdict(candidate))


def write_candidate_validation_results(
    paths: B2PathContract,
    evidence: Mapping[str, Any],
    history: Mapping[str, Sequence[float]],
) -> None:
    """Persist one candidate's Validation-only training evidence."""

    candidate_id = str(evidence.get("candidate_id", ""))
    _require(candidate_id in B2_CANDIDATES, "Unknown B2 candidate evidence")
    _require(evidence.get("selection_split") == "validation", "B2 best epoch is not Validation-selected")
    _require(evidence.get("test_used_for_epoch_selection") is False, "B2 Test selected an epoch")
    candidate_root = paths.candidate_roots[candidate_id]
    _require(candidate_root.is_dir(), "B2 candidate root is not initialized")
    keys = tuple(history)
    lengths = {len(tuple(history[key])) for key in keys}
    _require(keys and len(lengths) == 1, "B2 history is malformed")
    with (candidate_root / "training_history.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", *keys))
        for index in range(next(iter(lengths))):
            writer.writerow((index + 1, *(history[key][index] for key in keys)))
    _write_json_exclusive(
        candidate_root / "best_checkpoint.json",
        {
            "best_epoch": evidence["best_epoch"],
            "best_val_loss": evidence["best_val_loss"],
            "checkpoint_path": evidence["checkpoint_path"],
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "checkpoint_sha256_verified": evidence["checkpoint_sha256_verified"],
            "selection_split": evidence["selection_split"],
            "test_used_for_epoch_selection": evidence["test_used_for_epoch_selection"],
        },
    )
    _write_json_exclusive(
        candidate_root / "validation_metrics_original_scale.json",
        {
            "candidate_id": candidate_id,
            "validation_metrics": evidence["validation_metrics"],
            "target_name": "DC_POWER",
            "scale": "original",
            "unit": "DC_POWER (raw dataset scale; documented unit: kW)",
        },
    )


def _checkpoint_lock_mapping(locked: B2CheckpointLock) -> Mapping[str, Any]:
    return {
        "candidate_id": locked.candidate_id,
        "strategy": locked.strategy,
        "learning_rate": locked.learning_rate,
        "best_epoch": locked.best_epoch,
        "best_val_loss": locked.best_val_loss,
        "checkpoint_path": locked.checkpoint_path,
        "checkpoint_sha256": locked.checkpoint_sha256,
        "checkpoint_sha256_verified": locked.checkpoint_sha256_verified,
        "selection_split": locked.selection_split,
        "test_used_for_epoch_selection": locked.test_used_for_epoch_selection,
        "transferred_layer_indices": locked.transferred_layer_indices,
        "trainable_layer_indices": locked.trainable_layer_indices,
        "frozen_layer_indices": locked.frozen_layer_indices,
        "transferred_layer_names": locked.transferred_layer_names,
        "trainable_layer_names": locked.trainable_layer_names,
        "frozen_layer_names": locked.frozen_layer_names,
        "trainable_params": locked.trainable_params,
        "total_params": locked.total_params,
        "trainable_ratio": locked.trainable_ratio,
        "execution_git_head": locked.execution_git_head,
        "run_root": locked.run_root,
    }


def locked_candidate_mapping(locked: B2LockedCandidate) -> Mapping[str, Any]:
    _validate_complete_locked_candidate(locked)
    return MappingProxyType(
        {
            **_checkpoint_lock_mapping(locked),
            "validation_metrics_normalized": dict(locked.validation_metrics_normalized),
            "validation_metrics_original_scale": dict(
                locked.validation_metrics_original_scale
            ),
        }
    )


def write_locked_candidate_validation_results(
    paths: B2PathContract,
    locked: B2LockedCandidate,
    history: Mapping[str, Sequence[float]],
) -> None:
    """Persist evidence sourced only from the reloaded Validation-best model."""

    _validate_complete_locked_candidate(locked)
    candidate_root = paths.candidate_roots[locked.candidate_id]
    _require(candidate_root.is_dir(), "B2 candidate root is not initialized")
    _require(locked.run_root.resolve() == paths.run_root.resolve(), "B2 lock/run path mismatch")
    keys = tuple(history)
    lengths = {len(tuple(history[key])) for key in keys}
    _require(keys and len(lengths) == 1, "B2 history is malformed")
    with (candidate_root / "training_history.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", *keys))
        for index in range(next(iter(lengths))):
            writer.writerow((index + 1, *(history[key][index] for key in keys)))
    _write_json_exclusive(
        candidate_root / "best_checkpoint.json",
        {
            key: value
            for key, value in _checkpoint_lock_mapping(locked).items()
            if key
            in {
                "best_epoch",
                "best_val_loss",
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_sha256_verified",
                "selection_split",
                "test_used_for_epoch_selection",
                "execution_git_head",
            }
        },
    )
    _write_json_exclusive(
        candidate_root / "validation_metrics_normalized.json",
        dict(locked.validation_metrics_normalized),
    )
    _write_json_exclusive(
        candidate_root / "validation_metrics_original_scale.json",
        dict(locked.validation_metrics_original_scale),
    )
    _write_json_exclusive(candidate_root / "locked_candidate.json", locked_candidate_mapping(locked))


def write_b2_validation_registry(
    paths: B2PathContract,
    locked_candidates: Mapping[str, B2LockedCandidate],
) -> Path:
    _require(
        tuple(locked_candidates) == tuple(B2_CANDIDATES),
        "B2 Validation registry must contain exactly P1/P2/P3",
    )
    registry_path = paths.summary_root / "validation_registry.json"
    _write_json_exclusive(
        registry_path,
        {
            "experiment": B2_EXPERIMENT_ID,
            "selection_split": "validation",
            "test_used_for_epoch_selection": False,
            "candidate_order": tuple(B2_CANDIDATES),
            "candidates": {
                candidate_id: locked_candidate_mapping(locked_candidates[candidate_id])
                for candidate_id in B2_CANDIDATES
            },
            "training_validation_complete": True,
            "supplementary_test_executed": False,
            "automatic_b2_2_started": False,
        },
    )
    return registry_path


def _default_b2_runner_dependencies() -> B2RunnerDependencies:
    return B2RunnerDependencies(
        git_collector=collect_b2_execution_git_provenance,
        runtime_gate=enforce_b2_cpu_seed_determinism,
        path_factory=lambda run_id, output_base: b2_path_contract(
            run_id, output_base=output_base
        ),
        baseline_loader=load_fixed_wotl_baseline,
        source_checkpoint_validator=validate_locked_source_checkpoint,
        source_model_loader=load_locked_source_model,
        profile_loader=load_b2_target_training_validation,
        run_initializer=initialize_b2_run,
        candidate_builder=build_b2_candidate_model,
        candidate_fitter=fit_b2_candidate_validation_only,
        checkpoint_reloader=reload_b2_best_checkpoint,
        validation_evaluator=evaluate_b2_validation_checkpoint,
        candidate_writer=write_locked_candidate_validation_results,
    )


def run_b2_training_validation(
    run_id: str,
    expected_execution_git_head: str,
    *,
    output_base: Path | str = B2_OUTPUT_BASE,
    _dependencies: B2RunnerDependencies | None = None,
    _allow_fake_dependencies: bool = False,
) -> B2TrainingValidationResult:
    """Run the fixed P1/P2/P3 Training/Validation lifecycle and stop."""

    validate_b2_config()
    _require(
        _dependencies is None or _allow_fake_dependencies,
        "B2 runner dependency injection is disabled for production",
    )
    dependencies = _dependencies or _default_b2_runner_dependencies()
    git = dependencies.git_collector()
    validate_b2_execution_git_provenance(git, expected_git_head=expected_execution_git_head)
    environment = dependencies.runtime_gate()
    paths = dependencies.path_factory(run_id, output_base)
    validate_new_b2_destination(paths)
    baseline = dependencies.baseline_loader()
    dependencies.source_checkpoint_validator()
    source_model = dependencies.source_model_loader()
    accessed_files: list[Path] = []
    profile = dependencies.profile_loader(accessed_files)
    manifest = dict(
        build_run_manifest(
            run_id=run_id,
            execution_git_head=git.head,
            tracked_dirty=git.tracked_dirty,
            untracked_paths=git.untracked_paths,
        )
    )
    manifest["execution_environment"] = dict(environment)
    dependencies.run_initializer(paths, manifest, baseline)

    locked_candidates: dict[str, B2LockedCandidate] = {}
    total_epochs = 0
    for candidate_id in B2_CANDIDATES:
        contract = dependencies.candidate_builder(
            source_model,
            candidate_id,
            output_dir=paths.candidate_roots[candidate_id],
        )
        fit = dependencies.candidate_fitter(
            contract,
            profile.training.sequences,
            profile.validation.sequences,
            checkpoint_directory=paths.candidate_roots[candidate_id] / "checkpoints",
        )
        checkpoint_lock = lock_b2_best_checkpoint(
            contract,
            fit,
            execution_git_head=git.head,
            run_root=paths.run_root,
        )
        reloaded = dependencies.checkpoint_reloader(checkpoint_lock)
        validation_evaluation = dependencies.validation_evaluator(
            reloaded,
            profile.validation.sequences,
            profile.scalers.target_scaler,
        )
        locked = finalize_b2_locked_candidate(checkpoint_lock, validation_evaluation)
        dependencies.candidate_writer(paths, locked, fit.history)
        locked_candidates[candidate_id] = locked
        total_epochs += len(fit.history["val_loss"])

    immutable_registry = MappingProxyType(dict(locked_candidates))
    registry_path = write_b2_validation_registry(paths, immutable_registry)
    return B2TrainingValidationResult(
        paths=paths,
        locked_candidates=immutable_registry,
        validation_registry_path=registry_path,
        execution_git=git,
        accessed_files=tuple(accessed_files),
        total_training_epochs_executed=total_epochs,
    )


def write_supplementary_batch_results(
    paths: B2PathContract,
    data: B2TestData,
    result: B2SupplementaryBatchResult,
    baseline: B2Metrics,
) -> None:
    """Write only supplementary evidence beneath an initialized B2 run root."""

    _require(paths.run_root.is_dir(), "B2 run root is not initialized")
    _require(tuple(result.evaluations) == tuple(B2_CANDIDATES), "B2 result matrix incomplete")
    for candidate_id, evaluation in result.evaluations.items():
        candidate_root = paths.candidate_roots[candidate_id]
        _write_json_exclusive(
            candidate_root / "test_metrics_original_scale.json", asdict(evaluation.metrics)
        )
        with (candidate_root / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "y_true_original", "prediction_original", "residual"))
            for index, timestamp in enumerate(data.target_timestamp):
                truth = evaluation.y_true_original[index]
                prediction = evaluation.prediction_original[index]
                writer.writerow((str(timestamp), truth, prediction, truth - prediction))
    _write_json_exclusive(
        paths.comparison,
        {
            "experiment": B2_EXPERIMENT_ID,
            **dict(B2_DISCLOSURE),
            "formal_classification_preserved": B2_FORMAL_CLASSIFICATION,
            "fixed_wotl_baseline": asdict(baseline),
            "candidate_comparisons": {
                key: asdict(value) for key, value in result.comparisons.items()
            },
        },
    )
    _write_json_exclusive(
        paths.results_summary,
        {
            "experiment": B2_EXPERIMENT_ID,
            "b2_goal_achieved": result.b2_goal_achieved,
            "selected_candidate_id": result.selected_candidate_id,
            "test_disk_load_count": result.test_disk_load_count,
            "formal_final_test_replacement": False,
            "automatic_b2_2_started": False,
        },
    )


def candidate_validation_evidence(
    contract: B2ModelContract,
    fit: B2FitEvidence,
    validation_metrics: Mapping[str, float | int],
) -> Mapping[str, Any]:
    """Build the complete Validation-only candidate evidence record."""

    _require(fit.candidate_id == contract.candidate.candidate_id, "B2 candidate evidence mismatch")
    required_metrics = {
        "normalized_mae",
        "normalized_mse",
        "normalized_rmse",
        "normalized_r2",
        "original_mae",
        "original_mse",
        "original_rmse",
        "original_r2",
    }
    _require(
        required_metrics.issubset(validation_metrics),
        "B2 Validation normalized/original metrics are incomplete",
    )
    return MappingProxyType(
        {
            "candidate_id": contract.candidate.candidate_id,
            "strategy": contract.candidate.strategy,
            "learning_rate": contract.candidate.learning_rate,
            "transferred_layer_indices": contract.candidate.transferred_indices,
            "trainable_layer_indices": contract.candidate.trainable_indices,
            "frozen_layer_indices": contract.candidate.frozen_indices,
            "trainable_layer_names": contract.trainable_layer_names,
            "frozen_layer_names": contract.frozen_layer_names,
            "trainable_params": contract.trainable_params,
            "total_params": contract.total_params,
            "trainable_ratio": contract.trainable_ratio,
            "best_epoch": fit.best_epoch,
            "best_val_loss": fit.best_val_loss,
            "validation_metrics": dict(validation_metrics),
            "checkpoint_path": fit.checkpoint_path,
            "checkpoint_sha256": fit.checkpoint_sha256,
            "checkpoint_sha256_verified": fit.checkpoint_sha256_verified,
            "selection_split": fit.selection_split,
            "test_used_for_epoch_selection": fit.test_used_for_epoch_selection,
        }
    )
