"""Phase IV Formal Training + Validation runner for Solar Linear experiments.

Importing this module never starts training or creates a Formal directory.
Real execution requires an explicit boolean and ``SOLAR_RUN_FORMAL=1``.  The
module has no Target/Source Test loader and exposes no Final-Test operation.
"""

from __future__ import annotations

import os

# CPU visibility must be fixed before importing TensorFlow/Keras, including
# through the Phase-I model runtime.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import contextlib
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence
from unittest.mock import patch

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import load_model

from r2_config.solar_linear import (
    EXPECTED_TOTAL_PARAMS,
    EXPECTED_TRAINABLE_PARAMS,
    LINEAR_BATCH_SIZE,
    LINEAR_EXPERIMENTS,
    LINEAR_SEED,
    REPOSITORY_ROOT,
    LinearExperimentSpec,
)
from r2_config.solar_linear_formal import (
    FORMAL_CALLBACK_POLICY,
    FORMAL_DEVICE_POLICY,
    FORMAL_PROTOCOL_VERSION,
    RUN_ID_PATTERN,
    FormalPathContract,
    candidate_registry,
    formal_path_contract,
    validate_new_run_destination,
)
from r2_helpers import solar_data
from r2_helpers.solar_data import TrainingValidationProfile, TrainingValidationSplit
from r2_helpers.solar_linear_formal import (
    GitIdentity,
    ProtocolStage,
    SelectionRecord,
    SourceArchitectureEvidence,
    SourceCheckpointProvenance,
    ValidationCandidateScore,
    build_protocol_manifest,
    build_selection_record,
    build_source_checkpoint_provenance_from_selection,
    make_formal_callbacks,
    selection_record_mapping,
    select_validation_candidate,
    target_protocol_fingerprint,
    validate_protocol_manifest,
    validate_selection_record,
    validate_validation_candidate,
)
from r2_helpers.solar_linear_runtime import (
    build_linear_partial_ft_candidate,
    build_linear_source_model,
    build_linear_without_tl_model,
    rmse,
    validate_linear_model,
)


EXPECTED_ENVIRONMENT = MappingProxyType(
    {
        "python": "3.10.11",
        "tensorflow": "2.10.0",
        "keras": "2.10.0",
        "numpy": "1.23.5",
    }
)
FORBIDDEN_SPLIT_FILENAMES = frozenset(
    f"{scale}_scale_{split}.csv"
    for scale in ("normalized", "original")
    for split in ("test",)
)
KNOWN_UNTRACKED_PREFIXES = (
    "notebook/修改紀錄/",
    "notebook/執行紀錄/",
    "reports/Solar Energy Result/R2/Experiment_A/20260814T103322Z_seed1234/",
)


class FormalTrainingError(AssertionError):
    """Raised when a Phase-IV training/validation invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalTrainingError(message)


@dataclass(frozen=True)
class FormalEnvironment:
    python: str
    tensorflow: str
    keras: str
    numpy: str
    device: str
    cuda_visible_devices: str
    pythonhashseed: str
    seed: int
    deterministic_ops: bool


@dataclass(frozen=True)
class FormalGitProvenance:
    head: str
    branch: str
    tracked_dirty: bool
    known_untracked_paths: tuple[str, ...]
    unknown_untracked_paths: tuple[str, ...]

    @property
    def known_untracked_present(self) -> bool:
        return bool(self.known_untracked_paths)


@dataclass(frozen=True)
class FormalCandidatePlan:
    lifecycle: str
    candidate_id: str
    learning_rate: float
    candidate_root: Path
    checkpoint_pattern: Path


@dataclass(frozen=True)
class FormalTrainingPlan:
    experiment_id: str
    run_id: str
    spec: LinearExperimentSpec
    paths: FormalPathContract
    git: FormalGitProvenance
    environment: FormalEnvironment
    candidates: Mapping[str, tuple[FormalCandidatePlan, ...]]
    training_epochs_executed: int = 0


@dataclass(frozen=True)
class ValidationMetrics:
    normalized_loss: float
    normalized_mae: float
    normalized_rmse: float
    normalized_r2: float
    original_mae: float
    original_mse: float
    original_rmse: float
    original_r2: float
    prediction_count: int
    unit: str = "DC_POWER (raw dataset scale; documented unit: kW)"


@dataclass(frozen=True)
class BestCheckpoint:
    best_epoch: int
    best_val_loss: float
    path: Path
    sha256: str
    sha256_verified: bool


@dataclass(frozen=True)
class PartialFTStateAudit:
    layer1_changed: bool
    layer2_unchanged: bool
    bn3_unchanged: bool
    layer4_changed: bool
    bn5_unchanged: bool
    layer6_changed: bool


@dataclass(frozen=True)
class CandidateRunResult:
    plan: FormalCandidatePlan
    completed_epochs: int
    score: ValidationCandidateScore
    metrics: ValidationMetrics
    checkpoint: BestCheckpoint
    reloaded_model: Any
    state_audit: PartialFTStateAudit | None


@dataclass(frozen=True)
class FormalTrainingResult:
    plan: FormalTrainingPlan
    source_results: tuple[CandidateRunResult, ...]
    wotl_results: tuple[CandidateRunResult, ...]
    partial_ft_results: tuple[CandidateRunResult, ...]
    source_provenance: SourceCheckpointProvenance
    selection: SelectionRecord
    selection_path: Path
    manifest_path: Path
    accessed_files: tuple[Path, ...]
    total_training_epochs_executed: int


class LearningRateRecorder(Callback):
    """Passive callback that records the optimizer LR at each epoch start."""

    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        del epoch, logs
        value = getattr(self.model.optimizer, "learning_rate", self.model.optimizer.lr)
        if hasattr(value, "numpy"):
            value = value.numpy()
        self.values.append(float(value))


def collect_formal_environment() -> FormalEnvironment:
    return FormalEnvironment(
        python=platform.python_version(),
        tensorflow=tf.__version__,
        keras=keras.__version__,
        numpy=np.__version__,
        device="CPU",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        pythonhashseed=os.environ.get("PYTHONHASHSEED", ""),
        seed=LINEAR_SEED,
        deterministic_ops=FORMAL_DEVICE_POLICY.deterministic_ops,
    )


def validate_formal_environment(environment: FormalEnvironment) -> None:
    for name, expected in EXPECTED_ENVIRONMENT.items():
        _require(getattr(environment, name) == expected, f"Formal {name} version mismatch")
    _require(environment.device == "CPU", "Formal device must be CPU")
    _require(environment.cuda_visible_devices == "-1", "CUDA must be disabled before import")
    _require(
        environment.pythonhashseed == str(LINEAR_SEED),
        "PYTHONHASHSEED must be fixed before process startup",
    )
    _require(environment.seed == LINEAR_SEED, "Formal seed changed")
    _require(environment.deterministic_ops, "Deterministic ops policy is disabled")


def set_reproducibility(seed: int = LINEAR_SEED) -> None:
    _require(seed == LINEAR_SEED, "Formal seed must remain 1234")
    _require(os.environ.get("PYTHONHASHSEED") == str(seed), "Hash seed was not pre-set")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    enable = getattr(tf.config.experimental, "enable_op_determinism", None)
    _require(callable(enable), "TensorFlow deterministic ops API unavailable")
    enable()
    _require(not tf.config.list_physical_devices("GPU"), "TensorFlow exposes a GPU")


def _git_bytes(*arguments: str) -> bytes:
    safe_arg = f"safe.directory={REPOSITORY_ROOT.as_posix()}"
    completed = subprocess.run(
        ["git", "-c", safe_arg, *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    _require(completed.returncode == 0, f"Git command failed: {' '.join(arguments)}")
    return completed.stdout


def _untracked_paths() -> tuple[str, ...]:
    raw = _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries = raw.decode("utf-8").split("\0")
    return tuple(entry[3:].replace("\\", "/") for entry in entries if entry.startswith("?? "))


def collect_formal_git_provenance() -> FormalGitProvenance:
    head = _git_bytes("rev-parse", "HEAD").strip().decode("ascii")
    branch = _git_bytes("branch", "--show-current").strip().decode("utf-8")
    tracked_dirty = bool(
        _git_bytes("status", "--porcelain=v1", "--untracked-files=no")
    )
    untracked = _untracked_paths()
    known = tuple(
        path for path in untracked if any(path.startswith(prefix) for prefix in KNOWN_UNTRACKED_PREFIXES)
    )
    unknown = tuple(path for path in untracked if path not in known)
    return FormalGitProvenance(
        head=head,
        branch=branch,
        tracked_dirty=tracked_dirty,
        known_untracked_paths=known,
        unknown_untracked_paths=unknown,
    )


def validate_formal_git_provenance(
    provenance: FormalGitProvenance,
    *,
    expected_git_head: str,
) -> None:
    _require(provenance.head == expected_git_head, "Formal Git HEAD mismatch")
    _require(bool(provenance.branch), "Detached/unknown Formal Git branch")
    _require(not provenance.tracked_dirty, "Tracked working tree is dirty")
    _require(not provenance.unknown_untracked_paths, "Unexpected untracked paths present")


def candidate_root(
    paths: FormalPathContract,
    lifecycle: str,
    candidate_id: str,
) -> Path:
    bases = {
        "source": paths.source_candidates,
        "wotl": paths.target_wotl_candidates,
        "partial_ft": paths.target_partial_ft_candidates,
    }
    _require(lifecycle in bases, "Unknown Formal candidate lifecycle")
    registered = dict(candidate_registry()[lifecycle])
    _require(candidate_id in registered, "Candidate is not registered")
    root = bases[lifecycle] / candidate_id
    _require(root.resolve().is_relative_to(paths.run_root.resolve()), "Candidate root escaped run")
    return root


def _candidate_plans(paths: FormalPathContract) -> Mapping[str, tuple[FormalCandidatePlan, ...]]:
    plans: dict[str, tuple[FormalCandidatePlan, ...]] = {}
    for lifecycle, registered in candidate_registry().items():
        items = []
        for identifier, learning_rate in registered:
            root = candidate_root(paths, lifecycle, identifier)
            items.append(
                FormalCandidatePlan(
                    lifecycle=lifecycle,
                    candidate_id=identifier,
                    learning_rate=learning_rate,
                    candidate_root=root,
                    checkpoint_pattern=root / "checkpoint_epoch_{epoch:04d}.hdf5",
                )
            )
        plans[lifecycle] = tuple(items)
    return MappingProxyType(plans)


def prepare_formal_training_run(
    experiment_id: str,
    run_id: str,
    *,
    expected_git_head: str,
    git_provenance: FormalGitProvenance | None = None,
    environment: FormalEnvironment | None = None,
    path_factory: Callable[[str, str], FormalPathContract] = formal_path_contract,
    destination_validator: Callable[[FormalPathContract], None] = validate_new_run_destination,
) -> FormalTrainingPlan:
    _require(experiment_id in LINEAR_EXPERIMENTS, "Unknown Formal experiment")
    _require(bool(RUN_ID_PATTERN.fullmatch(run_id)), "Invalid Formal run ID")
    git = git_provenance or collect_formal_git_provenance()
    runtime = environment or collect_formal_environment()
    validate_formal_git_provenance(git, expected_git_head=expected_git_head)
    validate_formal_environment(runtime)
    paths = path_factory(experiment_id, run_id)
    destination_validator(paths)
    _require(not paths.run_root.exists(), "Formal run destination already exists")
    return FormalTrainingPlan(
        experiment_id=experiment_id,
        run_id=run_id,
        spec=LINEAR_EXPERIMENTS[experiment_id],
        paths=paths,
        git=git,
        environment=runtime,
        candidates=_candidate_plans(paths),
    )


def guard_data_path(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    _require(resolved.name not in FORBIDDEN_SPLIT_FILENAMES, f"Forbidden split access: {resolved}")
    return resolved


@contextlib.contextmanager
def record_data_access(accessed: list[Path]) -> Iterator[None]:
    original_path_open = Path.open
    original_read_csv = solar_data.pd.read_csv
    original_joblib_load = solar_data.joblib.load

    def record(path: Path | str) -> None:
        accessed.append(guard_data_path(path))

    def path_open(path: Path, *args: Any, **kwargs: Any):
        mode = args[0] if args else kwargs.get("mode", "r")
        if str(mode).startswith("r"):
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


def load_role_training_validation(
    spec: LinearExperimentSpec,
    role: str,
    accessed: list[Path],
) -> TrainingValidationProfile:
    _require(role in ("source", "target"), "Unknown Formal data role")
    profile_path = spec.source_profile_path if role == "source" else spec.target_profile_path
    plant = spec.source_plant if role == "source" else spec.target_plant
    with record_data_access(accessed):
        profile = solar_data.load_training_validation_profile(
            profile_path,
            expected_plant=plant,
            expected_profile=role,
            expected_profile_dir_name=f"{role}_profile",
        )
    validate_profile_counts(profile, role)
    return profile


def validate_profile_counts(profile: TrainingValidationProfile, role: str) -> None:
    expected = {
        "source": (2088, 523, 2083, 518),
        "target": (521, 131, 516, 126),
    }
    _require(role in expected, "Unknown profile-count role")
    actual = (
        profile.training.split_data.row_count,
        profile.validation.split_data.row_count,
        profile.training.sequences.count,
        profile.validation.sequences.count,
    )
    _require(actual == expected[role], f"Formal {role} row/sequence count mismatch: {actual}")


def build_candidate_model(
    plan: FormalCandidatePlan,
    *,
    locked_source_model: Any | None = None,
) -> Any:
    if plan.lifecycle == "source":
        _require(locked_source_model is None, "Source candidate cannot depend on Source")
        return build_linear_source_model(
            output_dir=plan.candidate_root,
            learning_rate=plan.learning_rate,
        )
    if plan.lifecycle == "wotl":
        _require(locked_source_model is None, "WOTL candidate cannot depend on Source")
        return build_linear_without_tl_model(
            output_dir=plan.candidate_root,
            learning_rate=plan.learning_rate,
        )
    _require(plan.lifecycle == "partial_ft", "Unknown model lifecycle")
    _require(locked_source_model is not None, "Partial FT requires the locked Source")
    return build_linear_partial_ft_candidate(
        locked_source_model,
        output_dir=plan.candidate_root,
        learning_rate=plan.learning_rate,
    ).model


def _history_mapping(history_object: Any) -> Mapping[str, tuple[float, ...]]:
    raw = history_object.history if hasattr(history_object, "history") else history_object
    _require(isinstance(raw, Mapping) and raw, "Formal fit returned no history")
    history = {
        str(key): tuple(float(value) for value in values) for key, values in raw.items()
    }
    lengths = {len(values) for values in history.values()}
    _require(len(lengths) == 1 and next(iter(lengths)) > 0, "Formal history lengths differ")
    _require(
        all(math.isfinite(value) for values in history.values() for value in values),
        "Formal history contains NaN/Inf",
    )
    _require("val_loss" in history, "Formal history lacks val_loss")
    return MappingProxyType(history)


def _fit_formal_candidate(
    model: Any,
    profile: TrainingValidationProfile,
    checkpoint_pattern: Path,
    *,
    callback_factory: Callable[[Path | str], tuple[Any, ...]] = make_formal_callbacks,
) -> tuple[Mapping[str, tuple[float, ...]], tuple[float, ...]]:
    callbacks = callback_factory(checkpoint_pattern)
    _require(len(callbacks) == 4, "Formal callback factory must return four callbacks")
    recorder = LearningRateRecorder()
    history_object = model.fit(
        profile.training.sequences.X_seq,
        profile.training.sequences.y_seq,
        validation_data=(
            profile.validation.sequences.X_seq,
            profile.validation.sequences.y_seq,
        ),
        epochs=FORMAL_CALLBACK_POLICY.maximum_epochs,
        batch_size=LINEAR_BATCH_SIZE,
        shuffle=False,
        callbacks=[*callbacks, recorder],
        verbose=2,
    )
    history = _history_mapping(history_object)
    _require(
        len(next(iter(history.values()))) <= FORMAL_CALLBACK_POLICY.maximum_epochs,
        "Formal fit exceeded maximum epochs",
    )
    completed_epochs = len(next(iter(history.values())))
    _require(
        len(recorder.values) == completed_epochs,
        "Learning-rate history length differs from completed epochs",
    )
    _require(
        all(math.isfinite(value) for value in recorder.values),
        "Learning-rate history contains NaN/Inf",
    )
    return history, tuple(recorder.values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_best_checkpoint(
    history: Mapping[str, Sequence[float]],
    candidate_directory: Path | str,
) -> BestCheckpoint:
    values = np.asarray(history.get("val_loss", ()), dtype=float)
    _require(values.ndim == 1 and len(values) > 0, "Cannot select checkpoint without val_loss")
    _require(np.all(np.isfinite(values)), "Cannot select checkpoint from non-finite val_loss")
    best_epoch = int(np.argmin(values)) + 1
    path = Path(candidate_directory) / f"checkpoint_epoch_{best_epoch:04d}.hdf5"
    _require(path.is_file() and path.stat().st_size > 0, "Best checkpoint missing/empty")
    first = _sha256(path)
    second = _sha256(path)
    _require(first == second, "Checkpoint SHA verification failed")
    return BestCheckpoint(
        best_epoch=best_epoch,
        best_val_loss=float(values[best_epoch - 1]),
        path=path,
        sha256=first,
        sha256_verified=True,
    )


def reload_best_checkpoint(checkpoint: BestCheckpoint) -> Any:
    _require(_sha256(checkpoint.path) == checkpoint.sha256, "Checkpoint changed before reload")
    model = load_model(str(checkpoint.path), custom_objects={"rmse": rmse})
    # A full-model checkpoint legitimately preserves a ReduceLROnPlateau-
    # reduced optimizer LR.  Candidate initial LR remains locked separately.
    validate_linear_model(model)
    return model


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum(np.square(y_true - y_pred)))
    total = float(np.sum(np.square(y_true - np.mean(y_true))))
    _require(total > 0, "Validation target variance is zero")
    return 1.0 - residual / total


def evaluate_validation_candidate(
    model: Any,
    validation: TrainingValidationSplit,
    target_scaler: Any,
) -> ValidationMetrics:
    y_true = np.asarray(validation.sequences.y_seq, dtype=float).reshape(-1)
    prediction = np.asarray(
        model.predict(validation.sequences.X_seq, batch_size=LINEAR_BATCH_SIZE, verbose=0),
        dtype=float,
    ).reshape(-1)
    _require(len(prediction) == len(y_true) > 0, "Validation prediction alignment mismatch")
    normalized_error = y_true - prediction
    normalized_mse = float(np.mean(np.square(normalized_error)))
    normalized_mae = float(np.mean(np.abs(normalized_error)))
    normalized_rmse = math.sqrt(normalized_mse)
    original_true = np.asarray(
        target_scaler.inverse_transform(y_true.reshape(-1, 1)), dtype=float
    ).reshape(-1)
    original_prediction = np.asarray(
        target_scaler.inverse_transform(prediction.reshape(-1, 1)), dtype=float
    ).reshape(-1)
    _require(original_true.shape == original_prediction.shape, "Original-scale shape mismatch")
    original_error = original_true - original_prediction
    original_mse = float(np.mean(np.square(original_error)))
    return ValidationMetrics(
        normalized_loss=normalized_mse,
        normalized_mae=normalized_mae,
        normalized_rmse=normalized_rmse,
        normalized_r2=_r2(y_true, prediction),
        original_mae=float(np.mean(np.abs(original_error))),
        original_mse=original_mse,
        original_rmse=math.sqrt(original_mse),
        original_r2=_r2(original_true, original_prediction),
        prediction_count=len(prediction),
    )


def build_validation_score(
    *,
    formal_plan: FormalTrainingPlan,
    candidate_plan: FormalCandidatePlan,
    checkpoint: BestCheckpoint,
    metrics: ValidationMetrics,
    trainable_params: int,
) -> ValidationCandidateScore:
    score = ValidationCandidateScore(
        protocol_version=FORMAL_PROTOCOL_VERSION,
        experiment_id=formal_plan.experiment_id,
        run_id=formal_plan.run_id,
        git_head=formal_plan.git.head,
        target_protocol_fingerprint=target_protocol_fingerprint(formal_plan.experiment_id),
        lifecycle=candidate_plan.lifecycle,
        candidate_id=candidate_plan.candidate_id,
        learning_rate=candidate_plan.learning_rate,
        best_epoch=checkpoint.best_epoch,
        validation_loss=checkpoint.best_val_loss,
        validation_original_mae=metrics.original_mae,
        validation_original_rmse=metrics.original_rmse,
        validation_original_r2=metrics.original_r2,
        trainable_params=trainable_params,
        checkpoint_path=checkpoint.path,
        checkpoint_sha256=checkpoint.sha256,
        checkpoint_sha256_verified=checkpoint.sha256_verified,
        validation_only=True,
        test_accessed=False,
        test_metrics_used=False,
    )
    validate_validation_candidate(score)
    return score


def snapshot_weight_layers(model: Any) -> Mapping[int, tuple[np.ndarray, ...]]:
    return MappingProxyType(
        {
            index: tuple(weight.copy() for weight in model.layers[index].get_weights())
            for index in (1, 2, 3, 4, 5, 6)
        }
    )


def _all_equal(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(first, second) for first, second in zip(left, right)
    )


def _any_changed(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> bool:
    return len(left) == len(right) and any(
        not np.array_equal(first, second) for first, second in zip(left, right)
    )


def audit_partial_ft_state(
    before: Mapping[int, tuple[np.ndarray, ...]],
    after: Mapping[int, tuple[np.ndarray, ...]],
) -> PartialFTStateAudit:
    _require(set(before) == set(after) == {1, 2, 3, 4, 5, 6}, "PFT audit layer set mismatch")
    audit = PartialFTStateAudit(
        layer1_changed=_any_changed(before[1], after[1]),
        layer2_unchanged=_all_equal(before[2], after[2]),
        bn3_unchanged=_all_equal(before[3], after[3]),
        layer4_changed=_any_changed(before[4], after[4]),
        bn5_unchanged=_all_equal(before[5], after[5]),
        layer6_changed=_any_changed(before[6], after[6]),
    )
    _require(all(asdict(audit).values()), "Partial FT state audit failed")
    return audit


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ProtocolStage):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not temporary.exists(), "Manifest temporary file already exists")
    _write_json_exclusive(temporary, payload)
    temporary.replace(path)


def _write_history(path: Path, history: Mapping[str, Sequence[float]]) -> None:
    keys = tuple(history)
    epochs = len(next(iter(history.values())))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch",) + keys)
        for index in range(epochs):
            writer.writerow((index + 1,) + tuple(history[key][index] for key in keys))


def _write_lr_history(path: Path, values: Sequence[float]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "learning_rate"))
        for index, value in enumerate(values, start=1):
            writer.writerow((index, value))


def formal_manifest_for_training(
    plan: FormalTrainingPlan,
    base_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest = _jsonable(base_manifest)
    _require(manifest["run_id"] == plan.run_id, "Manifest/plan run mismatch")
    _require(manifest["experiment_id"] == plan.experiment_id, "Manifest/plan experiment mismatch")
    manifest.update(
        {
            "dry_run": False,
            "formal_eligible": True,
            "config_locked": True,
            "selection_locked": False,
            "checkpoint_locked": False,
            "test_accessed": False,
            "test_metrics_used_for_selection": False,
            "test_authorized": False,
            "post_test_tuning_allowed": False,
            "protocol_stage": ProtocolStage.CONFIG_LOCKED_AWAITING_TRAINING.value,
            "tracked_dirty": plan.git.tracked_dirty,
            "known_untracked_present": plan.git.known_untracked_present,
            "known_untracked_paths": plan.git.known_untracked_paths,
            "unknown_untracked_paths": plan.git.unknown_untracked_paths,
        }
    )
    validate_protocol_manifest(manifest)
    return MappingProxyType(manifest)


def lock_manifest_after_selection(
    manifest: Mapping[str, Any],
    selection: SelectionRecord,
) -> Mapping[str, Any]:
    validate_selection_record(selection)
    locked = _jsonable(manifest)
    _require(locked["experiment_id"] == selection.experiment_id, "Manifest selection experiment mismatch")
    _require(locked["run_id"] == selection.run_id, "Manifest selection run mismatch")
    locked.update(
        {
            "config_locked": True,
            "selection_locked": True,
            "checkpoint_locked": True,
            "test_accessed": False,
            "test_metrics_used_for_selection": False,
            "test_authorized": False,
            "post_test_tuning_allowed": False,
            "formal_eligible": True,
            "dry_run": False,
            "protocol_stage": ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        }
    )
    validate_protocol_manifest(locked)
    return MappingProxyType(locked)


def protocol_git_identity(plan: FormalTrainingPlan) -> GitIdentity:
    """Map runner provenance without treating known evidence as code dirtiness."""

    return GitIdentity(
        head=plan.git.head,
        branch=plan.git.branch,
        dirty=plan.git.tracked_dirty,
    )


def write_and_reload_selection(
    selection: SelectionRecord,
    selection_path: Path | str,
) -> Mapping[str, Any]:
    path = Path(selection_path)
    _write_json_exclusive(path, selection_record_mapping(selection))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_selection_record(loaded)
    return MappingProxyType(loaded)


def _source_architecture(model: Any, git_head: str) -> SourceArchitectureEvidence:
    validate_linear_model(model)
    return SourceArchitectureEvidence(
        activation=model.layers[6].activation.__name__,
        git_head=git_head,
        layer_classes=tuple(type(layer).__name__ for layer in model.layers),
        input_shape=tuple(model.input_shape[1:]),
        output_shape=tuple(model.output_shape[1:]),
        total_params=model.count_params(),
        formal_eligible=True,
    )


def _execute_candidate(
    *,
    formal_plan: FormalTrainingPlan,
    candidate_plan: FormalCandidatePlan,
    profile: TrainingValidationProfile,
    model: Any,
    candidate_accessed_files: Sequence[Path],
    locked_source_provenance: SourceCheckpointProvenance | None = None,
) -> CandidateRunResult:
    candidate_plan.candidate_root.mkdir(parents=True, exist_ok=False)
    before = snapshot_weight_layers(model) if candidate_plan.lifecycle == "partial_ft" else None
    history, learning_rates = _fit_formal_candidate(
        model,
        profile,
        candidate_plan.checkpoint_pattern,
    )
    checkpoint = select_best_checkpoint(history, candidate_plan.candidate_root)
    reloaded = reload_best_checkpoint(checkpoint)
    metrics = evaluate_validation_candidate(
        reloaded,
        profile.validation,
        profile.scalers.target_scaler,
    )
    state_audit = None
    if before is not None:
        _require(locked_source_provenance is not None, "PFT locked Source provenance missing")
        state_audit = audit_partial_ft_state(before, snapshot_weight_layers(model))
    trainable_params = (
        EXPECTED_TRAINABLE_PARAMS
        if candidate_plan.lifecycle == "partial_ft"
        else EXPECTED_TOTAL_PARAMS
    )
    score = build_validation_score(
        formal_plan=formal_plan,
        candidate_plan=candidate_plan,
        checkpoint=checkpoint,
        metrics=metrics,
        trainable_params=trainable_params,
    )
    _write_history(candidate_plan.candidate_root / "history.csv", history)
    _write_lr_history(candidate_plan.candidate_root / "learning_rate_history.csv", learning_rates)
    _write_json_exclusive(
        candidate_plan.candidate_root / "best_checkpoint.json",
        asdict(checkpoint),
    )
    _write_json_exclusive(
        candidate_plan.candidate_root / "validation_metrics_original_scale.json",
        asdict(metrics),
    )
    _write_json_exclusive(
        candidate_plan.candidate_root / "access_log.json",
        {
            "opened_data_files": [
                str(path) for path in dict.fromkeys(candidate_accessed_files)
            ],
            "forbidden_split_opened": False,
            "test_accessed": False,
        },
    )
    _write_json_exclusive(
        candidate_plan.candidate_root / "candidate_manifest.json",
        {
            "protocol_version": FORMAL_PROTOCOL_VERSION,
            "experiment_id": formal_plan.experiment_id,
            "run_id": formal_plan.run_id,
            "git_head": formal_plan.git.head,
            "lifecycle": candidate_plan.lifecycle,
            "candidate_id": candidate_plan.candidate_id,
            "initial_learning_rate": candidate_plan.learning_rate,
            "completed_epochs": len(history["val_loss"]),
            "best_epoch": checkpoint.best_epoch,
            "checkpoint_sha256": checkpoint.sha256,
            "validation_only": True,
            "test_accessed": False,
            "test_metrics_used": False,
            "locked_source_checkpoint_sha256": (
                locked_source_provenance.source_checkpoint_sha256
                if locked_source_provenance is not None
                else None
            ),
            "partial_ft_state_audit": asdict(state_audit) if state_audit else None,
        },
    )
    return CandidateRunResult(
        plan=candidate_plan,
        completed_epochs=len(history["val_loss"]),
        score=score,
        metrics=metrics,
        checkpoint=checkpoint,
        reloaded_model=reloaded,
        state_audit=state_audit,
    )


def _reset_seed() -> None:
    random.seed(LINEAR_SEED)
    np.random.seed(LINEAR_SEED)
    tf.random.set_seed(LINEAR_SEED)


def run_formal_train_validation(
    experiment_id: str,
    run_id: str,
    *,
    expected_git_head: str,
    execute_training: bool = False,
) -> FormalTrainingPlan | FormalTrainingResult:
    """Prepare a run, or explicitly execute six Training/Validation candidates."""

    plan = prepare_formal_training_run(
        experiment_id,
        run_id,
        expected_git_head=expected_git_head,
    )
    if not execute_training:
        return plan
    _require(os.environ.get("SOLAR_RUN_FORMAL") == "1", "Formal execution opt-in missing")
    set_reproducibility()
    accessed: list[Path] = []
    identity = protocol_git_identity(plan)
    manifest_accessed: list[Path] = []
    with record_data_access(manifest_accessed):
        base_manifest, _ = build_protocol_manifest(experiment_id, plan.paths, identity)
    accessed.extend(manifest_accessed)
    manifest = formal_manifest_for_training(plan, base_manifest)
    plan.paths.run_root.mkdir(parents=True, exist_ok=False)
    manifest_path = plan.paths.protocol_manifest
    _write_json_exclusive(manifest_path, manifest)
    source_access_start = len(accessed)
    source_profile = load_role_training_validation(plan.spec, "source", accessed)
    source_accessed = tuple(accessed[source_access_start:])

    source_results = []
    for candidate in plan.candidates["source"]:
        _reset_seed()
        model = build_candidate_model(candidate)
        source_results.append(
            _execute_candidate(
                formal_plan=plan,
                candidate_plan=candidate,
                profile=source_profile,
                model=model,
                candidate_accessed_files=source_accessed,
            )
        )
    source_scores = tuple(result.score for result in source_results)
    selected_source_score = select_validation_candidate("source", source_scores)
    locked_source_result = next(
        result
        for result in source_results
        if result.score.candidate_id == selected_source_score.candidate_id
    )
    source_provenance = build_source_checkpoint_provenance_from_selection(
        experiment_id=experiment_id,
        run_id=run_id,
        source_scores=source_scores,
        expected_git_head=plan.git.head,
        architecture_evidence=_source_architecture(
            locked_source_result.reloaded_model,
            plan.git.head,
        ),
    )
    _require(
        locked_source_result.score.candidate_id
        == source_provenance.source_candidate_id,
        "Selected Source provenance/result mismatch",
    )

    # Target Training/Validation bytes are not opened until Source selection
    # and provenance have both been locked.
    target_access_start = len(accessed)
    target_profile = load_role_training_validation(plan.spec, "target", accessed)
    target_accessed = tuple(accessed[target_access_start:])

    wotl_results = []
    for candidate in plan.candidates["wotl"]:
        _reset_seed()
        wotl_results.append(
            _execute_candidate(
                formal_plan=plan,
                candidate_plan=candidate,
                profile=target_profile,
                model=build_candidate_model(candidate),
                candidate_accessed_files=target_accessed,
            )
        )

    partial_results = []
    for candidate in plan.candidates["partial_ft"]:
        _reset_seed()
        partial_results.append(
            _execute_candidate(
                formal_plan=plan,
                candidate_plan=candidate,
                profile=target_profile,
                model=build_candidate_model(
                    candidate,
                    locked_source_model=locked_source_result.reloaded_model,
                ),
                candidate_accessed_files=target_accessed,
                locked_source_provenance=source_provenance,
            )
        )

    selection = build_selection_record(
        experiment_id=experiment_id,
        selection_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_scores=source_scores,
        source_architecture_evidence=_source_architecture(
            locked_source_result.reloaded_model, plan.git.head
        ),
        expected_git_head=plan.git.head,
        wotl_scores=tuple(result.score for result in wotl_results),
        partial_ft_scores=tuple(result.score for result in partial_results),
    )
    plan.paths.selection.mkdir(parents=False, exist_ok=False)
    selection_path = plan.paths.selection / "selection.json"
    write_and_reload_selection(selection, selection_path)
    locked_manifest = lock_manifest_after_selection(manifest, selection)
    _write_json_replace(manifest_path, locked_manifest)
    _require(
        FORBIDDEN_SPLIT_FILENAMES.isdisjoint(path.name for path in accessed),
        "Forbidden split appeared in Formal access audit",
    )
    total_training_epochs = sum(
        result.completed_epochs
        for result in (*source_results, *wotl_results, *partial_results)
    )
    _require(total_training_epochs > 0, "Formal execution reported zero training epochs")
    return FormalTrainingResult(
        plan=plan,
        source_results=tuple(source_results),
        wotl_results=tuple(wotl_results),
        partial_ft_results=tuple(partial_results),
        source_provenance=source_provenance,
        selection=selection,
        selection_path=selection_path,
        manifest_path=manifest_path,
        accessed_files=tuple(dict.fromkeys(accessed)),
        total_training_epochs_executed=total_training_epochs,
    )
