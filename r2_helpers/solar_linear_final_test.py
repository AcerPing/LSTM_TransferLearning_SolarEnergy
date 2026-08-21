"""Fail-closed Final Target Test gate for locked Solar Linear Experiment B.

Importing this module and preparing an authorization are read-only operations.
The corrected Target Test is reachable only through ``OneTimeTestAccess`` after
both the explicit function argument and process environment opt-in are present.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from r2_config.solar_linear import (
    LINEAR_BATCH_SIZE,
    LINEAR_EXPERIMENTS,
    REPOSITORY_ROOT,
)
from r2_helpers.solar_linear_formal import (
    FormalProtocolError,
    ProtocolStage,
    validate_selection_record,
)


LOCKED_EXPERIMENT_ID = "B"
LOCKED_RUN_ID = "20260821T041718Z_seed1234"
SELECTION_GIT_HEAD = "ec6e881cf95f03bedd81adff10e55151a8b26ff5"

LOCKED_SOURCE_CANDIDATE_ID = "SRC_lr1e-4"
LOCKED_SOURCE_EPOCH = 500
LOCKED_SOURCE_SHA256 = (
    "5cd3db4501d6c6720fbdd8ee3cce692b01cecb82619574f8c6944a8795182d91"
)
LOCKED_WOTL_CANDIDATE_ID = "WOTL_lr1e-4"
LOCKED_WOTL_EPOCH = 499
LOCKED_WOTL_SHA256 = (
    "b8162994b7138cc619a79811b470c05087e329e5087813f4229cf0819a3a56b6"
)
LOCKED_PARTIAL_FT_CANDIDATE_ID = "PFT_lr3e-5"
LOCKED_PARTIAL_FT_EPOCH = 4
LOCKED_PARTIAL_FT_SHA256 = (
    "1d14bdb481a362e8e974d42f44bc83dffa7034b4ea3e19ac30196c316463dc37"
)

LOCKED_RUN_ROOT = (
    LINEAR_EXPERIMENTS[LOCKED_EXPERIMENT_ID].output_root / LOCKED_RUN_ID
)
LOCKED_FINAL_TEST_ROOT = LOCKED_RUN_ROOT / "final_test"
EXPECTED_TEST_ROWS = 2612
EXPECTED_TEST_SEQUENCES = 2607
EXPECTED_X_TEST_SHAPE = (EXPECTED_TEST_SEQUENCES, 5, 5)
EXPECTED_Y_TEST_SHAPE = (EXPECTED_TEST_SEQUENCES,)


class FinalTestError(AssertionError):
    """Raised before any prohibited or inconsistent Final Test action."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalTestError(message)


class FinalTestState(str, Enum):
    TEST_AUTHORIZED = "TEST_AUTHORIZED"
    TEST_ACCESS_STARTED = "TEST_ACCESS_STARTED"
    TEST_ACCESSED_INCOMPLETE = "TEST_ACCESSED_INCOMPLETE"
    TEST_COMPLETED = "TEST_COMPLETED"


class TransferClassification(str, Enum):
    POSITIVE_TRANSFER = "Positive Transfer"
    PARTIAL_POSITIVE_TRANSFER = "Partial Positive Transfer"
    MIXED_RESULT = "Mixed Result"
    NEGATIVE_TRANSFER = "Negative Transfer"


@dataclass(frozen=True)
class GitState:
    head: str
    tracked_dirty: bool


_AUTHORIZATION_BINDINGS: dict[object, tuple[Path, Path]] = {}


@dataclass(frozen=True)
class FinalTestAuthorization:
    experiment_id: str
    run_id: str
    selection_git_head: str
    execution_git_head: str
    source_candidate_id: str
    source_checkpoint_sha256: str
    wotl_candidate_id: str
    wotl_checkpoint_path: Path
    wotl_checkpoint_sha256: str
    partial_ft_candidate_id: str
    partial_ft_checkpoint_path: Path
    partial_ft_checkpoint_sha256: str
    comparison_pair_locked: bool
    selection_locked: bool
    test_authorized: bool
    authorization_timestamp: str
    test_access_count: int
    post_test_tuning_allowed: bool
    test_accessed: bool = False
    test_completed: bool = False
    state: FinalTestState = FinalTestState.TEST_AUTHORIZED
    _proof: object = None


@dataclass(frozen=True)
class TargetTestData:
    X_test: np.ndarray
    y_test: np.ndarray
    target_timestamp: np.ndarray
    target_scaler: Any
    accessed_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class OriginalScaleMetrics:
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
class PredictionEvaluation:
    prediction_normalized: np.ndarray
    prediction_original: np.ndarray
    y_true_original: np.ndarray
    metrics: OriginalScaleMetrics
    normalized_mse: float
    normalized_mae: float
    normalized_rmse: float


@dataclass(frozen=True)
class FinalTestExecutionResult:
    authorization: FinalTestAuthorization
    wotl: PredictionEvaluation
    partial_ft: PredictionEvaluation
    classification: TransferClassification
    output_root: Path


class OneTimeTestAccess:
    """Allow exactly one successful invocation of the controlled Test loader."""

    def __init__(self, loader: Callable[[], TargetTestData]) -> None:
        self._loader = loader
        self._access_count = 0

    @property
    def access_count(self) -> int:
        return self._access_count

    def load(self) -> TargetTestData:
        _require(self._access_count == 0, "Target Test may be loaded only once")
        data = self._loader()
        validate_target_test_data(data)
        self._access_count = 1
        return data


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"Required checkpoint is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"Required JSON is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalTestError(f"Cannot read JSON {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _git_bytes(*arguments: str) -> bytes:
    safe = f"safe.directory={REPOSITORY_ROOT.as_posix()}"
    completed = subprocess.run(
        ["git", "-c", safe, *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    _require(completed.returncode == 0, f"Git command failed: {' '.join(arguments)}")
    return completed.stdout


def collect_git_state() -> GitState:
    return GitState(
        head=_git_bytes("rev-parse", "HEAD").strip().decode("ascii"),
        tracked_dirty=bool(
            _git_bytes("status", "--porcelain=v1", "--untracked-files=no")
        ),
    )


def _checkpoint_path(
    run_root: Path,
    lifecycle_directory: str,
    candidate_id: str,
    epoch: int,
) -> Path:
    return (
        run_root
        / lifecycle_directory
        / candidate_id
        / f"checkpoint_epoch_{epoch:04d}.hdf5"
    )


def _selected_comparison(
    selection: Mapping[str, Any], field: str, candidate_id: str
) -> Mapping[str, Any]:
    comparison = selection.get(field)
    _require(isinstance(comparison, Sequence), f"{field} must be a sequence")
    matches = [
        item
        for item in comparison
        if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
    ]
    _require(len(matches) == 1, f"Locked candidate missing/duplicated in {field}")
    return matches[0]


def _validate_locked_candidate(
    evidence: Mapping[str, Any],
    *,
    candidate_id: str,
    epoch: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> None:
    _require(evidence.get("candidate_id") == candidate_id, "Candidate identity mismatch")
    _require(evidence.get("best_epoch") == epoch, f"{candidate_id} epoch mismatch")
    _require(
        Path(str(evidence.get("checkpoint_path", ""))).resolve()
        == checkpoint_path.resolve(),
        f"{candidate_id} checkpoint path mismatch",
    )
    _require(
        evidence.get("checkpoint_sha256") == checkpoint_sha256,
        f"{candidate_id} checkpoint SHA evidence mismatch",
    )
    _require(
        evidence.get("checkpoint_sha256_verified") is True,
        f"{candidate_id} checkpoint SHA is not verified",
    )


def _validate_protocol_manifest(
    manifest: Mapping[str, Any], selection: Mapping[str, Any]
) -> None:
    _require(manifest.get("experiment_id") == LOCKED_EXPERIMENT_ID, "Manifest experiment mismatch")
    _require(manifest.get("run_id") == LOCKED_RUN_ID, "Manifest run ID mismatch")
    git = manifest.get("git")
    _require(isinstance(git, Mapping), "Manifest Git identity missing")
    _require(git.get("head") == SELECTION_GIT_HEAD, "Manifest Git HEAD mismatch")
    _require(manifest.get("formal_eligible") is True, "Run is not formal eligible")
    _require(manifest.get("dry_run") is False, "Dry-run cannot authorize Final Test")
    _require(manifest.get("config_locked") is True, "Manifest config is not locked")
    _require(manifest.get("selection_locked") is True, "Manifest selection is not locked")
    _require(manifest.get("checkpoint_locked") is True, "Manifest checkpoint is not locked")
    _require(manifest.get("test_accessed") is False, "Manifest reports prior Test access")
    _require(manifest.get("test_authorized") is False, "Manifest Test is already authorized")
    _require(
        manifest.get("test_metrics_used_for_selection") is False,
        "Manifest selection used Test metrics",
    )
    _require(
        manifest.get("post_test_tuning_allowed") is False,
        "Manifest permits post-Test tuning",
    )
    _require(
        manifest.get("protocol_stage")
        == ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        "Manifest protocol stage cannot authorize Test",
    )
    _require(
        manifest.get("target_protocol_fingerprint")
        == selection.get("target_protocol_fingerprint"),
        "Manifest/selection Target protocol mismatch",
    )


def _validate_not_completed(run_root: Path, manifest: Mapping[str, Any]) -> None:
    stage = manifest.get("protocol_stage")
    _require(
        stage not in (ProtocolStage.TEST_COMPLETED, ProtocolStage.TEST_COMPLETED.value, FinalTestState.TEST_COMPLETED.value),
        "Formal run is already TEST_COMPLETED",
    )
    final_root = run_root / "final_test"
    _validate_output_root_available(final_root)


def _validate_output_root_available(final_root: Path) -> None:
    """Fail closed for every pre-existing Final Test output state."""

    if not final_root.exists():
        return
    _require(final_root.is_dir(), "Inconsistent Final Test state: output root is not a directory")
    if (final_root / "comparison.json").exists():
        raise FinalTestError("Final Test comparison already exists")
    state_path = final_root / "final_state.json"
    _require(
        state_path.is_file(),
        "Inconsistent Final Test state: output root exists without final_state.json",
    )
    state = _load_json(state_path).get("state")
    if state == FinalTestState.TEST_ACCESS_STARTED.value:
        raise FinalTestError(
            "Final Test access attempt already started; pristine Test status cannot be "
            "re-established; re-run prohibited"
        )
    if state == FinalTestState.TEST_ACCESSED_INCOMPLETE.value:
        raise FinalTestError("Target Test was accessed but Final Test is incomplete; re-run prohibited")
    if state == FinalTestState.TEST_COMPLETED.value:
        raise FinalTestError("Formal run is already TEST_COMPLETED")
    raise FinalTestError(f"Inconsistent Final Test state: {state!r}")


def prepare_final_test_authorization(
    experiment_id: str = LOCKED_EXPERIMENT_ID,
    run_id: str = LOCKED_RUN_ID,
    *,
    human_authorized: bool,
    expected_execution_git_head: str,
    run_root: Path | str | None = None,
    git_state: GitState | None = None,
    authorization_timestamp: str | None = None,
    sha256_func: Callable[[Path], str] = _sha256,
    selection_validator: Callable[[Mapping[str, Any]], None] = validate_selection_record,
    _allow_test_root: bool = False,
) -> FinalTestAuthorization:
    """Validate the sealed selection and issue an in-memory authorization."""

    _require(human_authorized is True, "Explicit human Final Test authorization required")
    _require(experiment_id == LOCKED_EXPERIMENT_ID, "Only Experiment B may be authorized")
    _require(run_id == LOCKED_RUN_ID, "Final Test run ID mismatch")
    root = Path(run_root) if run_root is not None else LOCKED_RUN_ROOT
    if not _allow_test_root:
        _require(root.resolve() == LOCKED_RUN_ROOT.resolve(), "Unexpected Formal run root")
        _require(git_state is None, "Production authorization requires runtime Git collection")
        _require(sha256_func is _sha256, "Production authorization requires real SHA-256")
        _require(
            selection_validator is validate_selection_record,
            "Production authorization requires the locked selection validator",
        )

    _require(
        isinstance(expected_execution_git_head, str)
        and len(expected_execution_git_head) == 40
        and all(character in "0123456789abcdef" for character in expected_execution_git_head),
        "Expected execution Git HEAD must be an explicit lowercase 40-character SHA",
    )
    state = git_state or collect_git_state()
    _require(state.head == expected_execution_git_head, "Final Test execution Git HEAD mismatch")
    _require(not state.tracked_dirty, "Tracked working tree is dirty")

    selection = _load_json(root / "selection" / "selection.json")
    manifest = _load_json(root / "protocol_manifest.json")
    try:
        selection_validator(selection)
    except FormalProtocolError as exc:
        raise FinalTestError(f"Selection validation failed: {exc}") from exc
    _validate_protocol_manifest(manifest, selection)
    _validate_not_completed(root, manifest)

    _require(selection.get("experiment_id") == LOCKED_EXPERIMENT_ID, "Selection experiment mismatch")
    _require(selection.get("run_id") == LOCKED_RUN_ID, "Selection run ID mismatch")
    _require(selection.get("git_head") == SELECTION_GIT_HEAD, "Selection Git HEAD mismatch")
    _require(selection.get("selection_locked") is True, "Selection is not locked")
    _require(selection.get("comparison_pair_locked") is True, "Comparison pair is not locked")
    _require(selection.get("test_accessed") is False, "Selection reports prior Test access")
    _require(selection.get("test_authorized") is False, "Selection Test is already authorized")
    _require(
        selection.get("test_metrics_used_for_selection") is False,
        "Selection used Test metrics",
    )
    _require(
        selection.get("post_test_tuning_allowed") is False,
        "Selection permits post-Test tuning",
    )

    source_path = _checkpoint_path(
        root, "source_candidates", LOCKED_SOURCE_CANDIDATE_ID, LOCKED_SOURCE_EPOCH
    )
    wotl_path = _checkpoint_path(
        root, "target_wotl_candidates", LOCKED_WOTL_CANDIDATE_ID, LOCKED_WOTL_EPOCH
    )
    partial_path = _checkpoint_path(
        root,
        "target_partial_ft_candidates",
        LOCKED_PARTIAL_FT_CANDIDATE_ID,
        LOCKED_PARTIAL_FT_EPOCH,
    )

    _require(selection.get("source_selected_candidate") == LOCKED_SOURCE_CANDIDATE_ID, "Source candidate mismatch")
    _require(selection.get("source_checkpoint_sha256") == LOCKED_SOURCE_SHA256, "Source SHA mismatch")
    _require(selection.get("wotl_selected_candidate") == LOCKED_WOTL_CANDIDATE_ID, "WOTL candidate mismatch")
    _require(selection.get("wotl_selected_checkpoint_sha256") == LOCKED_WOTL_SHA256, "WOTL SHA mismatch")
    _require(selection.get("partial_ft_selected_candidate") == LOCKED_PARTIAL_FT_CANDIDATE_ID, "Partial FT candidate mismatch")
    _require(selection.get("partial_ft_selected_checkpoint_sha256") == LOCKED_PARTIAL_FT_SHA256, "Partial FT SHA mismatch")

    _validate_locked_candidate(
        _selected_comparison(selection, "source_validation_comparison", LOCKED_SOURCE_CANDIDATE_ID),
        candidate_id=LOCKED_SOURCE_CANDIDATE_ID,
        epoch=LOCKED_SOURCE_EPOCH,
        checkpoint_path=source_path,
        checkpoint_sha256=LOCKED_SOURCE_SHA256,
    )
    _validate_locked_candidate(
        _selected_comparison(selection, "wotl_validation_comparison", LOCKED_WOTL_CANDIDATE_ID),
        candidate_id=LOCKED_WOTL_CANDIDATE_ID,
        epoch=LOCKED_WOTL_EPOCH,
        checkpoint_path=wotl_path,
        checkpoint_sha256=LOCKED_WOTL_SHA256,
    )
    _validate_locked_candidate(
        _selected_comparison(selection, "partial_ft_validation_comparison", LOCKED_PARTIAL_FT_CANDIDATE_ID),
        candidate_id=LOCKED_PARTIAL_FT_CANDIDATE_ID,
        epoch=LOCKED_PARTIAL_FT_EPOCH,
        checkpoint_path=partial_path,
        checkpoint_sha256=LOCKED_PARTIAL_FT_SHA256,
    )

    for path, expected in (
        (source_path, LOCKED_SOURCE_SHA256),
        (wotl_path, LOCKED_WOTL_SHA256),
        (partial_path, LOCKED_PARTIAL_FT_SHA256),
    ):
        _require(path.is_file() and path.stat().st_size > 0, f"Checkpoint missing/empty: {path}")
        _require(sha256_func(path) == expected, f"Actual checkpoint SHA mismatch: {path.name}")

    timestamp = authorization_timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    _require(isinstance(timestamp, str) and bool(timestamp), "Authorization timestamp missing")
    proof = object()
    _AUTHORIZATION_BINDINGS[proof] = (wotl_path.resolve(), partial_path.resolve())
    return FinalTestAuthorization(
        experiment_id=LOCKED_EXPERIMENT_ID,
        run_id=LOCKED_RUN_ID,
        selection_git_head=SELECTION_GIT_HEAD,
        execution_git_head=state.head,
        source_candidate_id=LOCKED_SOURCE_CANDIDATE_ID,
        source_checkpoint_sha256=LOCKED_SOURCE_SHA256,
        wotl_candidate_id=LOCKED_WOTL_CANDIDATE_ID,
        wotl_checkpoint_path=wotl_path,
        wotl_checkpoint_sha256=LOCKED_WOTL_SHA256,
        partial_ft_candidate_id=LOCKED_PARTIAL_FT_CANDIDATE_ID,
        partial_ft_checkpoint_path=partial_path,
        partial_ft_checkpoint_sha256=LOCKED_PARTIAL_FT_SHA256,
        comparison_pair_locked=True,
        selection_locked=True,
        test_authorized=True,
        authorization_timestamp=timestamp,
        test_access_count=0,
        post_test_tuning_allowed=False,
        _proof=proof,
    )


def validate_authorization(authorization: FinalTestAuthorization) -> None:
    _require(isinstance(authorization, FinalTestAuthorization), "Invalid Final Test authorization")
    try:
        binding = _AUTHORIZATION_BINDINGS.get(authorization._proof)
    except TypeError:
        binding = None
    _require(binding is not None, "Invalid Final Test authorization proof")
    locked_wotl_path, locked_partial_path = binding
    _require(authorization.experiment_id == LOCKED_EXPERIMENT_ID, "Authorization experiment mismatch")
    _require(authorization.run_id == LOCKED_RUN_ID, "Authorization run mismatch")
    _require(
        authorization.selection_git_head == SELECTION_GIT_HEAD,
        "Authorization selection Git mismatch",
    )
    _require(
        isinstance(authorization.execution_git_head, str)
        and len(authorization.execution_git_head) == 40
        and all(character in "0123456789abcdef" for character in authorization.execution_git_head),
        "Authorization execution Git mismatch",
    )
    _require(
        authorization.source_candidate_id == LOCKED_SOURCE_CANDIDATE_ID,
        "Authorization Source candidate mismatch",
    )
    _require(
        authorization.source_checkpoint_sha256 == LOCKED_SOURCE_SHA256,
        "Authorization Source checkpoint SHA mismatch",
    )
    _require(
        authorization.wotl_candidate_id == LOCKED_WOTL_CANDIDATE_ID,
        "Authorization WOTL candidate mismatch",
    )
    _require(
        authorization.wotl_checkpoint_path.resolve() == locked_wotl_path,
        "Authorization WOTL checkpoint path mismatch",
    )
    _require(
        authorization.wotl_checkpoint_sha256 == LOCKED_WOTL_SHA256,
        "Authorization WOTL checkpoint SHA mismatch",
    )
    _require(
        authorization.partial_ft_candidate_id == LOCKED_PARTIAL_FT_CANDIDATE_ID,
        "Authorization Partial FT candidate mismatch",
    )
    _require(
        authorization.partial_ft_checkpoint_path.resolve() == locked_partial_path,
        "Authorization Partial FT checkpoint path mismatch",
    )
    _require(
        authorization.partial_ft_checkpoint_sha256 == LOCKED_PARTIAL_FT_SHA256,
        "Authorization Partial FT checkpoint SHA mismatch",
    )
    _require(authorization.comparison_pair_locked, "Authorization pair is not locked")
    _require(authorization.selection_locked, "Authorization selection is not locked")
    _require(authorization.test_authorized, "Final Test is not authorized")
    _require(authorization.test_access_count == 0, "Authorization already consumed Test")
    _require(not authorization.test_accessed, "Authorization reports prior Test access")
    _require(not authorization.test_completed, "Authorization already completed")
    _require(not authorization.post_test_tuning_allowed, "Authorization permits post-Test tuning")
    _require(authorization.state is FinalTestState.TEST_AUTHORIZED, "Authorization state mismatch")


def validate_target_test_data(data: TargetTestData) -> None:
    X = np.asarray(data.X_test)
    y = np.asarray(data.y_test)
    timestamps = np.asarray(data.target_timestamp)
    _require(X.shape == EXPECTED_X_TEST_SHAPE, f"Target Test X shape mismatch: {X.shape}")
    _require(y.shape == EXPECTED_Y_TEST_SHAPE, f"Target Test y shape mismatch: {y.shape}")
    _require(timestamps.shape == EXPECTED_Y_TEST_SHAPE, "Target Test timestamp shape mismatch")
    _require(np.isfinite(X).all(), "Target Test X contains NaN/Inf")
    _require(np.isfinite(y).all(), "Target Test y contains NaN/Inf")
    _require(hasattr(data.target_scaler, "inverse_transform"), "Target scaler is invalid")


def load_locked_target_test() -> TargetTestData:
    """The sole production loader for corrected Plant1 Target Test."""

    from r2_helpers import solar_data

    spec = LINEAR_EXPERIMENTS[LOCKED_EXPERIMENT_ID]
    profile_dir = spec.target_profile_path
    checksums = solar_data.validate_checksums_for_splits(profile_dir, splits=("test",))
    metadata = solar_data.load_profile_metadata(profile_dir)
    solar_data.validate_manifest(
        metadata,
        expected_plant="Plant1",
        expected_profile="target",
        expected_profile_dir_name="target_profile",
    )
    scalers = solar_data.validate_scalers(metadata)
    split = solar_data.load_split(metadata, "test")
    solar_data.validate_feature_transform_consistency(split, scalers.feature_scaler)
    solar_data.validate_target_round_trip(split, scalers.target_scaler)
    sequences = solar_data.build_sequences(
        split.X_normalized, split.y_normalized, split.timestamps
    )
    _require(split.row_count == EXPECTED_TEST_ROWS, "Target Test row count mismatch")
    data = TargetTestData(
        X_test=sequences.X_seq,
        y_test=sequences.y_seq,
        target_timestamp=sequences.target_timestamp,
        target_scaler=scalers.target_scaler,
        accessed_files=(
            profile_dir / "file_checksums.csv",
            *(profile_dir / name for name in sorted(checksums)),
        ),
    )
    validate_target_test_data(data)
    return data


def _default_model_loader(path: Path) -> Any:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from keras.models import load_model
    from r2_helpers.solar_linear_runtime import rmse

    return load_model(str(path), custom_objects={"rmse": rmse})


def _default_model_validator(model: Any) -> None:
    from r2_helpers.solar_linear_runtime import validate_linear_model

    validate_linear_model(model)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum(np.square(y_true - y_pred)))
    total = float(np.sum(np.square(y_true - np.mean(y_true))))
    _require(total > 0, "Target Test target variance is zero")
    return 1.0 - residual / total


def compute_original_scale_metrics(
    y_true_original: np.ndarray,
    prediction_original: np.ndarray,
) -> OriginalScaleMetrics:
    truth = np.asarray(y_true_original, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_original, dtype=np.float64).reshape(-1)
    _require(truth.shape == prediction.shape and len(truth) > 0, "Prediction alignment mismatch")
    _require(np.isfinite(truth).all() and np.isfinite(prediction).all(), "Non-finite prediction metric input")
    error = truth - prediction
    mse = float(np.mean(np.square(error)))
    return OriginalScaleMetrics(
        mae=float(np.mean(np.abs(error))),
        mse=mse,
        rmse=math.sqrt(mse),
        r2=_r2(truth, prediction),
        prediction_count=len(prediction),
        negative_prediction_count=int(np.count_nonzero(prediction < 0)),
    )


def evaluate_raw_prediction(
    y_true_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
    target_scaler: Any,
) -> PredictionEvaluation:
    truth = np.asarray(y_true_normalized, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_normalized, dtype=np.float64).reshape(-1)
    _require(truth.shape == prediction.shape and len(truth) > 0, "Normalized prediction alignment mismatch")
    error = truth - prediction
    normalized_mse = float(np.mean(np.square(error)))
    truth_original = np.asarray(
        target_scaler.inverse_transform(truth.reshape(-1, 1)), dtype=np.float64
    ).reshape(-1)
    prediction_original = np.asarray(
        target_scaler.inverse_transform(prediction.reshape(-1, 1)), dtype=np.float64
    ).reshape(-1)
    return PredictionEvaluation(
        prediction_normalized=prediction,
        prediction_original=prediction_original,
        y_true_original=truth_original,
        metrics=compute_original_scale_metrics(truth_original, prediction_original),
        normalized_mse=normalized_mse,
        normalized_mae=float(np.mean(np.abs(error))),
        normalized_rmse=math.sqrt(normalized_mse),
    )


def classify_transfer(
    wotl: OriginalScaleMetrics,
    partial_ft: OriginalScaleMetrics,
) -> TransferClassification:
    errors_improve = (
        partial_ft.mae < wotl.mae
        and partial_ft.mse < wotl.mse
        and partial_ft.rmse < wotl.rmse
    )
    r2_improves = partial_ft.r2 > wotl.r2
    if errors_improve and r2_improves:
        return TransferClassification.POSITIVE_TRANSFER
    if errors_improve:
        return TransferClassification.PARTIAL_POSITIVE_TRANSFER
    all_unfavorable = (
        partial_ft.mae >= wotl.mae
        and partial_ft.mse >= wotl.mse
        and partial_ft.rmse >= wotl.rmse
        and partial_ft.r2 <= wotl.r2
    )
    if all_unfavorable:
        return TransferClassification.NEGATIVE_TRANSFER
    return TransferClassification.MIXED_RESULT


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


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_json_atomic_replace(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not temporary.exists(), f"Stale atomic-write file exists: {temporary.name}")
    try:
        _write_json_exclusive(temporary, payload)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _authorization_payload(authorization: FinalTestAuthorization) -> dict[str, Any]:
    payload = asdict(authorization)
    payload.pop("_proof")
    return payload


def _final_state_payload(
    authorization: FinalTestAuthorization,
    *,
    state: FinalTestState,
    test_accessed: bool,
    test_access_count: int,
    test_completed: bool,
    test_access_attempt_count: int,
    test_access_start_timestamp: str,
    test_access_timestamp: str | None = None,
) -> dict[str, Any]:
    payload = _authorization_payload(
        replace(
            authorization,
            state=state,
            test_accessed=test_accessed,
            test_access_count=test_access_count,
            test_completed=test_completed,
            post_test_tuning_allowed=False,
        )
    )
    payload["test_access_attempt_count"] = test_access_attempt_count
    payload["test_access_start_timestamp"] = test_access_start_timestamp
    if test_access_timestamp is not None:
        payload["test_access_timestamp"] = test_access_timestamp
    return payload


def _write_predictions(
    path: Path,
    data: TargetTestData,
    wotl: PredictionEvaluation,
    partial_ft: PredictionEvaluation,
) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "timestamp",
                "y_true_original",
                "wotl_pred_original",
                "partial_ft_pred_original",
                "wotl_residual",
                "partial_ft_residual",
                "y_true_normalized",
                "wotl_pred_normalized",
                "partial_ft_pred_normalized",
            )
        )
        for index, timestamp in enumerate(data.target_timestamp):
            truth = wotl.y_true_original[index]
            writer.writerow(
                (
                    str(timestamp),
                    truth,
                    wotl.prediction_original[index],
                    partial_ft.prediction_original[index],
                    truth - wotl.prediction_original[index],
                    truth - partial_ft.prediction_original[index],
                    data.y_test[index],
                    wotl.prediction_normalized[index],
                    partial_ft.prediction_normalized[index],
                )
            )


def execute_final_test(
    authorization: FinalTestAuthorization,
    *,
    execute_test: bool,
    output_root: Path | str | None = None,
    test_loader: Callable[[], TargetTestData] = load_locked_target_test,
    model_loader: Callable[[Path], Any] = _default_model_loader,
    model_validator: Callable[[Any], None] = _default_model_validator,
    sha256_func: Callable[[Path], str] = _sha256,
    git_state_provider: Callable[[], GitState] = collect_git_state,
    _allow_test_dependencies: bool = False,
) -> FinalTestExecutionResult | None:
    """Execute exactly one locked WOTL/PFT Test comparison after dual opt-in."""

    validate_authorization(authorization)
    if not execute_test:
        return None
    _require(
        os.environ.get("SOLAR_RUN_FINAL_TEST") == "1",
        "SOLAR_RUN_FINAL_TEST=1 opt-in missing",
    )
    root = Path(output_root) if output_root is not None else LOCKED_FINAL_TEST_ROOT
    if not _allow_test_dependencies:
        _require(root.resolve() == LOCKED_FINAL_TEST_ROOT.resolve(), "Unexpected Final Test output root")
        _require(test_loader is load_locked_target_test, "Production requires locked Target Test loader")
        _require(model_loader is _default_model_loader, "Production requires locked model loader")
        _require(model_validator is _default_model_validator, "Production requires linear model validator")
        _require(sha256_func is _sha256, "Production requires real checkpoint SHA-256")
        _require(
            git_state_provider is collect_git_state,
            "Production execution requires runtime Git collection",
        )
    current_git = git_state_provider()
    _require(
        current_git.head == authorization.execution_git_head,
        "Final Test execute-time Git HEAD mismatch",
    )
    _require(not current_git.tracked_dirty, "Tracked working tree became dirty before Final Test")
    _validate_output_root_available(root)

    for path, authorization_sha, locked_sha in (
        (
            authorization.wotl_checkpoint_path,
            authorization.wotl_checkpoint_sha256,
            LOCKED_WOTL_SHA256,
        ),
        (
            authorization.partial_ft_checkpoint_path,
            authorization.partial_ft_checkpoint_sha256,
            LOCKED_PARTIAL_FT_SHA256,
        ),
    ):
        actual_sha = sha256_func(path)
        _require(actual_sha == authorization_sha, "Checkpoint changed after authorization")
        _require(actual_sha == locked_sha, "Checkpoint no longer matches locked selection")

    wotl_model = model_loader(authorization.wotl_checkpoint_path)
    partial_model = model_loader(authorization.partial_ft_checkpoint_path)
    model_validator(wotl_model)
    model_validator(partial_model)
    _require(callable(getattr(wotl_model, "predict", None)), "WOTL model is not usable")
    _require(callable(getattr(partial_model, "predict", None)), "Partial FT model is not usable")

    root.mkdir(parents=False, exist_ok=False)
    _write_json_exclusive(root / "authorization.json", _authorization_payload(authorization))
    test_access_start_timestamp = _utc_timestamp()
    _write_json_exclusive(
        root / "final_state.json",
        _final_state_payload(
            authorization,
            state=FinalTestState.TEST_ACCESS_STARTED,
            test_accessed=False,
            test_access_count=0,
            test_completed=False,
            test_access_attempt_count=1,
            test_access_start_timestamp=test_access_start_timestamp,
        ),
    )
    access = OneTimeTestAccess(test_loader)
    try:
        data = access.load()
    except Exception as exc:
        raise FinalTestError(
            "Final Test loader failed after TEST_ACCESS_STARTED; re-run prohibited"
        ) from exc
    _require(access.access_count == 1, "Target Test access count must equal one")
    test_access_timestamp = _utc_timestamp()
    _write_json_atomic_replace(
        root / "final_state.json",
        _final_state_payload(
            authorization,
            state=FinalTestState.TEST_ACCESSED_INCOMPLETE,
            test_accessed=True,
            test_access_count=1,
            test_completed=False,
            test_access_attempt_count=1,
            test_access_start_timestamp=test_access_start_timestamp,
            test_access_timestamp=test_access_timestamp,
        ),
    )

    try:
        wotl_prediction = np.asarray(
            wotl_model.predict(data.X_test, batch_size=LINEAR_BATCH_SIZE, verbose=0),
            dtype=np.float64,
        ).reshape(-1)
        partial_prediction = np.asarray(
            partial_model.predict(data.X_test, batch_size=LINEAR_BATCH_SIZE, verbose=0),
            dtype=np.float64,
        ).reshape(-1)
        _require(len(wotl_prediction) == EXPECTED_TEST_SEQUENCES, "WOTL prediction count mismatch")
        _require(len(partial_prediction) == EXPECTED_TEST_SEQUENCES, "Partial FT prediction count mismatch")

        wotl = evaluate_raw_prediction(data.y_test, wotl_prediction, data.target_scaler)
        partial = evaluate_raw_prediction(data.y_test, partial_prediction, data.target_scaler)
        classification = classify_transfer(wotl.metrics, partial.metrics)
        sealed = replace(
            authorization,
            test_access_count=1,
            test_accessed=True,
            test_completed=True,
            state=FinalTestState.TEST_COMPLETED,
            post_test_tuning_allowed=False,
        )

        _write_json_exclusive(root / "wotl_metrics_original_scale.json", asdict(wotl.metrics))
        _write_json_exclusive(root / "partial_ft_metrics_original_scale.json", asdict(partial.metrics))
        _write_json_exclusive(
            root / "access_log.json",
            {
                "test_access_count": access.access_count,
                "test_accessed": True,
                "test_completed": True,
                "opened_data_files": data.accessed_files,
                "post_test_tuning_allowed": False,
                "test_access_timestamp": test_access_timestamp,
            },
        )
        _write_predictions(root / "predictions.csv", data, wotl, partial)
        _write_json_exclusive(
            root / "comparison.json",
            {
                "experiment_id": sealed.experiment_id,
                "run_id": sealed.run_id,
                "selection_git_head": sealed.selection_git_head,
                "execution_git_head": sealed.execution_git_head,
                "classification": classification,
                "metrics_scale": "original",
                "target_name": "DC_POWER",
                "unit": "DC_POWER (raw dataset scale; documented unit: kW)",
                "wotl_candidate_id": sealed.wotl_candidate_id,
                "partial_ft_candidate_id": sealed.partial_ft_candidate_id,
                "wotl_metrics_original_scale": asdict(wotl.metrics),
                "partial_ft_metrics_original_scale": asdict(partial.metrics),
                "post_test_tuning_allowed": False,
                "state": FinalTestState.TEST_COMPLETED,
            },
        )
        _write_json_atomic_replace(
            root / "final_state.json",
            _final_state_payload(
                sealed,
                state=FinalTestState.TEST_COMPLETED,
                test_accessed=True,
                test_access_count=1,
                test_completed=True,
                test_access_attempt_count=1,
                test_access_start_timestamp=test_access_start_timestamp,
                test_access_timestamp=test_access_timestamp,
            ),
        )
    except Exception as exc:
        raise FinalTestError(
            "Final Test failed after durable Test access; state remains TEST_ACCESSED_INCOMPLETE"
        ) from exc
    return FinalTestExecutionResult(
        authorization=sealed,
        wotl=wotl,
        partial_ft=partial,
        classification=classification,
        output_root=root,
    )
