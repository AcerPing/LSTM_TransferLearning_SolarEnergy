"""Fail-closed one-time Final Target Test helper for Solar Linear Experiment A2.

The public preflight is read-only.  Authorization is in-memory only.  Real
Target Test access is possible only from ``execute_a2_final_test`` after an
explicit authorization, ``execute_test=True``, and ``SOLAR_RUN_FINAL_TEST=1``.
Importing this module never opens data or creates an output directory.
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

from r2_config.solar_linear import LINEAR_BATCH_SIZE, LINEAR_EXPERIMENTS, REPOSITORY_ROOT
from r2_config.solar_linear_formal import formal_path_contract
from r2_helpers.solar_linear_formal import (
    FormalProtocolError,
    ProtocolStage,
    target_protocol_fingerprint,
    validate_selection_record,
)


LOCKED_EXPERIMENT_ID = "A2"
LOCKED_RUN_ID = "20260918T051937Z_seed1234"
LOCKED_DIRECTION = "Plant1_to_Plant2"
LOCKED_SOURCE_PLANT = "Plant1"
LOCKED_TARGET_PLANT = "Plant2"
LOCKED_PROTOCOL_VERSION = "solar-linear-v1.0"
LOCKED_ACTIVATION = "linear"
SELECTION_GIT_HEAD = "51c496e40750591fefc007d972fcf5c34dbc30bc"

LOCKED_SOURCE_CANDIDATE_ID = "SRC_lr1e-4"
LOCKED_SOURCE_EPOCH = 500
LOCKED_SOURCE_SHA256 = "6bbd2adebdc1652cf47665a97f7339466b13894df71270e93c397dc8ab1538c9"
LOCKED_WOTL_CANDIDATE_ID = "WOTL_lr1e-4"
LOCKED_WOTL_EPOCH = 500
LOCKED_WOTL_SHA256 = "230ccb26aebb088ccb855ccdeaddc4f88f5a58f406a91632ce07b7ffda8a09d8"
LOCKED_PARTIAL_FT_CANDIDATE_ID = "PFT_lr3e-5"
LOCKED_PARTIAL_FT_EPOCH = 500
LOCKED_PARTIAL_FT_SHA256 = "c95ce2c964cf75a65f24669a03c74bb43e1c212cd49040ffd516d0b3082e1a5b"

LOCKED_RUN_ROOT = LINEAR_EXPERIMENTS[LOCKED_EXPERIMENT_ID].output_root / LOCKED_RUN_ID
EXPECTED_TEST_ROWS = 2612
EXPECTED_TEST_SEQUENCES = 2607
EXPECTED_X_TEST_SHAPE = (EXPECTED_TEST_SEQUENCES, 5, 5)
EXPECTED_Y_TEST_SHAPE = (EXPECTED_TEST_SEQUENCES,)
TARGET_NAME = "DC_POWER"
TARGET_UNIT = "kW"
ENVIRONMENT_OPT_IN = "SOLAR_RUN_FINAL_TEST"
CANONICAL_ARTIFACT_FILENAMES = frozenset(
    {
        "authorization.json",
        "final_state.json",
        "wotl_metrics_original_scale.json",
        "partial_ft_metrics_original_scale.json",
        "access_log.json",
        "predictions.csv",
        "comparison.json",
    }
)


class A2FinalTestError(AssertionError):
    """Raised before an inconsistent or unauthorized A2 Final Test action."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A2FinalTestError(message)


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


@dataclass(frozen=True)
class ScalerAudit:
    target_plant: str
    target_profile: str
    feature_scaler_path: Path
    target_scaler_path: Path
    feature_scaler_sha256: str
    target_scaler_sha256: str
    feature_fit_split: str
    target_fit_split: str
    feature_fit_rows: int
    target_fit_rows: int
    separate_artifacts: bool
    time_columns_scaled: bool
    clipping_applied: bool


@dataclass(frozen=True)
class A2FinalTestPreflight:
    experiment_id: str
    run_id: str
    direction: str
    source_plant: str
    target_plant: str
    selection_git_head: str
    execution_git_head: str
    run_root: Path
    final_root: Path
    legacy_final_test_root: Path
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    wotl_checkpoint_path: Path
    wotl_checkpoint_sha256: str
    partial_ft_checkpoint_path: Path
    partial_ft_checkpoint_sha256: str
    scaler_audit: ScalerAudit
    historical_target_test_previously_revealed: bool
    test_accessed: bool = False
    test_authorized: bool = False
    structurally_ready: bool = True


@dataclass(frozen=True)
class A2FinalTestAuthorization:
    experiment_id: str
    run_id: str
    direction: str
    target_plant: str
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
    run_root: Path
    final_root: Path
    comparison_pair_locked: bool
    selection_locked: bool
    test_authorized: bool
    authorization_timestamp: str
    test_access_count: int
    post_test_tuning_allowed: bool
    historical_target_test_previously_revealed: bool
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
    target_name: str = TARGET_NAME
    scale: str = "original"
    unit: str = TARGET_UNIT


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
class A2FinalTestExecutionResult:
    authorization: A2FinalTestAuthorization
    wotl: PredictionEvaluation
    partial_ft: PredictionEvaluation
    classification: TransferClassification
    output_root: Path


class OneTimeTestAccess:
    """Allow exactly one successful invocation of a controlled Test loader."""

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


_AUTHORIZATION_BINDINGS: dict[object, tuple[Path, Path, Path]] = {}


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"Required file is missing: {path}")
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
        raise A2FinalTestError(f"Cannot read JSON {path}: {exc}") from exc
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
    run_root: Path, lifecycle_directory: str, candidate_id: str, epoch: int
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
    _require(evidence.get("validation_only") is True, f"{candidate_id} is not validation-only")
    _require(evidence.get("test_accessed") is False, f"{candidate_id} accessed Test")
    _require(evidence.get("test_metrics_used") is False, f"{candidate_id} used Test metrics")
    _require(evidence.get("git_head") == SELECTION_GIT_HEAD, f"{candidate_id} provenance mismatch")


def _validate_output_roots_available(final_root: Path, legacy_root: Path) -> None:
    for path, label in ((final_root, "A2 final"), (legacy_root, "legacy final_test")):
        if not path.exists():
            continue
        _require(path.is_dir(), f"Inconsistent {label} state: output root is not a directory")
        state_path = path / "final_state.json"
        if not state_path.is_file():
            raise A2FinalTestError(f"Inconsistent {label} state: root already exists")
        state = _load_json(state_path).get("state")
        if state == FinalTestState.TEST_ACCESS_STARTED.value:
            raise A2FinalTestError(
                "Final Test access attempt already started; pristine Test status cannot be restored"
            )
        if state == FinalTestState.TEST_ACCESSED_INCOMPLETE.value:
            raise A2FinalTestError("Target Test was accessed but evaluation is incomplete; re-run prohibited")
        if state == FinalTestState.TEST_COMPLETED.value:
            raise A2FinalTestError("Formal Final Test is already TEST_COMPLETED")
        raise A2FinalTestError(f"Inconsistent {label} state: {state!r}")


def _validate_manifest_and_output_root(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    run_root: Path,
    *,
    path_contract_resolver: Callable[[str, str], Any],
) -> tuple[Path, Path]:
    required_identity = {
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "run_id": LOCKED_RUN_ID,
        "direction": LOCKED_DIRECTION,
        "source_plant": LOCKED_SOURCE_PLANT,
        "target_plant": LOCKED_TARGET_PLANT,
        "protocol_version": LOCKED_PROTOCOL_VERSION,
        "activation": LOCKED_ACTIVATION,
    }
    for field, expected in required_identity.items():
        _require(manifest.get(field) == expected, f"Manifest {field} mismatch")
    _require(manifest.get("formal_eligible") is True, "Run is not formal eligible")
    _require(manifest.get("dry_run") is False, "Dry-run cannot authorize Final Test")
    _require(manifest.get("config_locked") is True, "Manifest config is not locked")
    _require(manifest.get("selection_locked") is True, "Manifest selection is not locked")
    _require(manifest.get("checkpoint_locked") is True, "Manifest checkpoint is not locked")
    for field in (
        "test_accessed",
        "test_metrics_used_for_selection",
        "test_authorized",
        "post_test_tuning_allowed",
        "source_test_authorized",
        "source_test_accessed",
        "target_test_authorized",
        "target_test_accessed",
        "final_test_authorized",
        "final_test_executed",
    ):
        _require(manifest.get(field) is False, f"Manifest Test-state field is not false: {field}")
    _require(
        manifest.get("historical_target_test_previously_revealed") is True,
        "Historical Plant2 Test exposure disclosure mismatch",
    )
    _require(
        manifest.get("protocol_stage")
        == ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        "Manifest lifecycle cannot authorize Test",
    )
    git = manifest.get("git")
    _require(isinstance(git, Mapping), "Manifest Git identity missing")
    _require(git.get("head") == SELECTION_GIT_HEAD, "Manifest selection Git HEAD mismatch")
    _require(
        manifest.get("target_protocol_fingerprint")
        == target_protocol_fingerprint(LOCKED_EXPERIMENT_ID),
        "Manifest Target protocol fingerprint mismatch",
    )
    _require(
        manifest.get("target_protocol_fingerprint")
        == selection.get("target_protocol_fingerprint"),
        "Manifest/selection Target protocol mismatch",
    )

    formal_paths = manifest.get("formal_paths")
    _require(isinstance(formal_paths, Mapping), "Manifest formal paths missing")
    manifest_run_root = Path(str(formal_paths.get("run_root", "")))
    _require(manifest_run_root.resolve() == run_root.resolve(), "Manifest run root mismatch")
    manifest_final = Path(str(formal_paths.get("final", "")))
    contract = path_contract_resolver(LOCKED_EXPERIMENT_ID, LOCKED_RUN_ID)
    contract_final = Path(contract.final)
    _require(
        manifest_final.resolve() == contract_final.resolve(),
        "Manifest final root disagrees with formal path contract",
    )
    _require(manifest_final.parent.resolve() == run_root.resolve(), "Final root escaped locked run")
    _require(manifest_final.name == "final", "A2 output root must use sealed final/ convention")
    legacy_final_test = run_root / "final_test"
    _validate_output_roots_available(manifest_final, legacy_final_test)
    return manifest_final, legacy_final_test


def _validate_real_target_scalers(manifest: Mapping[str, Any]) -> ScalerAudit:
    import joblib

    spec = LINEAR_EXPERIMENTS[LOCKED_EXPERIMENT_ID]
    profile_dir = spec.target_profile_path
    _require(spec.target_plant == LOCKED_TARGET_PLANT, "Configured A2 Target plant mismatch")
    _require(
        Path(str(manifest.get("target_profile_path", ""))).resolve()
        == profile_dir.resolve(),
        "Manifest A2 Target profile path mismatch",
    )
    scaler_manifest = _load_json(profile_dir / "scaler_manifest.json")
    _require(scaler_manifest.get("plant") == LOCKED_TARGET_PLANT, "Target scaler is not Plant2")
    _require(scaler_manifest.get("profile") == "target", "Target scaler profile mismatch")
    _require(
        scaler_manifest.get("feature_scaler_fit_split") == "training",
        "Feature scaler was not fit on training",
    )
    _require(
        scaler_manifest.get("target_scaler_fit_split") == "training",
        "Target scaler was not fit on training",
    )
    _require(
        scaler_manifest.get("feature_scaler_n_samples_seen") == 521,
        "Feature scaler fit-row count mismatch",
    )
    _require(
        scaler_manifest.get("target_scaler_n_samples_seen") == 521,
        "Target scaler fit-row count mismatch",
    )
    _require(scaler_manifest.get("time_columns_scaled") is False, "Cyclic time columns changed")
    _require(scaler_manifest.get("clipping_applied") is False, "Scaler clipping is enabled")

    feature_path = profile_dir / str(scaler_manifest.get("feature_scaler_file", ""))
    target_path = profile_dir / str(scaler_manifest.get("target_scaler_file", ""))
    _require(feature_path.resolve() != target_path.resolve(), "Feature/target scaler artifacts overlap")
    feature = joblib.load(feature_path)
    target = joblib.load(target_path)
    _require(feature is not target, "Feature/target scaler objects overlap")
    _require(int(np.asarray(feature.n_samples_seen_).reshape(-1)[0]) == 521, "Feature scaler object row mismatch")
    _require(int(np.asarray(target.n_samples_seen_).reshape(-1)[0]) == 521, "Target scaler object row mismatch")
    _require(int(feature.n_features_in_) == 3, "Feature scaler dimension mismatch")
    _require(int(target.n_features_in_) == 1, "Target scaler dimension mismatch")
    _require(feature.clip is False and target.clip is False, "Scaler object clipping is enabled")

    feature_sha = _sha256(feature_path)
    target_sha = _sha256(target_path)
    manifest_feature_sha = manifest.get("feature_scaler_sha256")
    manifest_target_sha = manifest.get("target_scaler_sha256")
    _require(isinstance(manifest_feature_sha, Mapping), "Feature scaler SHA evidence missing")
    _require(isinstance(manifest_target_sha, Mapping), "Target scaler SHA evidence missing")
    _require(manifest_feature_sha.get("target") == feature_sha, "Feature scaler SHA mismatch")
    _require(manifest_target_sha.get("target") == target_sha, "Target scaler SHA mismatch")

    feature_provenance = manifest.get("feature_scaler_provenance")
    target_provenance = manifest.get("target_scaler_provenance")
    _require(isinstance(feature_provenance, Mapping), "Feature scaler provenance missing")
    _require(isinstance(target_provenance, Mapping), "Target scaler provenance missing")
    feature_target = feature_provenance.get("target")
    target_target = target_provenance.get("target")
    _require(isinstance(feature_target, Mapping), "Target feature-scaler provenance missing")
    _require(isinstance(target_target, Mapping), "Target target-scaler provenance missing")
    _require(feature_target.get("fit_split") == "training", "Feature provenance fit split mismatch")
    _require(target_target.get("fit_split") == "training", "Target provenance fit split mismatch")
    _require(feature_target.get("fit_row_count") == 521, "Feature provenance fit rows mismatch")
    _require(target_target.get("fit_row_count") == 521, "Target provenance fit rows mismatch")
    _require(Path(str(feature_target.get("path", ""))).resolve() == feature_path.resolve(), "Feature scaler provenance path mismatch")
    _require(Path(str(target_target.get("path", ""))).resolve() == target_path.resolve(), "Target scaler provenance path mismatch")
    return ScalerAudit(
        target_plant=LOCKED_TARGET_PLANT,
        target_profile="target",
        feature_scaler_path=feature_path,
        target_scaler_path=target_path,
        feature_scaler_sha256=feature_sha,
        target_scaler_sha256=target_sha,
        feature_fit_split="training",
        target_fit_split="training",
        feature_fit_rows=521,
        target_fit_rows=521,
        separate_artifacts=True,
        time_columns_scaled=False,
        clipping_applied=False,
    )


def _validate_scaler_audit(audit: ScalerAudit) -> None:
    _require(audit.target_plant == LOCKED_TARGET_PLANT, "Plant1/B scaler provenance rejected")
    _require(audit.target_profile == "target", "Target scaler profile mismatch")
    _require(audit.separate_artifacts, "Feature/target scalers are not separate")
    _require(audit.feature_fit_split == "training", "Feature scaler fit split mismatch")
    _require(audit.target_fit_split == "training", "Target scaler fit split mismatch")
    _require(audit.feature_fit_rows == 521, "Feature scaler fit rows mismatch")
    _require(audit.target_fit_rows == 521, "Target scaler fit rows mismatch")
    _require(not audit.time_columns_scaled, "Cyclic time columns must remain unscaled")
    _require(not audit.clipping_applied, "Scaler clipping must remain disabled")
    _require(audit.feature_scaler_path.resolve() != audit.target_scaler_path.resolve(), "Scaler paths overlap")


def preflight_a2_final_test(
    experiment_id: str = LOCKED_EXPERIMENT_ID,
    run_id: str = LOCKED_RUN_ID,
    *,
    expected_execution_git_head: str | None = None,
    run_root: Path | str | None = None,
    git_state: GitState | None = None,
    sha256_func: Callable[[Path], str] = _sha256,
    selection_validator: Callable[[Mapping[str, Any]], None] = validate_selection_record,
    scaler_validator: Callable[[Mapping[str, Any]], ScalerAudit] = _validate_real_target_scalers,
    path_contract_resolver: Callable[[str, str], Any] = formal_path_contract,
    _allow_test_root: bool = False,
) -> A2FinalTestPreflight:
    """Validate the real or synthetic locked A2 run without authorizing Test."""

    _require(experiment_id == LOCKED_EXPERIMENT_ID, "Only Experiment A2 is supported")
    _require(run_id == LOCKED_RUN_ID, "A2 Final Test run ID mismatch")
    root = Path(run_root) if run_root is not None else LOCKED_RUN_ROOT
    if not _allow_test_root:
        _require(root.resolve() == LOCKED_RUN_ROOT.resolve(), "Unexpected A2 Formal run root")
        _require(git_state is None, "Production preflight requires runtime Git collection")
        _require(sha256_func is _sha256, "Production preflight requires real SHA-256")
        _require(selection_validator is validate_selection_record, "Production preflight requires committed selection validator")
        _require(scaler_validator is _validate_real_target_scalers, "Production preflight requires real scaler validator")
        _require(path_contract_resolver is formal_path_contract, "Production preflight requires formal path contract")

    state = git_state or collect_git_state()
    _require(not state.tracked_dirty, "Tracked working tree is dirty")
    if expected_execution_git_head is not None:
        _require(state.head == expected_execution_git_head, "Execution Git HEAD mismatch")

    selection = _load_json(root / "selection" / "selection.json")
    manifest = _load_json(root / "protocol_manifest.json")
    try:
        selection_validator(selection)
    except FormalProtocolError as exc:
        raise A2FinalTestError(f"Selection validation failed: {exc}") from exc
    final_root, legacy_final_test = _validate_manifest_and_output_root(
        manifest,
        selection,
        root,
        path_contract_resolver=path_contract_resolver,
    )

    _require(selection.get("experiment_id") == LOCKED_EXPERIMENT_ID, "Selection experiment mismatch")
    _require(selection.get("run_id") == LOCKED_RUN_ID, "Selection run ID mismatch")
    _require(selection.get("protocol_version") == LOCKED_PROTOCOL_VERSION, "Selection protocol mismatch")
    _require(selection.get("git_head") == SELECTION_GIT_HEAD, "Selection/training provenance mismatch")
    _require(selection.get("config_locked") is True, "Selection config is not locked")
    _require(selection.get("selection_locked") is True, "Selection is not locked")
    _require(selection.get("checkpoint_locked") is True, "Checkpoint is not locked")
    _require(selection.get("comparison_pair_locked") is True, "Comparison pair is not locked")
    for field in (
        "test_metrics_used_for_selection",
        "test_accessed",
        "test_authorized",
        "post_test_tuning_allowed",
    ):
        _require(selection.get(field) is False, f"Selection Test-state field is not false: {field}")

    source_path = _checkpoint_path(root, "source_candidates", LOCKED_SOURCE_CANDIDATE_ID, LOCKED_SOURCE_EPOCH)
    wotl_path = _checkpoint_path(root, "target_wotl_candidates", LOCKED_WOTL_CANDIDATE_ID, LOCKED_WOTL_EPOCH)
    partial_path = _checkpoint_path(root, "target_partial_ft_candidates", LOCKED_PARTIAL_FT_CANDIDATE_ID, LOCKED_PARTIAL_FT_EPOCH)
    locked = (
        (
            "source_selected_candidate",
            "source_checkpoint_sha256",
            "source_validation_comparison",
            LOCKED_SOURCE_CANDIDATE_ID,
            LOCKED_SOURCE_EPOCH,
            LOCKED_SOURCE_SHA256,
            source_path,
        ),
        (
            "wotl_selected_candidate",
            "wotl_selected_checkpoint_sha256",
            "wotl_validation_comparison",
            LOCKED_WOTL_CANDIDATE_ID,
            LOCKED_WOTL_EPOCH,
            LOCKED_WOTL_SHA256,
            wotl_path,
        ),
        (
            "partial_ft_selected_candidate",
            "partial_ft_selected_checkpoint_sha256",
            "partial_ft_validation_comparison",
            LOCKED_PARTIAL_FT_CANDIDATE_ID,
            LOCKED_PARTIAL_FT_EPOCH,
            LOCKED_PARTIAL_FT_SHA256,
            partial_path,
        ),
    )
    for selected_field, sha_field, comparison_field, candidate, epoch, expected_sha, path in locked:
        _require(selection.get(selected_field) == candidate, f"{candidate} selection mismatch")
        _require(selection.get(sha_field) == expected_sha, f"{candidate} selected SHA mismatch")
        _validate_locked_candidate(
            _selected_comparison(selection, comparison_field, candidate),
            candidate_id=candidate,
            epoch=epoch,
            checkpoint_path=path,
            checkpoint_sha256=expected_sha,
        )
        _require(path.is_file() and path.stat().st_size > 0, f"Checkpoint missing/empty: {path}")
        _require(sha256_func(path) == expected_sha, f"Actual checkpoint SHA mismatch: {candidate}")

    scaler_audit = scaler_validator(manifest)
    _validate_scaler_audit(scaler_audit)
    return A2FinalTestPreflight(
        experiment_id=LOCKED_EXPERIMENT_ID,
        run_id=LOCKED_RUN_ID,
        direction=LOCKED_DIRECTION,
        source_plant=LOCKED_SOURCE_PLANT,
        target_plant=LOCKED_TARGET_PLANT,
        selection_git_head=SELECTION_GIT_HEAD,
        execution_git_head=state.head,
        run_root=root,
        final_root=final_root,
        legacy_final_test_root=legacy_final_test,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=LOCKED_SOURCE_SHA256,
        wotl_checkpoint_path=wotl_path,
        wotl_checkpoint_sha256=LOCKED_WOTL_SHA256,
        partial_ft_checkpoint_path=partial_path,
        partial_ft_checkpoint_sha256=LOCKED_PARTIAL_FT_SHA256,
        scaler_audit=scaler_audit,
        historical_target_test_previously_revealed=True,
    )


def prepare_a2_final_test_authorization(
    *,
    human_authorized: bool,
    expected_execution_git_head: str,
    experiment_id: str = LOCKED_EXPERIMENT_ID,
    run_id: str = LOCKED_RUN_ID,
    run_root: Path | str | None = None,
    git_state: GitState | None = None,
    authorization_timestamp: str | None = None,
    sha256_func: Callable[[Path], str] = _sha256,
    selection_validator: Callable[[Mapping[str, Any]], None] = validate_selection_record,
    scaler_validator: Callable[[Mapping[str, Any]], ScalerAudit] = _validate_real_target_scalers,
    path_contract_resolver: Callable[[str, str], Any] = formal_path_contract,
    _allow_test_root: bool = False,
) -> A2FinalTestAuthorization:
    """Issue an in-memory authorization; never open Test or write an artifact."""

    _require(human_authorized is True, "Explicit human Final Test authorization required")
    _require(
        isinstance(expected_execution_git_head, str)
        and len(expected_execution_git_head) == 40
        and all(character in "0123456789abcdef" for character in expected_execution_git_head),
        "Expected execution Git HEAD must be a lowercase 40-character SHA",
    )
    preflight = preflight_a2_final_test(
        experiment_id,
        run_id,
        expected_execution_git_head=expected_execution_git_head,
        run_root=run_root,
        git_state=git_state,
        sha256_func=sha256_func,
        selection_validator=selection_validator,
        scaler_validator=scaler_validator,
        path_contract_resolver=path_contract_resolver,
        _allow_test_root=_allow_test_root,
    )
    timestamp = authorization_timestamp or _utc_timestamp()
    proof = object()
    _AUTHORIZATION_BINDINGS[proof] = (
        preflight.wotl_checkpoint_path.resolve(),
        preflight.partial_ft_checkpoint_path.resolve(),
        preflight.final_root.resolve(),
    )
    return A2FinalTestAuthorization(
        experiment_id=LOCKED_EXPERIMENT_ID,
        run_id=LOCKED_RUN_ID,
        direction=LOCKED_DIRECTION,
        target_plant=LOCKED_TARGET_PLANT,
        selection_git_head=SELECTION_GIT_HEAD,
        execution_git_head=preflight.execution_git_head,
        source_candidate_id=LOCKED_SOURCE_CANDIDATE_ID,
        source_checkpoint_sha256=LOCKED_SOURCE_SHA256,
        wotl_candidate_id=LOCKED_WOTL_CANDIDATE_ID,
        wotl_checkpoint_path=preflight.wotl_checkpoint_path,
        wotl_checkpoint_sha256=LOCKED_WOTL_SHA256,
        partial_ft_candidate_id=LOCKED_PARTIAL_FT_CANDIDATE_ID,
        partial_ft_checkpoint_path=preflight.partial_ft_checkpoint_path,
        partial_ft_checkpoint_sha256=LOCKED_PARTIAL_FT_SHA256,
        run_root=preflight.run_root,
        final_root=preflight.final_root,
        comparison_pair_locked=True,
        selection_locked=True,
        test_authorized=True,
        authorization_timestamp=timestamp,
        test_access_count=0,
        post_test_tuning_allowed=False,
        historical_target_test_previously_revealed=True,
        _proof=proof,
    )


def validate_authorization(authorization: A2FinalTestAuthorization) -> None:
    _require(isinstance(authorization, A2FinalTestAuthorization), "Invalid A2 authorization")
    binding = _AUTHORIZATION_BINDINGS.get(authorization._proof)
    _require(binding is not None, "Authorization proof is invalid")
    _require(authorization.experiment_id == LOCKED_EXPERIMENT_ID, "Authorization experiment mismatch")
    _require(authorization.run_id == LOCKED_RUN_ID, "Authorization run mismatch")
    _require(authorization.direction == LOCKED_DIRECTION, "Authorization direction mismatch")
    _require(authorization.target_plant == LOCKED_TARGET_PLANT, "Authorization Target mismatch")
    _require(authorization.selection_git_head == SELECTION_GIT_HEAD, "Authorization selection provenance mismatch")
    _require(authorization.source_candidate_id == LOCKED_SOURCE_CANDIDATE_ID, "Authorization Source mismatch")
    _require(authorization.source_checkpoint_sha256 == LOCKED_SOURCE_SHA256, "Authorization Source SHA mismatch")
    _require(authorization.wotl_candidate_id == LOCKED_WOTL_CANDIDATE_ID, "Authorization WOTL mismatch")
    _require(authorization.wotl_checkpoint_sha256 == LOCKED_WOTL_SHA256, "Authorization WOTL SHA mismatch")
    _require(authorization.partial_ft_candidate_id == LOCKED_PARTIAL_FT_CANDIDATE_ID, "Authorization PFT mismatch")
    _require(authorization.partial_ft_checkpoint_sha256 == LOCKED_PARTIAL_FT_SHA256, "Authorization PFT SHA mismatch")
    _require(
        binding
        == (
            authorization.wotl_checkpoint_path.resolve(),
            authorization.partial_ft_checkpoint_path.resolve(),
            authorization.final_root.resolve(),
        ),
        "Authorization path binding mismatch",
    )
    _require(authorization.final_root.parent.resolve() == authorization.run_root.resolve(), "Authorization final root escaped run")
    _require(authorization.final_root.name == "final", "Authorization output root mismatch")
    _require(authorization.comparison_pair_locked, "Authorization comparison pair is unlocked")
    _require(authorization.selection_locked, "Authorization selection is unlocked")
    _require(authorization.test_authorized, "Final Test is not authorized")
    _require(authorization.test_access_count == 0, "Authorization already consumed Test")
    _require(not authorization.test_accessed, "Authorization reports prior Test access")
    _require(not authorization.test_completed, "Authorization is already completed")
    _require(not authorization.post_test_tuning_allowed, "Authorization permits post-Test tuning")
    _require(authorization.historical_target_test_previously_revealed, "Historical disclosure lost")
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


def load_locked_a2_target_test() -> TargetTestData:
    """Sole production loader for corrected Plant2 Target Test; execution only."""

    from r2_helpers import solar_data

    spec = LINEAR_EXPERIMENTS[LOCKED_EXPERIMENT_ID]
    profile_dir = spec.target_profile_path
    checksums = solar_data.validate_checksums_for_splits(profile_dir, splits=("test",))
    metadata = solar_data.load_profile_metadata(profile_dir)
    solar_data.validate_manifest(
        metadata,
        expected_plant=LOCKED_TARGET_PLANT,
        expected_profile="target",
        expected_profile_dir_name="target_profile",
    )
    scalers = solar_data.validate_scalers(metadata)
    split = solar_data.load_split(metadata, "test")
    solar_data.validate_feature_transform_consistency(split, scalers.feature_scaler)
    solar_data.validate_target_round_trip(split, scalers.target_scaler)
    sequences = solar_data.build_sequences(split.X_normalized, split.y_normalized, split.timestamps)
    _require(split.row_count == EXPECTED_TEST_ROWS, "Target Test row count mismatch")
    data = TargetTestData(
        X_test=sequences.X_seq,
        y_test=sequences.y_seq,
        target_timestamp=sequences.target_timestamp,
        target_scaler=scalers.target_scaler,
        accessed_files=(profile_dir / "file_checksums.csv", *(profile_dir / name for name in sorted(checksums))),
    )
    validate_target_test_data(data)
    return data


def _default_model_loader(path: Path) -> Any:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from keras.models import load_model
    from r2_helpers.solar_linear_runtime import rmse

    return load_model(str(path), custom_objects={"rmse": rmse}, compile=False)


def _default_model_validator(model: Any) -> None:
    from r2_helpers.solar_linear_runtime import validate_linear_model

    validate_linear_model(model)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum(np.square(y_true - y_pred)))
    total = float(np.sum(np.square(y_true - np.mean(y_true))))
    _require(total > 0, "Target variance is zero")
    return 1.0 - residual / total


def compute_original_scale_metrics(
    y_true_original: np.ndarray, prediction_original: np.ndarray
) -> OriginalScaleMetrics:
    truth = np.asarray(y_true_original, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_original, dtype=np.float64).reshape(-1)
    _require(truth.shape == prediction.shape and len(truth) > 0, "Prediction alignment mismatch")
    _require(np.isfinite(truth).all() and np.isfinite(prediction).all(), "Non-finite metric input")
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
    _require(np.isfinite(truth).all() and np.isfinite(prediction).all(), "Non-finite normalized input")
    error = truth - prediction
    normalized_mse = float(np.mean(np.square(error)))
    truth_original = np.asarray(target_scaler.inverse_transform(truth.reshape(-1, 1)), dtype=np.float64).reshape(-1)
    prediction_original = np.asarray(target_scaler.inverse_transform(prediction.reshape(-1, 1)), dtype=np.float64).reshape(-1)
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
    wotl: OriginalScaleMetrics, partial_ft: OriginalScaleMetrics
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


def _authorization_payload(authorization: A2FinalTestAuthorization) -> dict[str, Any]:
    payload = asdict(authorization)
    payload.pop("_proof")
    return payload


def _final_state_payload(
    authorization: A2FinalTestAuthorization,
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


def execute_a2_final_test(
    authorization: A2FinalTestAuthorization,
    *,
    execute_test: bool,
    output_root: Path | str | None = None,
    test_loader: Callable[[], TargetTestData] = load_locked_a2_target_test,
    model_loader: Callable[[Path], Any] = _default_model_loader,
    model_validator: Callable[[Any], None] = _default_model_validator,
    sha256_func: Callable[[Path], str] = _sha256,
    git_state_provider: Callable[[], GitState] = collect_git_state,
    _allow_test_dependencies: bool = False,
) -> A2FinalTestExecutionResult | None:
    """Execute one locked A2 WOTL/PFT comparison after dual opt-in."""

    validate_authorization(authorization)
    if not execute_test:
        return None
    _require(os.environ.get(ENVIRONMENT_OPT_IN) == "1", f"{ENVIRONMENT_OPT_IN}=1 opt-in missing")
    root = Path(output_root) if output_root is not None else authorization.final_root
    if not _allow_test_dependencies:
        _require(root.resolve() == authorization.final_root.resolve(), "Unexpected A2 Final output root")
        _require(test_loader is load_locked_a2_target_test, "Production requires locked Plant2 Test loader")
        _require(model_loader is _default_model_loader, "Production requires locked model loader")
        _require(model_validator is _default_model_validator, "Production requires linear model validator")
        _require(sha256_func is _sha256, "Production requires real checkpoint SHA-256")
        _require(git_state_provider is collect_git_state, "Production requires runtime Git collection")
    _require(root.resolve() == authorization.final_root.resolve(), "Output root differs from sealed A2 final root")
    current_git = git_state_provider()
    _require(current_git.head == authorization.execution_git_head, "Final Test execute-time Git HEAD mismatch")
    _require(not current_git.tracked_dirty, "Tracked working tree became dirty before Final Test")
    _validate_output_roots_available(root, authorization.run_root / "final_test")

    for path, expected in (
        (authorization.wotl_checkpoint_path, LOCKED_WOTL_SHA256),
        (authorization.partial_ft_checkpoint_path, LOCKED_PARTIAL_FT_SHA256),
    ):
        _require(_sha256(path) == expected if sha256_func is _sha256 else sha256_func(path) == expected, "Checkpoint changed after authorization")

    wotl_model = model_loader(authorization.wotl_checkpoint_path)
    partial_model = model_loader(authorization.partial_ft_checkpoint_path)
    model_validator(wotl_model)
    model_validator(partial_model)
    _require(callable(getattr(wotl_model, "predict", None)), "WOTL model is not usable")
    _require(callable(getattr(partial_model, "predict", None)), "Partial FT model is not usable")

    root.mkdir(parents=False, exist_ok=False)
    _write_json_exclusive(root / "authorization.json", _authorization_payload(authorization))
    access_started_at = _utc_timestamp()
    _write_json_exclusive(
        root / "final_state.json",
        _final_state_payload(
            authorization,
            state=FinalTestState.TEST_ACCESS_STARTED,
            test_accessed=False,
            test_access_count=0,
            test_completed=False,
            test_access_attempt_count=1,
            test_access_start_timestamp=access_started_at,
        ),
    )
    access = OneTimeTestAccess(test_loader)
    try:
        data = access.load()
    except Exception as exc:
        raise A2FinalTestError("Final Test loader failed after TEST_ACCESS_STARTED; re-run prohibited") from exc
    _require(access.access_count == 1, "Target Test access count must equal one")
    accessed_at = _utc_timestamp()
    _write_json_atomic_replace(
        root / "final_state.json",
        _final_state_payload(
            authorization,
            state=FinalTestState.TEST_ACCESSED_INCOMPLETE,
            test_accessed=True,
            test_access_count=1,
            test_completed=False,
            test_access_attempt_count=1,
            test_access_start_timestamp=access_started_at,
            test_access_timestamp=accessed_at,
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
        _require(len(partial_prediction) == EXPECTED_TEST_SEQUENCES, "PFT prediction count mismatch")
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
                "test_access_count": 1,
                "test_accessed": True,
                "test_completed": True,
                "opened_data_files": data.accessed_files,
                "post_test_tuning_allowed": False,
                "test_access_timestamp": accessed_at,
            },
        )
        _write_predictions(root / "predictions.csv", data, wotl, partial)
        _write_json_exclusive(
            root / "comparison.json",
            {
                "experiment_id": LOCKED_EXPERIMENT_ID,
                "run_id": LOCKED_RUN_ID,
                "direction": LOCKED_DIRECTION,
                "target_plant": LOCKED_TARGET_PLANT,
                "selection_git_head": SELECTION_GIT_HEAD,
                "execution_git_head": sealed.execution_git_head,
                "wotl_candidate_id": LOCKED_WOTL_CANDIDATE_ID,
                "partial_ft_candidate_id": LOCKED_PARTIAL_FT_CANDIDATE_ID,
                "metrics_scale": "original",
                "target_name": TARGET_NAME,
                "unit": TARGET_UNIT,
                "classification": classification,
                "wotl_metrics_original_scale": asdict(wotl.metrics),
                "partial_ft_metrics_original_scale": asdict(partial.metrics),
                "post_test_tuning_allowed": False,
                "historical_target_test_previously_revealed": True,
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
                test_access_start_timestamp=access_started_at,
                test_access_timestamp=accessed_at,
            ),
        )
    except Exception as exc:
        raise A2FinalTestError("Final Test failed after durable Test access; state remains TEST_ACCESSED_INCOMPLETE") from exc
    return A2FinalTestExecutionResult(
        authorization=sealed,
        wotl=wotl,
        partial_ft=partial,
        classification=classification,
        output_root=root,
    )
