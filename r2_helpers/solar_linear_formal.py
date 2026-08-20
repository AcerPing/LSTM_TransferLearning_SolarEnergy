"""Phase-III dry-run contracts for Solar Linear formal training.

This module prepares immutable policy, callback, provenance, selection, and
authorization objects.  It intentionally has no training or inference entry
point and never creates the formal namespace.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from r2_config.solar_linear import (
    EXPECTED_TRAINABLE_PARAMS,
    EXPECTED_TOTAL_PARAMS,
    LINEAR_EXPERIMENTS,
    PARTIAL_FT_STRATEGY_ID,
    PROTECTED_LEGACY_ROOTS,
    REPOSITORY_ROOT,
)
from r2_config.solar_linear_formal import (
    FORMAL_CALLBACK_POLICY,
    FORMAL_DEVICE_POLICY,
    FORMAL_LIFECYCLES,
    FORMAL_LINEAR_LAYER_CLASSES,
    FORMAL_PROTOCOL_VERSION,
    HISTORICAL_TARGET_TEST_PREVIOUSLY_REVEALED,
    LEARNING_RATE_CANDIDATES,
    LINEAR_SMOKE_BASE,
    RUN_ID_PATTERN,
    VALIDATION_DIAGNOSTIC_TIE_RTOL,
    VALIDATION_LOSS_TIE_RTOL,
    VALIDATION_TIE_ATOL,
    FormalPathContract,
    candidate_id,
    candidate_registry,
    formal_path_contract,
    validate_formal_policy,
    validate_new_run_destination,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_PROTOCOL_MANIFEST_FIELDS = (
    "protocol_version",
    "target_protocol_fingerprint",
    "experiment_id",
    "run_id",
    "direction",
    "source_plant",
    "target_plant",
    "source_profile_path",
    "target_profile_path",
    "activation",
    "features",
    "target",
    "window",
    "horizon",
    "frequency",
    "seed",
    "batch_size",
    "optimizer",
    "loss",
    "learning_rate_candidates",
    "candidate_registry",
    "callback_policy",
    "partial_ft_strategy_id",
    "transferred_layers",
    "trainable_layers",
    "frozen_layers",
    "bn_frozen",
    "training_rows",
    "validation_rows",
    "test_rows",
    "training_sequences",
    "validation_sequences",
    "test_sequences",
    "feature_scaler_sha256",
    "feature_scaler_provenance",
    "target_scaler_sha256",
    "target_scaler_provenance",
    "git",
    "environment",
    "device_policy",
    "historical_target_test_previously_revealed",
    "test_accessed",
    "test_metrics_used_for_selection",
    "selection_locked",
    "config_locked",
    "checkpoint_locked",
    "test_authorized",
    "post_test_tuning_allowed",
    "formal_eligible",
    "dry_run",
    "protocol_stage",
)
REQUIRED_SELECTION_FIELDS = (
    "protocol_version",
    "experiment_id",
    "run_id",
    "git_head",
    "target_protocol_fingerprint",
    "selection_timestamp",
    "source_selected_candidate",
    "source_checkpoint_sha256",
    "source_checkpoint_sha256_verified",
    "wotl_selected_candidate",
    "wotl_selected_checkpoint_sha256",
    "wotl_selected_checkpoint_sha256_verified",
    "partial_ft_selected_candidate",
    "partial_ft_selected_checkpoint_sha256",
    "partial_ft_selected_checkpoint_sha256_verified",
    "comparison_pair_locked",
    "selection_basis",
    "source_validation_comparison",
    "wotl_validation_comparison",
    "partial_ft_validation_comparison",
    "config_locked",
    "selection_locked",
    "checkpoint_locked",
    "test_metrics_used_for_selection",
    "test_accessed",
    "test_authorized",
    "post_test_tuning_allowed",
    "protocol_stage",
)


class FormalProtocolError(AssertionError):
    """Raised when a locked formal protocol invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalProtocolError(message)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GitIdentity:
    head: str
    branch: str
    dirty: bool


@dataclass(frozen=True)
class ScalerProvenance:
    profile_path: Path
    feature_scaler_path: Path
    feature_scaler_sha256: str
    target_scaler_path: Path
    target_scaler_sha256: str
    fit_split: str
    fit_row_count: int
    accessed_files: tuple[Path, ...]


@dataclass(frozen=True)
class ValidationCandidateScore:
    protocol_version: str
    experiment_id: str
    run_id: str
    git_head: str
    target_protocol_fingerprint: str
    lifecycle: str
    candidate_id: str
    learning_rate: float
    best_epoch: int
    validation_loss: float
    validation_original_mae: float
    validation_original_rmse: float
    validation_original_r2: float
    trainable_params: int
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_sha256_verified: bool
    validation_only: bool = True
    test_accessed: bool = False
    test_metrics_used: bool = False


def _coerce_validation_candidate_score(value: Any) -> ValidationCandidateScore:
    """Restore one score from the exact dataclass or JSON-compatible shape."""

    if isinstance(value, ValidationCandidateScore):
        return value
    _require(isinstance(value, Mapping), "Validation comparison entry must be a mapping")
    expected_fields = {item.name for item in fields(ValidationCandidateScore)}
    actual_fields = set(value)
    missing = sorted(expected_fields - actual_fields)
    unknown = sorted(actual_fields - expected_fields)
    _require(not missing, f"Validation comparison entry missing fields: {missing}")
    _require(not unknown, f"Validation comparison entry has unknown fields: {unknown}")

    payload = dict(value)
    string_fields = (
        "protocol_version",
        "experiment_id",
        "run_id",
        "git_head",
        "target_protocol_fingerprint",
        "lifecycle",
        "candidate_id",
        "checkpoint_sha256",
    )
    for field_name in string_fields:
        _require(isinstance(payload[field_name], str), f"{field_name} must be a string")
    for field_name in (
        "learning_rate",
        "validation_loss",
        "validation_original_mae",
        "validation_original_rmse",
        "validation_original_r2",
    ):
        _require(
            isinstance(payload[field_name], (int, float))
            and not isinstance(payload[field_name], bool),
            f"{field_name} must be numeric",
        )
        payload[field_name] = float(payload[field_name])
    for field_name in ("best_epoch", "trainable_params"):
        _require(
            isinstance(payload[field_name], int)
            and not isinstance(payload[field_name], bool),
            f"{field_name} must be an integer",
        )
    for field_name in (
        "checkpoint_sha256_verified",
        "validation_only",
        "test_accessed",
        "test_metrics_used",
    ):
        _require(isinstance(payload[field_name], bool), f"{field_name} must be boolean")
    checkpoint_path = payload["checkpoint_path"]
    _require(
        isinstance(checkpoint_path, (str, Path)),
        "checkpoint_path must be a string or Path",
    )
    payload["checkpoint_path"] = Path(checkpoint_path)
    return ValidationCandidateScore(**payload)


@dataclass(frozen=True)
class SourceCheckpointProvenance:
    protocol_version: str
    experiment_id: str
    source_run_id: str
    source_candidate_id: str
    source_activation: str
    source_learning_rate: float
    source_best_epoch: int
    source_validation_loss: float
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    source_checkpoint_sha256_verified: bool
    source_git_head: str
    source_training_profile: Path
    source_validation_profile: Path
    source_layer_classes: tuple[str, ...]
    source_input_shape: tuple[int, int]
    source_output_shape: tuple[int, ...]
    source_total_params: int
    validation_selected: bool
    formal_eligible: bool


@dataclass(frozen=True)
class SourceArchitectureEvidence:
    activation: str
    git_head: str
    layer_classes: tuple[str, ...]
    input_shape: tuple[int, int]
    output_shape: tuple[int, ...]
    total_params: int
    formal_eligible: bool


class ProtocolStage(str, Enum):
    CONFIG_LOCKED_AWAITING_TRAINING = "config_locked_awaiting_training"
    SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION = (
        "selection_locked_awaiting_test_authorization"
    )
    TEST_AUTHORIZED = "test_authorized"
    TEST_COMPLETED = "test_completed"


@dataclass(frozen=True)
class SelectionRecord:
    protocol_version: str
    experiment_id: str
    run_id: str
    git_head: str
    target_protocol_fingerprint: str
    selection_timestamp: str
    source_selected_candidate: str
    source_checkpoint_sha256: str
    source_checkpoint_sha256_verified: bool
    wotl_selected_candidate: str
    wotl_selected_checkpoint_sha256: str
    wotl_selected_checkpoint_sha256_verified: bool
    partial_ft_selected_candidate: str
    partial_ft_selected_checkpoint_sha256: str
    partial_ft_selected_checkpoint_sha256_verified: bool
    comparison_pair_locked: bool
    selection_basis: tuple[str, ...]
    source_validation_comparison: tuple[ValidationCandidateScore, ...]
    wotl_validation_comparison: tuple[ValidationCandidateScore, ...]
    partial_ft_validation_comparison: tuple[ValidationCandidateScore, ...]
    config_locked: bool
    selection_locked: bool
    checkpoint_locked: bool
    test_metrics_used_for_selection: bool
    test_accessed: bool
    test_authorized: bool
    post_test_tuning_allowed: bool
    protocol_stage: ProtocolStage


_AUTHORIZATION_PROOF = object()


@dataclass(frozen=True)
class FinalTestAuthorization:
    experiment_id: str
    run_id: str
    wotl_selected_candidate: str
    wotl_checkpoint_sha256: str
    partial_ft_selected_candidate: str
    partial_ft_checkpoint_sha256: str
    comparison_pair_locked: bool
    protocol_stage: ProtocolStage
    test_authorized: bool
    _proof: object


@dataclass(frozen=True)
class FormalDryRun:
    paths: FormalPathContract
    manifest: Mapping[str, Any]
    candidate_checkpoint_patterns: Mapping[str, Path]
    callbacks: Mapping[str, tuple[Any, ...]]
    accessed_files: tuple[Path, ...]
    formal_root_created: bool
    training_epochs_executed: int = 0


def target_protocol_fingerprint(experiment_id: str) -> str:
    """Identify the immutable Target-side protocol without reading any split."""

    _require(experiment_id in LINEAR_EXPERIMENTS, "Unknown Target experiment")
    spec = LINEAR_EXPERIMENTS[experiment_id]
    payload = {
        "protocol_version": FORMAL_PROTOCOL_VERSION,
        "experiment_id": experiment_id,
        "direction": spec.direction,
        "target_plant": spec.target_plant,
        "target_profile_path": spec.target_profile_path.as_posix(),
        "activation": spec.activation,
        "features": spec.features,
        "target": spec.target_column,
        "window": spec.window,
        "horizon": spec.horizon,
        "frequency": spec.frequency,
        "seed": spec.seed,
        "batch_size": spec.batch_size,
        "optimizer": spec.optimizer,
        "loss": spec.loss,
        "partial_ft_strategy_id": spec.partial_ft_strategy_id,
        "transferred_layers": spec.transferred_indices,
        "trainable_layers": spec.trainable_indices,
        "frozen_layers": spec.frozen_indices,
        "bn_frozen": spec.batch_normalization_frozen,
        "callback_policy": asdict(FORMAL_CALLBACK_POLICY),
        "device_policy": asdict(FORMAL_DEVICE_POLICY),
        "candidate_registry": {
            lifecycle: entries for lifecycle, entries in candidate_registry().items()
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def collect_git_identity() -> GitIdentity:
    safe_argument = f"safe.directory={REPOSITORY_ROOT.as_posix()}"

    def git(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", "-c", safe_argument, *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
        )
        _require(result.returncode == 0, f"Git command failed: {' '.join(arguments)}")
        return result.stdout

    return GitIdentity(
        head=git("rev-parse", "HEAD").strip().decode("ascii"),
        branch=git("branch", "--show-current").strip().decode("utf-8"),
        dirty=bool(git("status", "--porcelain")),
    )


def _scaler_provenance(profile_path: Path) -> ScalerProvenance:
    feature_path = profile_path / "feature_scaler.joblib"
    target_path = profile_path / "target_scaler.joblib"
    manifest_path = profile_path / "scaler_manifest.json"
    for path in (feature_path, target_path, manifest_path):
        _require(path.is_file(), f"Required scaler provenance file missing: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalProtocolError(f"Cannot read scaler provenance: {exc}") from exc
    _require(
        manifest.get("feature_scaler_fit_split") == "training"
        and manifest.get("target_scaler_fit_split") == "training",
        "Scalers were not fit on Training",
    )
    feature_rows = manifest.get("feature_scaler_n_samples_seen")
    target_rows = manifest.get("target_scaler_n_samples_seen")
    _require(
        isinstance(feature_rows, int)
        and feature_rows > 0
        and feature_rows == target_rows,
        "Scaler fit row count missing or inconsistent",
    )
    return ScalerProvenance(
        profile_path=profile_path,
        feature_scaler_path=feature_path,
        feature_scaler_sha256=_sha256(feature_path),
        target_scaler_path=target_path,
        target_scaler_sha256=_sha256(target_path),
        fit_split="training",
        fit_row_count=feature_rows,
        accessed_files=(feature_path, target_path, manifest_path),
    )


@lru_cache(maxsize=1)
def environment_versions() -> Mapping[str, str]:
    return MappingProxyType(
        {
            "python": platform.python_version(),
            "tensorflow": importlib.metadata.version("tensorflow"),
            "keras": importlib.metadata.version("keras"),
            "numpy": importlib.metadata.version("numpy"),
        }
    )


def _callback_policy_manifest() -> Mapping[str, Any]:
    return _deep_freeze(asdict(FORMAL_CALLBACK_POLICY))


def build_protocol_manifest(
    experiment_id: str,
    paths: FormalPathContract,
    git_identity: GitIdentity,
) -> tuple[Mapping[str, Any], tuple[Path, ...]]:
    """Build a complete immutable dry-run manifest without a formal write."""

    _require(experiment_id in LINEAR_EXPERIMENTS, "Unknown experiment")
    _require(paths.experiment_id == experiment_id, "Manifest/path experiment mismatch")
    spec = LINEAR_EXPERIMENTS[experiment_id]
    source_scalers = _scaler_provenance(spec.source_profile_path)
    target_scalers = _scaler_provenance(spec.target_profile_path)
    source_rows = {"training": 2088, "validation": 523, "test": 653}
    target_rows = {"training": 521, "validation": 131, "test": 2612}
    source_sequences = {"training": 2083, "validation": 518, "test": 648}
    target_sequences = {"training": 516, "validation": 126, "test": 2607}
    registry = candidate_registry()
    manifest = {
        "protocol_version": FORMAL_PROTOCOL_VERSION,
        "target_protocol_fingerprint": target_protocol_fingerprint(experiment_id),
        "experiment_id": experiment_id,
        "run_id": paths.run_id,
        "direction": spec.direction,
        "source_plant": spec.source_plant,
        "target_plant": spec.target_plant,
        "source_profile_path": str(spec.source_profile_path),
        "target_profile_path": str(spec.target_profile_path),
        "activation": spec.activation,
        "features": spec.features,
        "target": spec.target_column,
        "window": spec.window,
        "horizon": spec.horizon,
        "frequency": spec.frequency,
        "seed": spec.seed,
        "batch_size": spec.batch_size,
        "optimizer": spec.optimizer,
        "loss": spec.loss,
        "learning_rate_candidates": dict(LEARNING_RATE_CANDIDATES),
        "candidate_registry": {
            lifecycle: tuple(identifier for identifier, _ in entries)
            for lifecycle, entries in registry.items()
        },
        "callback_policy": _callback_policy_manifest(),
        "partial_ft_strategy_id": spec.partial_ft_strategy_id,
        "transferred_layers": spec.transferred_indices,
        "trainable_layers": spec.trainable_indices,
        "frozen_layers": spec.frozen_indices,
        "bn_frozen": spec.batch_normalization_frozen,
        "training_rows": {"source": source_rows["training"], "target": target_rows["training"]},
        "validation_rows": {"source": source_rows["validation"], "target": target_rows["validation"]},
        "test_rows": {"source": source_rows["test"], "target": target_rows["test"]},
        "training_sequences": {"source": source_sequences["training"], "target": target_sequences["training"]},
        "validation_sequences": {"source": source_sequences["validation"], "target": target_sequences["validation"]},
        "test_sequences": {"source": source_sequences["test"], "target": target_sequences["test"]},
        "feature_scaler_sha256": {
            "source": source_scalers.feature_scaler_sha256,
            "target": target_scalers.feature_scaler_sha256,
        },
        "feature_scaler_provenance": {
            "source": {
                "path": str(source_scalers.feature_scaler_path),
                "fit_split": source_scalers.fit_split,
                "fit_row_count": source_scalers.fit_row_count,
            },
            "target": {
                "path": str(target_scalers.feature_scaler_path),
                "fit_split": target_scalers.fit_split,
                "fit_row_count": target_scalers.fit_row_count,
            },
        },
        "target_scaler_sha256": {
            "source": source_scalers.target_scaler_sha256,
            "target": target_scalers.target_scaler_sha256,
        },
        "target_scaler_provenance": {
            "source": {
                "path": str(source_scalers.target_scaler_path),
                "fit_split": source_scalers.fit_split,
                "fit_row_count": source_scalers.fit_row_count,
            },
            "target": {
                "path": str(target_scalers.target_scaler_path),
                "fit_split": target_scalers.fit_split,
                "fit_row_count": target_scalers.fit_row_count,
            },
        },
        "git": asdict(git_identity),
        "environment": dict(environment_versions()),
        "device_policy": asdict(FORMAL_DEVICE_POLICY),
        "historical_target_test_previously_revealed": (
            HISTORICAL_TARGET_TEST_PREVIOUSLY_REVEALED[experiment_id]
        ),
        "test_accessed": False,
        "test_metrics_used_for_selection": False,
        "selection_locked": False,
        "config_locked": True,
        "checkpoint_locked": False,
        "test_authorized": False,
        "post_test_tuning_allowed": False,
        "formal_eligible": False,
        "dry_run": True,
        "protocol_stage": ProtocolStage.CONFIG_LOCKED_AWAITING_TRAINING.value,
        "formal_paths": {
            "run_root": str(paths.run_root),
            "protocol_manifest": str(paths.protocol_manifest),
            "selection": str(paths.selection),
            "final": str(paths.final),
        },
    }
    frozen = _deep_freeze(manifest)
    validate_protocol_manifest(frozen)
    return frozen, source_scalers.accessed_files + target_scalers.accessed_files


def validate_protocol_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_PROTOCOL_MANIFEST_FIELDS if field not in manifest]
    _require(not missing, f"Protocol manifest missing fields: {missing}")
    _require(manifest["protocol_version"] == FORMAL_PROTOCOL_VERSION, "Protocol version mismatch")
    _require(
        isinstance(manifest["run_id"], str)
        and bool(RUN_ID_PATTERN.fullmatch(manifest["run_id"])),
        "Protocol manifest run ID is invalid",
    )
    formal_paths = manifest.get("formal_paths")
    _require(isinstance(formal_paths, Mapping), "Protocol manifest paths are missing")
    run_root = formal_paths.get("run_root")
    _require(isinstance(run_root, str), "Protocol manifest run root is invalid")
    _require(
        Path(run_root).name == manifest["run_id"],
        "Protocol manifest run ID/path mismatch",
    )
    _require(
        manifest["target_protocol_fingerprint"]
        == target_protocol_fingerprint(manifest["experiment_id"]),
        "Target protocol fingerprint mismatch",
    )
    _require(manifest["activation"] == "linear", "Formal activation must be Linear")
    _require(manifest["config_locked"] is True, "Formal config must be locked")
    _require(manifest["test_accessed"] is False, "Dry-run cannot access Test")
    _require(manifest["test_metrics_used_for_selection"] is False, "Test metrics cannot select")
    _require(manifest["test_authorized"] is False, "Test authorization must default false")
    _require(manifest["post_test_tuning_allowed"] is False, "Locked protocol cannot be tuned")
    _require(isinstance(manifest["dry_run"], bool), "dry_run must be boolean")
    _require(isinstance(manifest["formal_eligible"], bool), "formal_eligible must be boolean")
    if manifest["dry_run"]:
        _require(manifest["formal_eligible"] is False, "Dry-run cannot be formal eligible")


def candidate_checkpoint_pattern(
    paths: FormalPathContract,
    lifecycle: str,
    identifier: str,
) -> Path:
    registry = dict(candidate_registry()[lifecycle])
    _require(identifier in registry, "Unknown candidate ID")
    bases = {
        "source": paths.source_candidates,
        "wotl": paths.target_wotl_candidates,
        "partial_ft": paths.target_partial_ft_candidates,
    }
    return bases[lifecycle] / identifier / "checkpoint_epoch_{epoch:04d}.hdf5"


def make_formal_callbacks(checkpoint_pattern: Path | str) -> tuple[Any, ...]:
    """Construct locked callback objects without touching their destination."""

    from keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TerminateOnNaN,
    )

    policy = FORMAL_CALLBACK_POLICY
    return (
        ModelCheckpoint(
            filepath=str(checkpoint_pattern),
            monitor=policy.checkpoint.monitor,
            save_best_only=policy.checkpoint.save_best_only,
            save_weights_only=policy.checkpoint.save_weights_only,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=policy.reduce_lr.monitor,
            factor=policy.reduce_lr.factor,
            patience=policy.reduce_lr.patience,
            min_lr=policy.reduce_lr.min_lr,
            cooldown=policy.reduce_lr.cooldown,
            verbose=1,
        ),
        EarlyStopping(
            monitor=policy.early_stopping.monitor,
            patience=policy.early_stopping.patience,
            min_delta=policy.early_stopping.min_delta,
            restore_best_weights=policy.early_stopping.restore_best_weights,
            verbose=1,
        ),
        TerminateOnNaN(),
    )


def prepare_formal_dry_run(
    experiment_id: str,
    run_id: str,
    *,
    git_identity: GitIdentity | None = None,
) -> FormalDryRun:
    """Prepare every formal contract in memory; write and epoch counts stay zero."""

    try:
        validate_formal_policy()
        paths = formal_path_contract(experiment_id, run_id)
        validate_new_run_destination(paths)
    except ValueError as exc:
        raise FormalProtocolError(str(exc)) from exc
    identity = git_identity or collect_git_identity()
    manifest, accessed = build_protocol_manifest(experiment_id, paths, identity)
    patterns: dict[str, Path] = {}
    callbacks: dict[str, tuple[Any, ...]] = {}
    for lifecycle, entries in candidate_registry().items():
        for identifier, _ in entries:
            pattern = candidate_checkpoint_pattern(paths, lifecycle, identifier)
            patterns[identifier] = pattern
            callbacks[identifier] = make_formal_callbacks(pattern)
    _require(not paths.run_root.exists(), "Dry-run created a formal run root")
    return FormalDryRun(
        paths=paths,
        manifest=manifest,
        candidate_checkpoint_patterns=MappingProxyType(patterns),
        callbacks=MappingProxyType(callbacks),
        accessed_files=accessed,
        formal_root_created=False,
    )


def validate_source_checkpoint_provenance(
    provenance: SourceCheckpointProvenance,
    *,
    expected_git_head: str,
) -> None:
    spec = LINEAR_EXPERIMENTS.get(provenance.experiment_id)
    _require(spec is not None, "Unknown Source provenance experiment")
    _require(provenance.protocol_version == FORMAL_PROTOCOL_VERSION, "Source protocol mismatch")
    _require(bool(RUN_ID_PATTERN.fullmatch(provenance.source_run_id)), "Invalid Source run ID")
    _require(provenance.source_activation == "linear", "Sigmoid Source is prohibited")
    _require(provenance.source_learning_rate in LEARNING_RATE_CANDIDATES["source"], "Unregistered Source LR")
    _require(
        provenance.source_candidate_id
        == candidate_id("source", provenance.source_learning_rate),
        "Source candidate ID/LR mismatch",
    )
    _require(1 <= provenance.source_best_epoch <= FORMAL_CALLBACK_POLICY.maximum_epochs, "Invalid Source best epoch")
    _require(
        math.isfinite(provenance.source_validation_loss)
        and provenance.source_validation_loss >= 0,
        "Invalid Source Validation loss",
    )
    _require(provenance.validation_selected, "Source checkpoint was not Validation-selected")
    _require(provenance.formal_eligible, "Source checkpoint is not formal eligible")
    _require(provenance.source_git_head == expected_git_head, "Source Git HEAD mismatch")
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", provenance.source_git_head)), "Invalid Source Git SHA")
    _require(bool(SHA256_PATTERN.fullmatch(provenance.source_checkpoint_sha256)), "Invalid Source checkpoint SHA")
    _require(provenance.source_checkpoint_sha256_verified, "Source checkpoint SHA is unverified")
    _require(
        provenance.source_layer_classes == FORMAL_LINEAR_LAYER_CLASSES,
        "Source layer topology mismatch",
    )
    _require(provenance.source_input_shape == (5, 5), "Source input shape mismatch")
    _require(provenance.source_output_shape == (1,), "Source output shape mismatch")
    _require(provenance.source_total_params == EXPECTED_TOTAL_PARAMS, "Source parameter count mismatch")
    candidate_root = (
        spec.output_root
        / provenance.source_run_id
        / "source_candidates"
        / provenance.source_candidate_id
    ).resolve()
    checkpoint = provenance.source_checkpoint_path.resolve()
    _require(
        checkpoint.name == f"checkpoint_epoch_{provenance.source_best_epoch:04d}.hdf5",
        "Source checkpoint filename/best epoch mismatch",
    )
    _require(checkpoint.is_relative_to(candidate_root), "Source checkpoint path is outside its formal candidate")
    _require(not checkpoint.is_relative_to(LINEAR_SMOKE_BASE.resolve()), "Smoke Source checkpoint rejected")
    _require(
        not any(checkpoint.is_relative_to(root.resolve()) for root in PROTECTED_LEGACY_ROOTS),
        "Historical Source checkpoint rejected",
    )
    _require(
        provenance.source_training_profile.resolve()
        == (spec.source_profile_path / "normalized_scale_training.csv").resolve(),
        "Source Training profile mismatch",
    )
    _require(
        provenance.source_validation_profile.resolve()
        == (spec.source_profile_path / "normalized_scale_validation.csv").resolve(),
        "Source Validation profile mismatch",
    )


def validate_validation_candidate(score: ValidationCandidateScore) -> None:
    _require(score.protocol_version == FORMAL_PROTOCOL_VERSION, "Candidate protocol mismatch")
    _require(score.experiment_id in LINEAR_EXPERIMENTS, "Unknown candidate experiment")
    _require(bool(RUN_ID_PATTERN.fullmatch(score.run_id)), "Invalid candidate run ID")
    _require(bool(GIT_HEAD_PATTERN.fullmatch(score.git_head)), "Invalid candidate Git HEAD")
    _require(
        score.target_protocol_fingerprint
        == target_protocol_fingerprint(score.experiment_id),
        "Candidate Target protocol mismatch",
    )
    _require(score.lifecycle in FORMAL_LIFECYCLES, "Unknown validation lifecycle")
    _require(score.candidate_id == candidate_id(score.lifecycle, score.learning_rate), "Candidate identity mismatch")
    _require(score.validation_only, "Candidate was not evaluated on Validation only")
    _require(not score.test_accessed, "Candidate accessed Test")
    _require(not score.test_metrics_used, "Candidate selection used Test metrics")
    _require(score.checkpoint_sha256_verified, "Candidate checkpoint SHA is unverified")
    _require(bool(SHA256_PATTERN.fullmatch(score.checkpoint_sha256)), "Invalid candidate checkpoint SHA")
    _require(
        1 <= score.best_epoch <= FORMAL_CALLBACK_POLICY.maximum_epochs,
        "Candidate best epoch is invalid",
    )
    values = (
        score.validation_loss,
        score.validation_original_mae,
        score.validation_original_rmse,
        score.validation_original_r2,
    )
    _require(all(math.isfinite(value) for value in values), "Non-finite Validation diagnostic")
    _require(score.validation_loss >= 0, "Validation loss must be non-negative")
    _require(score.validation_original_mae >= 0, "Validation MAE must be non-negative")
    _require(score.validation_original_rmse >= 0, "Validation RMSE must be non-negative")
    expected_params = EXPECTED_TRAINABLE_PARAMS if score.lifecycle == "partial_ft" else EXPECTED_TOTAL_PARAMS
    _require(score.trainable_params == expected_params, "Candidate trainable parameter count mismatch")
    directory_name = {
        "source": "source_candidates",
        "wotl": "target_wotl_candidates",
        "partial_ft": "target_partial_ft_candidates",
    }[score.lifecycle]
    candidate_root = (
        LINEAR_EXPERIMENTS[score.experiment_id].output_root
        / score.run_id
        / directory_name
        / score.candidate_id
    ).resolve()
    checkpoint = score.checkpoint_path.resolve()
    _require(checkpoint.is_relative_to(candidate_root), "Candidate checkpoint is outside its formal namespace")
    _require(
        bool(re.fullmatch(r"checkpoint_epoch_\d{4}\.hdf5", checkpoint.name)),
        "Candidate checkpoint filename is invalid",
    )
    _require(
        checkpoint.name == f"checkpoint_epoch_{score.best_epoch:04d}.hdf5",
        "Candidate checkpoint filename/best epoch mismatch",
    )
    _require(not checkpoint.is_relative_to(LINEAR_SMOKE_BASE.resolve()), "Smoke candidate checkpoint rejected")


def _lowest_close(
    scores: Sequence[ValidationCandidateScore],
    attribute: str,
    relative_tolerance: float,
) -> list[ValidationCandidateScore]:
    minimum = min(getattr(score, attribute) for score in scores)
    tolerance = max(VALIDATION_TIE_ATOL, abs(minimum) * relative_tolerance)
    return [score for score in scores if getattr(score, attribute) <= minimum + tolerance]


def _select_from_validated(scores: Sequence[ValidationCandidateScore]) -> ValidationCandidateScore:
    contenders = _lowest_close(scores, "validation_loss", VALIDATION_LOSS_TIE_RTOL)
    contenders = _lowest_close(contenders, "validation_original_rmse", VALIDATION_DIAGNOSTIC_TIE_RTOL)
    contenders = _lowest_close(contenders, "validation_original_mae", VALIDATION_DIAGNOSTIC_TIE_RTOL)
    highest_r2 = max(score.validation_original_r2 for score in contenders)
    r2_tolerance = max(VALIDATION_TIE_ATOL, abs(highest_r2) * VALIDATION_DIAGNOSTIC_TIE_RTOL)
    contenders = [score for score in contenders if score.validation_original_r2 >= highest_r2 - r2_tolerance]
    minimum_params = min(score.trainable_params for score in contenders)
    contenders = [score for score in contenders if score.trainable_params == minimum_params]
    minimum_lr = min(score.learning_rate for score in contenders)
    contenders = [score for score in contenders if score.learning_rate == minimum_lr]
    return sorted(contenders, key=lambda score: score.candidate_id)[0]


def select_validation_candidate(
    lifecycle: str,
    scores: Sequence[ValidationCandidateScore],
) -> ValidationCandidateScore:
    _require(lifecycle in FORMAL_LIFECYCLES, "Unknown selection lifecycle")
    for score in scores:
        validate_validation_candidate(score)
        _require(score.lifecycle == lifecycle, "Mixed lifecycle selection")
    expected_ids = {identifier for identifier, _ in candidate_registry()[lifecycle]}
    actual_ids = {score.candidate_id for score in scores}
    _require(len(scores) == len(actual_ids), "Duplicate Validation candidate")
    _require(actual_ids == expected_ids, "Every registered candidate must remain in selection")
    return _select_from_validated(scores)


def build_source_checkpoint_provenance_from_selection(
    *,
    experiment_id: str,
    run_id: str,
    source_scores: Sequence[ValidationCandidateScore],
    expected_git_head: str,
    architecture_evidence: SourceArchitectureEvidence,
) -> SourceCheckpointProvenance:
    """Require the complete Source registry, select it, then create provenance."""

    _require(experiment_id in LINEAR_EXPERIMENTS, "Unknown Source experiment")
    _require(bool(RUN_ID_PATTERN.fullmatch(run_id)), "Invalid Source run ID")
    _require(architecture_evidence.git_head == expected_git_head, "Source evidence Git HEAD mismatch")
    for score in source_scores:
        _require(score.experiment_id == experiment_id, "Source candidate experiment mismatch")
        _require(score.run_id == run_id, "Source candidate run mismatch")
        _require(score.git_head == expected_git_head, "Source candidate Git HEAD mismatch")
    selected = select_validation_candidate("source", source_scores)
    spec = LINEAR_EXPERIMENTS[experiment_id]
    provenance = SourceCheckpointProvenance(
        protocol_version=FORMAL_PROTOCOL_VERSION,
        experiment_id=experiment_id,
        source_run_id=run_id,
        source_candidate_id=selected.candidate_id,
        source_activation=architecture_evidence.activation,
        source_learning_rate=selected.learning_rate,
        source_best_epoch=selected.best_epoch,
        source_validation_loss=selected.validation_loss,
        source_checkpoint_path=selected.checkpoint_path,
        source_checkpoint_sha256=selected.checkpoint_sha256,
        source_checkpoint_sha256_verified=selected.checkpoint_sha256_verified,
        source_git_head=architecture_evidence.git_head,
        source_training_profile=(
            spec.source_profile_path / "normalized_scale_training.csv"
        ),
        source_validation_profile=(
            spec.source_profile_path / "normalized_scale_validation.csv"
        ),
        source_layer_classes=architecture_evidence.layer_classes,
        source_input_shape=architecture_evidence.input_shape,
        source_output_shape=architecture_evidence.output_shape,
        source_total_params=architecture_evidence.total_params,
        validation_selected=True,
        formal_eligible=architecture_evidence.formal_eligible,
    )
    validate_source_checkpoint_provenance(
        provenance,
        expected_git_head=expected_git_head,
    )
    return provenance


def build_selection_record(
    *,
    experiment_id: str,
    selection_timestamp: str,
    source_scores: Sequence[ValidationCandidateScore],
    source_architecture_evidence: SourceArchitectureEvidence,
    expected_git_head: str,
    wotl_scores: Sequence[ValidationCandidateScore],
    partial_ft_scores: Sequence[ValidationCandidateScore],
) -> SelectionRecord:
    _require(experiment_id in LINEAR_EXPERIMENTS, "Unknown selection experiment")
    _require(
        bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", selection_timestamp)),
        "Selection timestamp must be UTC second precision",
    )
    _require(bool(source_scores), "Source candidate registry is empty")
    locked_run_id = source_scores[0].run_id
    source_provenance = build_source_checkpoint_provenance_from_selection(
        experiment_id=experiment_id,
        run_id=locked_run_id,
        source_scores=source_scores,
        expected_git_head=expected_git_head,
        architecture_evidence=source_architecture_evidence,
    )
    for score in tuple(wotl_scores) + tuple(partial_ft_scores):
        _require(score.experiment_id == experiment_id, "Candidate/selection experiment mismatch")
        _require(score.run_id == source_provenance.source_run_id, "Candidate/Source run mismatch")
        _require(score.git_head == expected_git_head, "Candidate/Source Git HEAD mismatch")
    wotl_selected = select_validation_candidate("wotl", wotl_scores)
    partial_selected = select_validation_candidate("partial_ft", partial_ft_scores)
    record = SelectionRecord(
        protocol_version=FORMAL_PROTOCOL_VERSION,
        experiment_id=experiment_id,
        run_id=locked_run_id,
        git_head=expected_git_head,
        target_protocol_fingerprint=target_protocol_fingerprint(experiment_id),
        selection_timestamp=selection_timestamp,
        source_selected_candidate=source_provenance.source_candidate_id,
        source_checkpoint_sha256=source_provenance.source_checkpoint_sha256,
        source_checkpoint_sha256_verified=(
            source_provenance.source_checkpoint_sha256_verified
        ),
        wotl_selected_candidate=wotl_selected.candidate_id,
        wotl_selected_checkpoint_sha256=wotl_selected.checkpoint_sha256,
        wotl_selected_checkpoint_sha256_verified=(
            wotl_selected.checkpoint_sha256_verified
        ),
        partial_ft_selected_candidate=partial_selected.candidate_id,
        partial_ft_selected_checkpoint_sha256=partial_selected.checkpoint_sha256,
        partial_ft_selected_checkpoint_sha256_verified=(
            partial_selected.checkpoint_sha256_verified
        ),
        comparison_pair_locked=True,
        selection_basis=(
            "lowest_validation_loss",
            "lowest_validation_original_rmse_within_tolerance",
            "lowest_validation_original_mae_within_tolerance",
            "highest_validation_original_r2_within_tolerance",
            "fewer_trainable_parameters",
            "lower_learning_rate",
            "lexical_candidate_id",
        ),
        source_validation_comparison=tuple(source_scores),
        wotl_validation_comparison=tuple(wotl_scores),
        partial_ft_validation_comparison=tuple(partial_ft_scores),
        config_locked=True,
        selection_locked=True,
        checkpoint_locked=True,
        test_metrics_used_for_selection=False,
        test_accessed=False,
        test_authorized=False,
        post_test_tuning_allowed=False,
        protocol_stage=ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION,
    )
    validate_selection_record(record)
    return record


def selection_record_mapping(record: SelectionRecord) -> Mapping[str, Any]:
    payload = asdict(record)
    payload["protocol_stage"] = record.protocol_stage.value
    for field_name in (
        "source_validation_comparison",
        "wotl_validation_comparison",
        "partial_ft_validation_comparison",
    ):
        payload[field_name] = tuple(
            asdict(score) for score in getattr(record, field_name)
        )
    return _deep_freeze(payload)


def _reselect_record_lifecycle(
    values: Mapping[str, Any],
    *,
    lifecycle: str,
    comparison_field: str,
    selected_candidate_field: str,
    selected_sha_field: str,
) -> ValidationCandidateScore:
    raw_comparison = values[comparison_field]
    _require(
        isinstance(raw_comparison, Sequence)
        and not isinstance(raw_comparison, (str, bytes)),
        f"{comparison_field} must be a sequence",
    )
    comparison = tuple(
        _coerce_validation_candidate_score(item) for item in raw_comparison
    )
    for score in comparison:
        _require(
            score.experiment_id == values["experiment_id"],
            f"{lifecycle} comparison experiment mismatch",
        )
        _require(
            score.run_id == values["run_id"],
            f"{lifecycle} comparison run mismatch",
        )
        _require(
            score.protocol_version == values["protocol_version"],
            f"{lifecycle} comparison protocol mismatch",
        )
        _require(
            score.git_head == values["git_head"],
            f"{lifecycle} comparison Git HEAD mismatch",
        )
        _require(
            score.target_protocol_fingerprint
            == values["target_protocol_fingerprint"],
            f"{lifecycle} comparison Target protocol mismatch",
        )
    selected = select_validation_candidate(lifecycle, comparison)
    _require(
        selected.candidate_id == values[selected_candidate_field],
        f"{lifecycle} selected candidate does not match deterministic selection",
    )
    _require(
        selected.checkpoint_sha256 == values[selected_sha_field],
        f"{lifecycle} selected checkpoint SHA does not match deterministic selection",
    )
    _require(
        selected.checkpoint_sha256_verified is True,
        f"{lifecycle} deterministically selected checkpoint SHA is unverified",
    )
    return selected


def validate_selection_record(record: SelectionRecord | Mapping[str, Any]) -> None:
    values: Mapping[str, Any]
    if isinstance(record, SelectionRecord):
        values = selection_record_mapping(record)
    else:
        values = record
    missing = [field for field in REQUIRED_SELECTION_FIELDS if field not in values]
    _require(not missing, f"Selection record missing fields: {missing}")
    _require(values["protocol_version"] == FORMAL_PROTOCOL_VERSION, "Selection protocol mismatch")
    _require(
        isinstance(values["run_id"], str)
        and bool(RUN_ID_PATTERN.fullmatch(values["run_id"])),
        "Selection run ID is invalid",
    )
    _require(bool(GIT_HEAD_PATTERN.fullmatch(values["git_head"])), "Selection Git HEAD invalid")
    _require(
        values["target_protocol_fingerprint"]
        == target_protocol_fingerprint(values["experiment_id"]),
        "Selection Target protocol mismatch",
    )
    _require(values["config_locked"] is True, "Selection config is not locked")
    _require(values["selection_locked"] is True, "Selection is not locked")
    _require(values["checkpoint_locked"] is True, "Checkpoint is not locked")
    _require(values["comparison_pair_locked"] is True, "Comparison pair is not locked")
    _require(
        values["source_checkpoint_sha256_verified"] is True,
        "Source checkpoint SHA is unverified",
    )
    _require(
        values["wotl_selected_checkpoint_sha256_verified"] is True,
        "WOTL checkpoint SHA is unverified",
    )
    _require(
        values["partial_ft_selected_checkpoint_sha256_verified"] is True,
        "Partial FT checkpoint SHA is unverified",
    )
    for field_name in (
        "source_checkpoint_sha256",
        "wotl_selected_checkpoint_sha256",
        "partial_ft_selected_checkpoint_sha256",
    ):
        _require(
            bool(SHA256_PATTERN.fullmatch(values[field_name])),
            f"Invalid locked checkpoint SHA: {field_name}",
        )
    _reselect_record_lifecycle(
        values,
        lifecycle="source",
        comparison_field="source_validation_comparison",
        selected_candidate_field="source_selected_candidate",
        selected_sha_field="source_checkpoint_sha256",
    )
    _reselect_record_lifecycle(
        values,
        lifecycle="wotl",
        comparison_field="wotl_validation_comparison",
        selected_candidate_field="wotl_selected_candidate",
        selected_sha_field="wotl_selected_checkpoint_sha256",
    )
    _reselect_record_lifecycle(
        values,
        lifecycle="partial_ft",
        comparison_field="partial_ft_validation_comparison",
        selected_candidate_field="partial_ft_selected_candidate",
        selected_sha_field="partial_ft_selected_checkpoint_sha256",
    )
    _require("selected_method" not in values, "Cross-method winner field is prohibited")
    _require(
        "selected_checkpoint_sha256" not in values,
        "Single selected checkpoint field is prohibited",
    )
    _require(values["test_metrics_used_for_selection"] is False, "Selection used Test metrics")
    _require(values["test_accessed"] is False, "Selection accessed Test")
    _require(values["test_authorized"] is False, "Selection must await human Test authorization")
    _require(values["post_test_tuning_allowed"] is False, "Locked selection cannot be tuned")
    stage = values["protocol_stage"]
    if isinstance(stage, ProtocolStage):
        stage = stage.value
    _require(
        stage == ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        "Selection state machine stage mismatch",
    )


def authorize_final_test(
    manifest: Mapping[str, Any],
    selection: SelectionRecord,
    *,
    human_authorized: bool,
) -> FinalTestAuthorization:
    """Issue an in-memory authorization proof; no dataset is opened."""

    validate_selection_record(selection)
    _require(human_authorized is True, "Explicit human Test authorization is required")
    _require(manifest.get("formal_eligible") is True, "Run is not formal eligible")
    _require(
        manifest.get("dry_run") is False,
        "Dry-run manifest cannot authorize Final Test",
    )
    manifest_stage = manifest.get("protocol_stage")
    if isinstance(manifest_stage, ProtocolStage):
        manifest_stage = manifest_stage.value
    _require(
        manifest_stage
        == ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        "Manifest protocol stage cannot authorize Final Test",
    )
    _require(
        manifest.get("protocol_version") == selection.protocol_version,
        "Authorization protocol mismatch",
    )
    _require(
        manifest.get("target_protocol_fingerprint")
        == selection.target_protocol_fingerprint,
        "Authorization Target protocol mismatch",
    )
    _require(
        manifest.get("run_id") == selection.run_id,
        "Authorization run ID mismatch",
    )
    manifest_git = manifest.get("git")
    _require(isinstance(manifest_git, Mapping), "Authorization Git identity missing")
    _require(
        manifest_git.get("head") == selection.git_head,
        "Authorization Git HEAD mismatch",
    )
    _require(manifest.get("config_locked") is True, "Manifest config is not locked")
    _require(manifest.get("selection_locked") is True, "Manifest selection is not locked")
    _require(manifest.get("checkpoint_locked") is True, "Manifest checkpoint is not locked")
    _require(manifest.get("test_authorized") is False, "Manifest Test state is already authorized")
    _require(manifest.get("post_test_tuning_allowed") is False, "Manifest permits post-selection tuning")
    _require(selection.selection_locked, "Selection is not locked")
    _require(selection.checkpoint_locked, "Checkpoint is not locked")
    _require(manifest.get("test_accessed") is False, "Manifest reports prior Test access")
    _require(manifest.get("test_metrics_used_for_selection") is False, "Manifest selection used Test")
    _require(not selection.test_accessed, "Selection reports prior Test access")
    _require(not selection.test_metrics_used_for_selection, "Selection used Test metrics")
    _require(selection.comparison_pair_locked, "Comparison pair is not locked")
    _require(
        selection.wotl_selected_checkpoint_sha256_verified,
        "WOTL checkpoint SHA unverified",
    )
    _require(
        selection.partial_ft_selected_checkpoint_sha256_verified,
        "Partial FT checkpoint SHA unverified",
    )
    _require(
        bool(SHA256_PATTERN.fullmatch(selection.wotl_selected_checkpoint_sha256)),
        "WOTL checkpoint SHA missing",
    )
    _require(
        bool(SHA256_PATTERN.fullmatch(selection.partial_ft_selected_checkpoint_sha256)),
        "Partial FT checkpoint SHA missing",
    )
    _require(manifest.get("experiment_id") == selection.experiment_id, "Authorization experiment mismatch")
    _require(
        manifest.get("historical_target_test_previously_revealed")
        is HISTORICAL_TARGET_TEST_PREVIOUSLY_REVEALED[selection.experiment_id],
        "Historical Test disclosure mismatch",
    )
    authorized_selection = replace(
        selection,
        test_authorized=True,
        protocol_stage=ProtocolStage.TEST_AUTHORIZED,
    )
    return FinalTestAuthorization(
        experiment_id=authorized_selection.experiment_id,
        run_id=authorized_selection.run_id,
        wotl_selected_candidate=authorized_selection.wotl_selected_candidate,
        wotl_checkpoint_sha256=(
            authorized_selection.wotl_selected_checkpoint_sha256
        ),
        partial_ft_selected_candidate=(
            authorized_selection.partial_ft_selected_candidate
        ),
        partial_ft_checkpoint_sha256=(
            authorized_selection.partial_ft_selected_checkpoint_sha256
        ),
        comparison_pair_locked=True,
        protocol_stage=authorized_selection.protocol_stage,
        test_authorized=True,
        _proof=_AUTHORIZATION_PROOF,
    )


def validate_authorization_proof(authorization: FinalTestAuthorization) -> None:
    _require(
        isinstance(authorization, FinalTestAuthorization)
        and authorization._proof is _AUTHORIZATION_PROOF,
        "Invalid Final Test authorization proof",
    )
    _require(authorization.test_authorized, "Final Test is not authorized")
    _require(
        isinstance(authorization.run_id, str)
        and bool(RUN_ID_PATTERN.fullmatch(authorization.run_id)),
        "Authorization run ID is invalid",
    )
    _require(authorization.comparison_pair_locked, "Authorization pair is not locked")
    _require(
        bool(SHA256_PATTERN.fullmatch(authorization.wotl_checkpoint_sha256)),
        "Authorization WOTL SHA is invalid",
    )
    _require(
        bool(SHA256_PATTERN.fullmatch(authorization.partial_ft_checkpoint_sha256)),
        "Authorization Partial FT SHA is invalid",
    )
    _require(authorization.protocol_stage is ProtocolStage.TEST_AUTHORIZED, "Authorization stage mismatch")
