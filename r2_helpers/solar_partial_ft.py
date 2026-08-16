"""Zero-epoch contracts for Solar R2.5 partial fine-tuning.

The helpers in this module build and validate models, but never call ``fit``.
Target Test access is gated by an immutable validation-only selection record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from r2_config.solar_r25 import (
    BASELINE_MANIFEST_PATH,
    BASELINE_RUN_ROOT,
    BASELINE_SOURCE_BEST_EPOCH,
    BASELINE_SOURCE_CHECKPOINT_RELATIVE,
    BASELINE_SOURCE_CHECKPOINT_SHA256,
    BATCH_NORMALIZATION_INDICES,
    EXPECTED_LAYER_CLASSES,
    EXPECTED_PARAMETER_COUNTS,
    PARTIAL_STRATEGY_ORDER,
    PARTIAL_STRATEGY_SPECS,
    R25_BATCH_SIZE,
    R25_LEARNING_RATE,
    R25_LOSS,
    R25_MAX_EPOCHS,
    R25_OUTPUT_BASE,
    R25_SEED,
    TARGET_ADAPTER_INDICES,
    TRANSFERRED_LAYER_INDICES,
    WEIGHT_BEARING_LAYER_INDICES,
)
from r2_helpers.solar_runtime import (
    NON_TRANSFERRED_WEIGHT_LAYER_INDICES,
    R2_CUSTOM_OBJECTS,
    R2_INPUT_SHAPE,
    TRANSFERABLE_LAYER_INDICES,
    NamedSequenceSplit,
    ReloadedCheckpointModel,
    build_r2_model,
    configure_reproducibility,
    evaluate_test,
    reload_best_checkpoint,
    rmse,
)


class PartialFTContractError(AssertionError):
    """Raised when the R2.5 partial fine-tuning protocol is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PartialFTContractError(message)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BaselineSourceCheckpoint:
    run_root: Path
    manifest_path: Path
    checkpoint_path: Path
    sha256: str
    best_epoch: int


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    trainable_indices: tuple[int, ...]
    frozen_indices: tuple[int, ...]
    purpose: str


STRATEGY_REGISTRY: Mapping[str, StrategyDefinition] = MappingProxyType(
    {
        strategy_id: StrategyDefinition(
            strategy_id=strategy_id,
            trainable_indices=tuple(spec["trainable_indices"]),
            frozen_indices=tuple(spec["frozen_indices"]),
            purpose=str(spec["purpose"]),
        )
        for strategy_id, spec in PARTIAL_STRATEGY_SPECS.items()
    }
)


@dataclass(frozen=True)
class ParameterCounts:
    total_params: int
    trainable_params: int
    non_trainable_params: int
    trainable_percentage: float


@dataclass(frozen=True)
class PartialCandidateModel:
    strategy: StrategyDefinition
    model: Any
    parameter_counts: ParameterCounts
    layer_manifest: tuple[Mapping[str, Any], ...]
    optimizer_identity: int
    seed: int


@dataclass(frozen=True)
class ValidationScore:
    strategy_id: str
    split_name: str
    scale: str
    rmse: float
    mae: float
    r2: float
    best_epoch: int
    checkpoint_path: Path


_SELECTION_PROOF = object()


@dataclass(frozen=True)
class SelectionDecision:
    winner: ValidationScore
    candidates: tuple[ValidationScore, ...]
    _proof: object


@dataclass(frozen=True)
class SelectedTestAuthorization:
    strategy_id: str
    checkpoint_model: ReloadedCheckpointModel
    selection_path: Path


def validate_baseline_source_checkpoint(
    run_root: Path | str = BASELINE_RUN_ROOT,
) -> BaselineSourceCheckpoint:
    """Verify the immutable R2 PASS run and its selected Source checkpoint."""

    root = Path(run_root).resolve()
    manifest_path = root / BASELINE_MANIFEST_PATH.name
    _require(manifest_path.is_file(), f"Baseline manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartialFTContractError(f"Cannot read baseline manifest: {exc}") from exc

    _require(manifest.get("status") == "PASS", "Baseline run status must be PASS")
    _require(
        manifest.get("identity", {}).get("run_id") == root.name,
        "Baseline run ID mismatch",
    )
    record = manifest.get("checkpoints", {}).get("source_pretrain")
    _require(isinstance(record, dict), "Source checkpoint manifest record is missing")
    _require(
        record.get("path") == BASELINE_SOURCE_CHECKPOINT_RELATIVE.as_posix(),
        "Unexpected baseline Source checkpoint path",
    )
    _require(
        record.get("sha256") == BASELINE_SOURCE_CHECKPOINT_SHA256,
        "Baseline manifest Source SHA-256 differs from the approved value",
    )
    _require(
        record.get("best_epoch") == BASELINE_SOURCE_BEST_EPOCH,
        "Baseline Source best epoch mismatch",
    )

    checkpoint_path = (root / Path(record["path"])).resolve()
    _require(checkpoint_path.is_relative_to(root), "Source checkpoint escapes baseline root")
    _require(checkpoint_path.is_file(), f"Source checkpoint not found: {checkpoint_path}")
    actual_hash = sha256_file(checkpoint_path)
    _require(actual_hash == record["sha256"], "Source checkpoint SHA-256 mismatch")
    return BaselineSourceCheckpoint(
        run_root=root,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        sha256=actual_hash,
        best_epoch=int(record["best_epoch"]),
    )


def validate_exact_topology(model: Any) -> None:
    classes = tuple(type(layer).__name__ for layer in model.layers)
    _require(classes == EXPECTED_LAYER_CLASSES, f"Unexpected layer topology: {classes}")
    _require(tuple(model.input_shape[1:]) == R2_INPUT_SHAPE, "Input shape changed")
    _require(tuple(model.output_shape[1:]) == (1,), "Output shape changed")
    _require(
        model.layers[6].activation.__name__ == "sigmoid",
        "Output activation must remain sigmoid",
    )
    _require(
        tuple(TRANSFERABLE_LAYER_INDICES) == TRANSFERRED_LAYER_INDICES,
        "Legacy transfer scope changed",
    )
    _require(
        tuple(NON_TRANSFERRED_WEIGHT_LAYER_INDICES) == TARGET_ADAPTER_INDICES,
        "Target adapter scope changed",
    )


def parameter_counts(model: Any) -> ParameterCounts:
    def count(weights: Sequence[Any]) -> int:
        return int(sum(np.prod(tuple(weight.shape)) for weight in weights))

    trainable = count(model.trainable_weights)
    non_trainable = count(model.non_trainable_weights)
    total = trainable + non_trainable
    return ParameterCounts(
        total_params=total,
        trainable_params=trainable,
        non_trainable_params=non_trainable,
        trainable_percentage=(100.0 * trainable / total),
    )


def layer_contract_manifest(model: Any) -> tuple[Mapping[str, Any], ...]:
    """Record runtime names while retaining index/class as the control keys."""

    validate_exact_topology(model)
    rows = []
    for index, layer in enumerate(model.layers):
        layer_total = int(sum(np.prod(tuple(weight.shape)) for weight in layer.weights))
        rows.append(
            MappingProxyType(
                {
                    "index": index,
                    "runtime_name": layer.name,
                    "class_name": type(layer).__name__,
                    "trainable": bool(layer.trainable),
                    "total_params": layer_total,
                }
            )
        )
    return tuple(rows)


def apply_partial_strategy(model: Any, strategy_id: str) -> StrategyDefinition:
    validate_exact_topology(model)
    try:
        strategy = STRATEGY_REGISTRY[strategy_id]
    except KeyError as exc:
        raise PartialFTContractError(f"Unknown Partial FT strategy: {strategy_id}") from exc

    configured = set(strategy.trainable_indices) | set(strategy.frozen_indices)
    _require(
        configured == set(WEIGHT_BEARING_LAYER_INDICES),
        f"Strategy {strategy_id} does not classify every weight-bearing layer",
    )
    for index, expected_class in enumerate(EXPECTED_LAYER_CLASSES):
        _require(
            type(model.layers[index]).__name__ == expected_class,
            f"Layer {index} must be {expected_class}",
        )
        model.layers[index].trainable = index in strategy.trainable_indices

    _require(
        all(not model.layers[index].trainable for index in BATCH_NORMALIZATION_INDICES),
        "Both BatchNormalization layers must remain frozen",
    )
    return strategy


def compile_partial_model(model: Any) -> int:
    """Compile after applying flags with a new, candidate-local optimizer."""

    optimizer = Adam(learning_rate=R25_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=R25_LOSS,
        metrics=["mse", rmse, "mae", "mape", "msle"],
    )
    return id(optimizer)


def validate_partial_model(candidate: PartialCandidateModel) -> None:
    model = candidate.model
    strategy = candidate.strategy
    validate_exact_topology(model)
    actual_trainable = tuple(
        index
        for index in WEIGHT_BEARING_LAYER_INDICES
        if model.layers[index].trainable
    )
    actual_frozen = tuple(
        index
        for index in WEIGHT_BEARING_LAYER_INDICES
        if not model.layers[index].trainable
    )
    _require(actual_trainable == strategy.trainable_indices, "Trainable index mismatch")
    _require(actual_frozen == strategy.frozen_indices, "Frozen index mismatch")
    _require(type(model.optimizer).__name__ == "Adam", "Optimizer must remain Adam")
    _require(
        np.isclose(float(model.optimizer.learning_rate.numpy()), R25_LEARNING_RATE),
        "Partial FT learning rate changed",
    )
    loss_name = model.loss if isinstance(model.loss, str) else model.loss.__name__
    _require(loss_name in ("mse", "mean_squared_error"), "Loss must remain MSE")
    counts = parameter_counts(model)
    _require(counts == candidate.parameter_counts, "Parameter report is stale")
    _require(
        counts.total_params == EXPECTED_PARAMETER_COUNTS["total"],
        "Runtime total parameter count changed",
    )
    _require(
        counts.trainable_params == EXPECTED_PARAMETER_COUNTS[strategy.strategy_id],
        "Runtime trainable parameter count differs from the approved architecture",
    )


def build_partial_candidate(
    source_model: Any,
    *,
    strategy_id: str,
    output_dir: Path | str,
    seed: int = R25_SEED,
) -> PartialCandidateModel:
    """Build, transfer, mask, and freshly compile a candidate without fitting."""

    _require(seed == R25_SEED, f"R2.5 seed must remain {R25_SEED}")
    validate_exact_topology(source_model)
    configure_reproducibility(seed)
    model = build_r2_model(
        output_dir=output_dir,
        pretrained_model=source_model,
        freeze=False,
    )
    strategy = apply_partial_strategy(model, strategy_id)
    optimizer_identity = compile_partial_model(model)
    candidate = PartialCandidateModel(
        strategy=strategy,
        model=model,
        parameter_counts=parameter_counts(model),
        layer_manifest=layer_contract_manifest(model),
        optimizer_identity=optimizer_identity,
        seed=seed,
    )
    validate_partial_model(candidate)
    return candidate


def validate_partial_transfer(
    source_model: Any,
    target_model: Any,
    *,
    target_initial_model: Any | None = None,
) -> None:
    """Prove the unchanged 2--5 copy scope before any training occurs."""

    validate_exact_topology(source_model)
    validate_exact_topology(target_model)
    for index in TRANSFERRED_LAYER_INDICES:
        for source_weight, target_weight in zip(
            source_model.layers[index].get_weights(),
            target_model.layers[index].get_weights(),
        ):
            _require(
                np.array_equal(source_weight, target_weight),
                f"Transferred weights differ at layer {index}",
            )
    if target_initial_model is not None:
        validate_exact_topology(target_initial_model)
        for index in TARGET_ADAPTER_INDICES:
            for initial_weight, target_weight in zip(
                target_initial_model.layers[index].get_weights(),
                target_model.layers[index].get_weights(),
            ):
                _require(
                    np.array_equal(initial_weight, target_weight),
                    f"Target adapter {index} was incorrectly transferred",
                )


def batch_normalization_state(model: Any) -> Mapping[int, tuple[np.ndarray, ...]]:
    validate_exact_topology(model)
    return MappingProxyType(
        {
            index: tuple(weight.copy() for weight in model.layers[index].get_weights())
            for index in BATCH_NORMALIZATION_INDICES
        }
    )


def candidate_checkpoint_pattern(run_root: Path | str, strategy_id: str) -> Path:
    _require(strategy_id in STRATEGY_REGISTRY, "Unknown checkpoint strategy")
    return (
        Path(run_root)
        / "candidates"
        / strategy_id
        / "checkpoint_epoch_{epoch:04d}.hdf5"
    )


def make_partial_callbacks(
    run_root: Path | str,
    strategy_id: str,
) -> tuple[Any, ...]:
    """Create the unchanged R2 callback protocol in a candidate namespace."""

    pattern = candidate_checkpoint_pattern(run_root, strategy_id)
    return (
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(pattern),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    )


def validate_r25_output_root(run_root: Path | str) -> Path:
    root = Path(run_root).resolve()
    output_base = R25_OUTPUT_BASE.resolve()
    _require(root != output_base, "R2.5 run root must include a unique run ID")
    _require(root.is_relative_to(output_base), "R2.5 output escapes its dedicated root")
    _require(not root.exists(), f"R2.5 run root already exists: {root}")
    _require(not root.is_relative_to(BASELINE_RUN_ROOT.resolve()), "Cannot overwrite R2")
    return root


def _validate_validation_score(score: ValidationScore) -> None:
    _require(score.strategy_id in STRATEGY_REGISTRY, "Unknown validation strategy")
    _require(score.split_name == "validation", "Strategy selection accepts Validation only")
    _require(score.scale == "original", "Strategy selection requires original scale")
    _require(score.best_epoch > 0, "Best epoch must be positive")
    _require(
        all(np.isfinite(value) for value in (score.rmse, score.mae, score.r2)),
        "Validation metrics must be finite",
    )
    _require(score.rmse >= 0 and score.mae >= 0, "Validation errors must be non-negative")


def select_validation_winner(scores: Sequence[ValidationScore]) -> SelectionDecision:
    """Validate the sole approved Candidate B validation result."""

    for score in scores:
        _validate_validation_score(score)
    by_id = {score.strategy_id: score for score in scores}
    _require(len(by_id) == len(scores), "Duplicate strategy validation score")
    _require(
        tuple(sorted(by_id, key=PARTIAL_STRATEGY_ORDER.index))
        == PARTIAL_STRATEGY_ORDER,
        "Every approved candidate must have one Validation score",
    )
    ordered = tuple(by_id[strategy_id] for strategy_id in PARTIAL_STRATEGY_ORDER)
    winner = ordered[0]
    return SelectionDecision(winner=winner, candidates=ordered, _proof=_SELECTION_PROOF)


def lock_selection(
    selection_path: Path | str,
    decision: SelectionDecision,
    *,
    source_checkpoint_sha256: str,
) -> Path:
    """Create selection.json exactly once; no overwrite option exists."""

    _require(
        isinstance(decision, SelectionDecision) and decision._proof is _SELECTION_PROOF,
        "selection.json requires select_validation_winner() output",
    )
    winner = decision.winner
    for score in decision.candidates:
        _validate_validation_score(score)
    _require(
        source_checkpoint_sha256 == BASELINE_SOURCE_CHECKPOINT_SHA256,
        "Selection must reference the approved Source checkpoint",
    )
    path = Path(selection_path)
    _require(path.name == "selection.json", "Selection lock must be selection.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "locked": True,
        "selection_basis": "original_scale_validation",
        "selection_order": ["single_approved_strategy", "best_validation_checkpoint"],
        "selected_strategy_id": winner.strategy_id,
        "selected_checkpoint_path": str(winner.checkpoint_path.resolve()),
        "validation_metrics": {
            "RMSE": winner.rmse,
            "MAE": winner.mae,
            "R2": winner.r2,
        },
        "best_epoch": winner.best_epoch,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "candidate_strategy_ids": list(PARTIAL_STRATEGY_ORDER),
        "candidate_validation": {
            score.strategy_id: {
                "RMSE": score.rmse,
                "MAE": score.mae,
                "R2": score.r2,
                "best_epoch": score.best_epoch,
                "checkpoint_path": str(score.checkpoint_path.resolve()),
            }
            for score in decision.candidates
        },
        "test_metrics_used_for_selection": False,
    }
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise PartialFTContractError(f"Selection is already locked: {path}") from exc
    return path


def load_selection_lock(selection_path: Path | str) -> Mapping[str, Any]:
    path = Path(selection_path)
    _require(path.is_file(), "Target Test is locked until selection.json exists")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartialFTContractError(f"Invalid selection lock: {exc}") from exc
    _require(payload.get("locked") is True, "Selection lock is not immutable/final")
    _require(
        payload.get("selection_basis") == "original_scale_validation",
        "Selection was not based on original-scale Validation",
    )
    _require(
        payload.get("test_metrics_used_for_selection") is False,
        "Test metrics cannot participate in selection",
    )
    _require(
        tuple(payload.get("candidate_strategy_ids", ())) == PARTIAL_STRATEGY_ORDER,
        "Selection candidate registry mismatch",
    )
    return MappingProxyType(payload)


def authorize_selected_test(
    selection_path: Path | str,
    *,
    strategy_id: str,
    checkpoint_model: ReloadedCheckpointModel,
) -> SelectedTestAuthorization:
    """Authorize only the locked winner and only a reloaded best checkpoint."""

    selection = load_selection_lock(selection_path)
    _require(
        strategy_id == selection["selected_strategy_id"],
        "Non-selected strategy may not access Target Test",
    )
    _require(
        isinstance(checkpoint_model, ReloadedCheckpointModel),
        "Final Target Test requires a reloaded checkpoint wrapper",
    )
    expected = Path(selection["selected_checkpoint_path"]).resolve()
    _require(
        checkpoint_model.checkpoint_path.resolve() == expected,
        "Reloaded checkpoint is not the selected best-Validation checkpoint",
    )
    return SelectedTestAuthorization(
        strategy_id=strategy_id,
        checkpoint_model=checkpoint_model,
        selection_path=Path(selection_path),
    )


def evaluate_selected_test(
    authorization: SelectedTestAuthorization,
    *,
    test: NamedSequenceSplit,
) -> Any:
    _require(
        isinstance(authorization, SelectedTestAuthorization),
        "Target Test requires selection authorization",
    )
    return evaluate_test(authorization.checkpoint_model, test=test, verbose=0)


def inverse_transform_original(
    normalized_values: np.ndarray,
    target_scaler: Any,
    *,
    expected_original: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(normalized_values).reshape(-1, 1)
    original = np.asarray(target_scaler.inverse_transform(values)).reshape(-1)
    _require(np.isfinite(original).all(), "Non-finite original-scale values")
    if expected_original is not None:
        expected = np.asarray(expected_original).reshape(-1)
        _require(len(expected) == len(original), "Original-scale count mismatch")
        _require(
            np.allclose(original, expected, rtol=1e-10, atol=1e-6),
            "Normalized/original target alignment mismatch",
        )
    return original


def zero_epoch_contract_summary(output_dir: Path | str) -> Mapping[str, Any]:
    """Load the approved Source and build both candidates without fitting."""

    baseline = validate_baseline_source_checkpoint()
    source = reload_best_checkpoint(baseline.checkpoint_path).model
    summaries: dict[str, Any] = {}
    optimizer_ids: list[int] = []
    for strategy_id in PARTIAL_STRATEGY_ORDER:
        candidate = build_partial_candidate(
            source,
            strategy_id=strategy_id,
            output_dir=output_dir,
        )
        validate_partial_transfer(source, candidate.model)
        optimizer_ids.append(candidate.optimizer_identity)
        summaries[strategy_id] = {
            "trainable_indices": list(candidate.strategy.trainable_indices),
            "frozen_indices": list(candidate.strategy.frozen_indices),
            "parameter_counts": candidate.parameter_counts.__dict__,
            "layers": [dict(row) for row in candidate.layer_manifest],
        }
    _require(len(set(optimizer_ids)) == len(optimizer_ids), "Candidates share an optimizer")
    return MappingProxyType(
        {
            "baseline_status": "PASS",
            "source_checkpoint": str(baseline.checkpoint_path),
            "source_checkpoint_sha256": baseline.sha256,
            "strategies": summaries,
            "training_epochs_executed": 0,
            "batch_size": R25_BATCH_SIZE,
            "max_epochs": R25_MAX_EPOCHS,
            "seed": R25_SEED,
        }
    )
