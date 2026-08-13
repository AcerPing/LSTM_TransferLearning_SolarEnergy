"""Build-only runtime contract for corrected-R1 Solar R2 experiments.

This module preserves the Legacy model and transfer-learning strategy while
keeping training, validation, checkpoint selection, and test evaluation roles
explicitly separated.
"""

from __future__ import annotations

import os
import random
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import TimeDistributed
from keras.models import load_model

from r2_config.solar_r2 import FEATURE_COLUMNS, WINDOW

# Keras 2.10 removed the Legacy import module used by utils.model.  Provide
# only that historical symbol in memory so the unchanged Legacy file imports.
if "keras.layers.wrappers" not in sys.modules:
    wrappers_module = types.ModuleType("keras.layers.wrappers")
    wrappers_module.TimeDistributed = TimeDistributed
    sys.modules["keras.layers.wrappers"] = wrappers_module

from utils.model import build_model, rmse


R2_INPUT_SHAPE = (WINDOW, len(FEATURE_COLUMNS))
R2_BATCH_SIZE = 128
R2_DEFAULT_SEED = 1234
TRANSFERABLE_LAYER_INDICES = (2, 3, 4, 5)
NON_TRANSFERRED_WEIGHT_LAYER_INDICES = (1, 6)
R2_METHOD_NAMES = ("without_tl", "tl_freeze", "tl_full_finetune")
R2_CUSTOM_OBJECTS: Mapping[str, Any] = MappingProxyType({"rmse": rmse})

EXPECTED_LAYER_TOPOLOGY = (
    "InputLayer",
    "TimeDistributed",
    "LSTM",
    "BatchNormalization",
    "LSTM",
    "BatchNormalization",
    "Dense",
)

_RELOAD_PROOF = object()


class RuntimeContractError(AssertionError):
    """Raised when the R2 model/runtime lifecycle contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


@dataclass(frozen=True)
class ReproducibilityStatus:
    seed: int
    pythonhashseed: str | None
    pythonhashseed_matches_seed: bool
    strict_hash_reproducibility: bool
    deterministic_ops_requested: bool
    deterministic_ops_enabled: bool


@dataclass(frozen=True)
class CheckpointPaths:
    source_pretrain: Path
    without_tl: Path
    tl_freeze: Path
    tl_full_finetune: Path

    def as_dict(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                "source_pretrain": self.source_pretrain,
                "without_tl": self.without_tl,
                "tl_freeze": self.tl_freeze,
                "tl_full_finetune": self.tl_full_finetune,
            }
        )


@dataclass(frozen=True)
class NamedSequenceSplit:
    name: str
    X_seq: np.ndarray
    y_seq: np.ndarray

    def __post_init__(self) -> None:
        X = np.asarray(self.X_seq)
        y = np.asarray(self.y_seq)
        _require(X.ndim == 3, f"{self.name} X_seq must be 3-D")
        _require(y.ndim in (1, 2), f"{self.name} y_seq must be 1-D or 2-D")
        _require(len(X) == len(y), f"{self.name} X/y sequence counts differ")


@dataclass(frozen=True)
class ReloadedCheckpointModel:
    model: Any
    checkpoint_path: Path
    _proof: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class TestEvaluation:
    checkpoint_path: Path
    y_true: np.ndarray
    predictions: np.ndarray


def configure_reproducibility(
    seed: int = R2_DEFAULT_SEED,
    *,
    deterministic_ops: bool = True,
) -> ReproducibilityStatus:
    """Set runtime RNGs while only inspecting, never changing, PYTHONHASHSEED."""

    _require(isinstance(seed, int) and seed >= 0, "Seed must be a non-negative int")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    deterministic_enabled = False
    if deterministic_ops:
        enable_determinism = getattr(
            getattr(tf.config, "experimental", object()),
            "enable_op_determinism",
            None,
        )
        _require(
            callable(enable_determinism),
            "TensorFlow does not expose enable_op_determinism",
        )
        try:
            enable_determinism()
        except Exception as exc:
            raise RuntimeContractError(
                f"Cannot enable deterministic TensorFlow operations: {exc}"
            ) from exc
        deterministic_enabled = True

    hash_seed = os.environ.get("PYTHONHASHSEED")
    hash_matches = hash_seed == str(seed)
    return ReproducibilityStatus(
        seed=seed,
        pythonhashseed=hash_seed,
        pythonhashseed_matches_seed=hash_matches,
        strict_hash_reproducibility=hash_matches,
        deterministic_ops_requested=deterministic_ops,
        deterministic_ops_enabled=deterministic_enabled,
    )


def checkpoint_paths(run_root: Path | str) -> CheckpointPaths:
    """Return the four independent R2 best-checkpoint destinations."""

    root = Path(run_root)
    paths = CheckpointPaths(
        source_pretrain=root / "source" / "pretrain" / "best_model.hdf5",
        without_tl=root / "target" / "without_tl" / "best_model.hdf5",
        tl_freeze=root / "target" / "tl_freeze" / "best_model.hdf5",
        tl_full_finetune=(
            root / "target" / "tl_full_finetune" / "best_model.hdf5"
        ),
    )
    values = tuple(paths.as_dict().values())
    _require(len(set(values)) == len(values), "Checkpoint paths must be unique")
    return paths


def _learning_rate(model: Any) -> float:
    value = getattr(model.optimizer, "learning_rate", None)
    if value is None:
        value = model.optimizer.lr
    return float(K.get_value(value))


def _validate_architecture(model: Any) -> None:
    topology = tuple(type(layer).__name__ for layer in model.layers)
    _require(topology == EXPECTED_LAYER_TOPOLOGY, f"Layer topology mismatch: {topology}")
    _require(tuple(model.input_shape[1:]) == R2_INPUT_SHAPE, "Model input shape mismatch")
    _require(tuple(model.output_shape[1:]) == (1,), "Model output shape mismatch")
    _require(model.layers[1].layer.units == 10, "TimeDistributed Dense units mismatch")
    _require(model.layers[2].units == 60, "First LSTM units mismatch")
    _require(model.layers[4].units == 60, "Second LSTM units mismatch")
    _require(model.layers[6].units == 1, "Output Dense units mismatch")
    _require(
        model.layers[6].activation.__name__ == "sigmoid",
        "R2 output activation must remain sigmoid",
    )


def validate_model_contract(model: Any, *, transfer_learning: bool) -> None:
    """Validate the immutable Legacy architecture and compile settings."""

    _validate_architecture(model)
    _require(type(model.optimizer).__name__ == "Adam", "Optimizer must be Adam")
    loss_name = model.loss if isinstance(model.loss, str) else model.loss.__name__
    _require(loss_name in ("mse", "mean_squared_error"), "Loss must be MSE")
    expected_lr = 1e-5 if transfer_learning else 1e-4
    _require(
        np.isclose(_learning_rate(model), expected_lr, rtol=0.0, atol=1e-10),
        f"Learning rate mismatch: expected {expected_lr}, got {_learning_rate(model)}",
    )


def build_r2_model(
    *,
    output_dir: Path | str,
    pretrained_model: Any | None = None,
    freeze: bool = False,
) -> Any:
    """Build the unchanged Legacy model for the fixed R2 (5, 5) input."""

    if pretrained_model is None:
        _require(not freeze, "freeze=True requires a pretrained model")
    else:
        _validate_architecture(pretrained_model)

    model = build_model(
        input_shape=R2_INPUT_SHAPE,
        gpu=False,
        write_result_out_dir=str(output_dir),
        pre_model=pretrained_model,
        freeze=freeze,
        noise=None,
        verbose=False,
        savefig=False,
    )
    validate_model_contract(model, transfer_learning=pretrained_model is not None)
    return model


def validate_transferred_weights(
    source_model: Any,
    target_model: Any,
    target_initial_model: Any,
    *,
    freeze: bool,
) -> None:
    """Prove exact Legacy copy scope, trainability, and independent variables."""

    for model in (source_model, target_model, target_initial_model):
        _validate_architecture(model)

    for index in TRANSFERABLE_LAYER_INDICES:
        source_weights = source_model.layers[index].get_weights()
        target_weights = target_model.layers[index].get_weights()
        _require(len(source_weights) == len(target_weights), f"Weight count mismatch at layer {index}")
        for source_weight, target_weight in zip(source_weights, target_weights):
            _require(source_weight.shape == target_weight.shape, f"Weight shape mismatch at layer {index}")
            _require(np.array_equal(source_weight, target_weight), f"Transferred weight mismatch at layer {index}")
        _require(
            target_model.layers[index].trainable is (not freeze),
            f"Trainable flag mismatch at layer {index}",
        )

    for index in NON_TRANSFERRED_WEIGHT_LAYER_INDICES:
        initial_weights = target_initial_model.layers[index].get_weights()
        target_weights = target_model.layers[index].get_weights()
        _require(len(initial_weights) == len(target_weights), f"Initial weight count mismatch at layer {index}")
        for initial_weight, target_weight in zip(initial_weights, target_weights):
            _require(
                np.array_equal(initial_weight, target_weight),
                f"Non-transferred layer {index} did not retain target initialization",
            )
        _require(target_model.layers[index].trainable, f"Layer {index} must remain trainable")

    for index in range(len(target_model.layers)):
        source_variables = source_model.layers[index].weights
        target_variables = target_model.layers[index].weights
        _require(len(source_variables) == len(target_variables), f"Variable count mismatch at layer {index}")
        for source_variable, target_variable in zip(source_variables, target_variables):
            _require(
                source_variable is not target_variable,
                f"Source and target share a weight object at layer {index}",
            )

    if not freeze:
        _require(
            all(layer.trainable for layer in target_model.layers),
            "Full fine-tuning requires every intended model layer to be trainable",
        )


def make_r2_callbacks(checkpoint_path: Path | str) -> tuple[Any, ...]:
    """Create the fixed Legacy callback set without touching the filesystem."""

    destination = Path(checkpoint_path)
    _require(destination.name == "best_model.hdf5", "Unexpected checkpoint filename")
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=str(destination),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    return reduce_lr, checkpoint, early_stopping


def fit_train_validation(
    model: Any,
    *,
    training: NamedSequenceSplit,
    validation: NamedSequenceSplit,
    callbacks: Sequence[Any],
    epochs: int,
    verbose: int = 1,
) -> Any:
    """Fit only explicit training/validation splits; no test argument exists."""

    _require(isinstance(training, NamedSequenceSplit), "training split type mismatch")
    _require(isinstance(validation, NamedSequenceSplit), "validation split type mismatch")
    _require(training.name == "training", "fit training split must be named training")
    _require(validation.name == "validation", "fit validation split must be named validation")
    _require(isinstance(epochs, int) and epochs > 0, "epochs must be a positive int")
    return model.fit(
        training.X_seq,
        training.y_seq,
        validation_data=(validation.X_seq, validation.y_seq),
        epochs=epochs,
        batch_size=R2_BATCH_SIZE,
        shuffle=True,
        callbacks=list(callbacks),
        verbose=verbose,
    )


def reload_best_checkpoint(checkpoint_path: Path | str) -> ReloadedCheckpointModel:
    """Load a complete saved best model with the Legacy custom RMSE object."""

    path = Path(checkpoint_path)
    _require(path.is_file(), f"Best checkpoint not found: {path}")
    try:
        model = load_model(str(path), custom_objects=dict(R2_CUSTOM_OBJECTS))
    except Exception as exc:
        raise RuntimeContractError(f"Cannot reload best checkpoint {path}: {exc}") from exc
    _validate_architecture(model)
    return ReloadedCheckpointModel(
        model=model,
        checkpoint_path=path,
        _proof=_RELOAD_PROOF,
    )


def evaluate_test(
    checkpoint_model: ReloadedCheckpointModel,
    *,
    test: NamedSequenceSplit,
    verbose: int = 0,
) -> TestEvaluation:
    """Predict an ordered test split using only a proven reloaded checkpoint."""

    _require(
        isinstance(checkpoint_model, ReloadedCheckpointModel)
        and checkpoint_model._proof is _RELOAD_PROOF,
        "Final test evaluation requires reload_best_checkpoint() output",
    )
    _require(isinstance(test, NamedSequenceSplit), "test split type mismatch")
    _require(test.name == "test", "Evaluation split must be named test")
    predictions = np.asarray(
        checkpoint_model.model.predict(
            test.X_seq,
            batch_size=R2_BATCH_SIZE,
            verbose=verbose,
        )
    )
    _require(len(predictions) == len(test.y_seq), "Prediction/test count mismatch")
    return TestEvaluation(
        checkpoint_path=checkpoint_model.checkpoint_path,
        y_true=np.asarray(test.y_seq).copy(),
        predictions=predictions.copy(),
    )
