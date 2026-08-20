"""Zero-epoch model contracts for the Solar A2/B Linear protocol.

This module constructs and inspects models only.  It deliberately exposes no
training, checkpoint, prediction, or Test-evaluation API.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from keras.layers import TimeDistributed
from keras.optimizers import Adam

from r2_config.solar_linear import (
    BATCH_NORMALIZATION_LAYER_INDICES,
    EXPECTED_NON_TRAINABLE_PARAMS,
    EXPECTED_TOTAL_PARAMS,
    EXPECTED_TRAINABLE_PARAMS,
    FROZEN_LAYER_INDICES,
    LINEAR_ACTIVATION,
    PARTIAL_FT_STRATEGY_ID,
    TRAINABLE_LAYER_INDICES,
    TRANSFERRED_LAYER_INDICES,
    WEIGHT_BEARING_LAYER_INDICES,
)
from r2_config.solar_r2 import FEATURE_COLUMNS, WINDOW

# Keras 2.10 no longer publishes the historical wrappers module imported by
# utils.model.  Match the existing R2 compatibility shim without editing the
# Legacy import path.
if "keras.layers.wrappers" not in sys.modules:
    wrappers_module = types.ModuleType("keras.layers.wrappers")
    wrappers_module.TimeDistributed = TimeDistributed
    sys.modules["keras.layers.wrappers"] = wrappers_module

from utils.model import build_model, rmse


LINEAR_INPUT_SHAPE = (WINDOW, len(FEATURE_COLUMNS))
EXPECTED_LAYER_CLASSES = (
    "InputLayer",
    "TimeDistributed",
    "LSTM",
    "BatchNormalization",
    "LSTM",
    "BatchNormalization",
    "Dense",
)
EXPECTED_WEIGHT_SHAPES: Mapping[int, tuple[tuple[int, ...], ...]] = (
    MappingProxyType(
        {
            0: (),
            1: ((5, 10), (10,)),
            2: ((10, 240), (60, 240), (240,)),
            3: ((60,), (60,), (60,), (60,)),
            4: ((60, 240), (60, 240), (240,)),
            5: ((60,), (60,), (60,), (60,)),
            6: ((60, 1), (1,)),
        }
    )
)


class LinearRuntimeContractError(AssertionError):
    """Raised when the Phase-I Linear runtime contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LinearRuntimeContractError(message)


def _learning_rate(model: Any) -> float:
    value = getattr(model.optimizer, "learning_rate", None)
    if value is None:
        value = model.optimizer.lr
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(value)


def _explicit_learning_rate(value: float | None) -> float:
    _require(value is not None, "Linear lifecycle requires an explicit learning rate")
    _require(
        isinstance(value, (int, float)) and np.isfinite(value) and value > 0,
        "Explicit Linear learning rate must be finite and positive",
    )
    return float(value)


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int
    non_trainable: int


@dataclass(frozen=True)
class LinearPartialCandidate:
    strategy_id: str
    model: Any
    parameter_counts: ParameterCounts
    initial_non_transferred_weights: Mapping[int, tuple[np.ndarray, ...]]


def parameter_counts(model: Any) -> ParameterCounts:
    """Count parameters from the model's current post-mask weight lists."""

    count = lambda weights: int(sum(np.prod(tuple(weight.shape)) for weight in weights))
    trainable = count(model.trainable_weights)
    non_trainable = count(model.non_trainable_weights)
    return ParameterCounts(
        total=trainable + non_trainable,
        trainable=trainable,
        non_trainable=non_trainable,
    )


def validate_linear_model(
    model: Any,
    *,
    expected_learning_rate: float | None = None,
) -> None:
    """Require the exact seven-layer Linear backbone and compile contract."""

    classes = tuple(type(layer).__name__ for layer in model.layers)
    _require(classes == EXPECTED_LAYER_CLASSES, f"Layer topology mismatch: {classes}")
    _require(tuple(model.input_shape[1:]) == LINEAR_INPUT_SHAPE, "Input shape mismatch")
    _require(tuple(model.output_shape[1:]) == (1,), "Output shape mismatch")
    _require(model.layers[1].layer.units == 10, "TimeDistributed Dense units changed")
    _require(model.layers[2].units == 60, "First LSTM units changed")
    _require(model.layers[4].units == 60, "Second LSTM units changed")
    _require(model.layers[6].units == 1, "Output Dense units changed")
    _require(
        model.layers[6].activation.__name__ == LINEAR_ACTIVATION,
        "Linear protocol requires a linear output activation",
    )
    for index, expected in EXPECTED_WEIGHT_SHAPES.items():
        actual = tuple(tuple(weight.shape) for weight in model.layers[index].weights)
        _require(actual == expected, f"Layer {index} weight shapes changed: {actual}")
    _require(model.count_params() == EXPECTED_TOTAL_PARAMS, "Total parameter count changed")
    _require(type(model.optimizer).__name__ == "Adam", "Optimizer must be Adam")
    loss_name = model.loss if isinstance(model.loss, str) else model.loss.__name__
    _require(loss_name in ("mse", "mean_squared_error"), "Loss must be MSE")
    if expected_learning_rate is not None:
        _require(
            np.isclose(
                _learning_rate(model),
                expected_learning_rate,
                rtol=0.0,
                atol=1e-10,
            ),
            "Linear optimizer learning rate differs from the explicit value",
        )


def _build_fresh_linear_model(
    *,
    output_dir: Path | str,
    learning_rate: float | None,
) -> Any:
    explicit_lr = _explicit_learning_rate(learning_rate)
    model = build_model(
        input_shape=LINEAR_INPUT_SHAPE,
        gpu=False,
        write_result_out_dir=str(output_dir),
        pre_model=None,
        freeze=False,
        noise=None,
        verbose=False,
        savefig=False,
        output_activation=LINEAR_ACTIVATION,
        learning_rate=explicit_lr,
    )
    validate_linear_model(model, expected_learning_rate=explicit_lr)
    return model


def build_linear_source_model(
    *, output_dir: Path | str, learning_rate: float | None
) -> Any:
    """Build a fresh Linear Source model; no data or output file is touched."""

    return _build_fresh_linear_model(output_dir=output_dir, learning_rate=learning_rate)


def build_linear_without_tl_model(
    *, output_dir: Path | str, learning_rate: float | None
) -> Any:
    """Build a fresh Linear target model with no Source weight dependency."""

    return _build_fresh_linear_model(output_dir=output_dir, learning_rate=learning_rate)


def _weights(model: Any, index: int) -> tuple[np.ndarray, ...]:
    return tuple(weight.copy() for weight in model.layers[index].get_weights())


def _assert_weights_equal(
    expected: tuple[np.ndarray, ...],
    actual: tuple[np.ndarray, ...],
    *,
    index: int,
    purpose: str,
) -> None:
    _require(len(expected) == len(actual), f"Layer {index} {purpose} weight count mismatch")
    for expected_weight, actual_weight in zip(expected, actual):
        _require(expected_weight.shape == actual_weight.shape, f"Layer {index} shape mismatch")
        _require(np.array_equal(expected_weight, actual_weight), f"Layer {index} {purpose} mismatch")


def build_linear_partial_ft_candidate(
    source_model: Any,
    *,
    output_dir: Path | str,
    learning_rate: float | None,
) -> LinearPartialCandidate:
    """Copy layers 2--5, apply Candidate-B mask, and freshly compile.

    A separate fresh target is constructed first so the function can prove
    that target adapters 1 and 6 retain target initialization.  No Source
    output weight is copied.
    """

    explicit_lr = _explicit_learning_rate(learning_rate)
    validate_linear_model(source_model)
    target_initial = _build_fresh_linear_model(
        output_dir=output_dir,
        learning_rate=explicit_lr,
    )
    initial_non_transferred = MappingProxyType(
        {index: _weights(target_initial, index) for index in (1, 6)}
    )
    target = build_model(
        input_shape=LINEAR_INPUT_SHAPE,
        gpu=False,
        write_result_out_dir=str(output_dir),
        pre_model=source_model,
        freeze=False,
        noise=None,
        verbose=False,
        savefig=False,
        output_activation=LINEAR_ACTIVATION,
        learning_rate=explicit_lr,
    )
    validate_linear_model(target, expected_learning_rate=explicit_lr)

    for index in WEIGHT_BEARING_LAYER_INDICES:
        target.layers[index].trainable = index in TRAINABLE_LAYER_INDICES
    target.compile(
        optimizer=Adam(learning_rate=explicit_lr),
        loss="mse",
        metrics=["mse", rmse, "mae", "mape", "msle"],
    )

    for index in TRANSFERRED_LAYER_INDICES:
        _require(
            type(source_model.layers[index]).__name__
            == type(target.layers[index]).__name__,
            f"Transferred layer {index} class mismatch",
        )
        _assert_weights_equal(
            _weights(source_model, index),
            _weights(target, index),
            index=index,
            purpose="transfer",
        )
    for index in (1, 6):
        _assert_weights_equal(
            initial_non_transferred[index],
            _weights(target, index),
            index=index,
            purpose="target initialization",
        )

    actual_trainable = tuple(
        index for index in WEIGHT_BEARING_LAYER_INDICES if target.layers[index].trainable
    )
    actual_frozen = tuple(
        index for index in WEIGHT_BEARING_LAYER_INDICES if not target.layers[index].trainable
    )
    _require(actual_trainable == TRAINABLE_LAYER_INDICES, "Trainable mask mismatch")
    _require(actual_frozen == FROZEN_LAYER_INDICES, "Frozen mask mismatch")
    _require(
        all(not target.layers[index].trainable for index in BATCH_NORMALIZATION_LAYER_INDICES),
        "BatchNormalization layers must remain frozen",
    )
    counts = parameter_counts(target)
    _require(counts.total == EXPECTED_TOTAL_PARAMS, "Partial FT total params changed")
    _require(counts.trainable == EXPECTED_TRAINABLE_PARAMS, "Partial FT trainable params changed")
    _require(
        counts.non_trainable == EXPECTED_NON_TRAINABLE_PARAMS,
        "Partial FT non-trainable params changed",
    )
    validate_linear_model(target, expected_learning_rate=explicit_lr)
    return LinearPartialCandidate(
        strategy_id=PARTIAL_FT_STRATEGY_ID,
        model=target,
        parameter_counts=counts,
        initial_non_transferred_weights=initial_non_transferred,
    )
