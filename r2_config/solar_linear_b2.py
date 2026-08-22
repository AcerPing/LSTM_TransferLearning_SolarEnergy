"""Immutable Phase B2-1 supplementary tuning configuration.

This protocol is explicitly post-Test and exploratory.  Importing this module
does not read data, build a model, create output, or start training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from r2_config.solar_linear import (
    BATCH_NORMALIZATION_LAYER_INDICES,
    LINEAR_BATCH_SIZE,
    LINEAR_EXPERIMENTS,
    LINEAR_LOSS,
    LINEAR_MAXIMUM_EPOCHS,
    LINEAR_SEED,
    REPOSITORY_ROOT,
    WEIGHT_BEARING_LAYER_INDICES,
)
from r2_config.solar_linear_formal import (
    FORMAL_CALLBACK_POLICY,
    FORMAL_DEVICE_POLICY,
)
from r2_config.solar_r2 import FEATURE_COLUMNS, FREQUENCY, HORIZON, TARGET_COLUMN, WINDOW


B2_PROTOCOL_VERSION = "solar-linear-b2-v1.0"
B2_EXPERIMENT_ID = "B2"
B2_DIRECTION = "Plant2_to_Plant1"
B2_SOURCE_PLANT = "Plant2"
B2_TARGET_PLANT = "Plant1"
B2_FORMAL_EXPERIMENT_ID = "B"
B2_FORMAL_RUN_ID = "20260821T041718Z_seed1234"
B2_SELECTION_GIT_HEAD = "ec6e881cf95f03bedd81adff10e55151a8b26ff5"
B2_FORMAL_FINAL_TEST_GIT_HEAD = "0e31520644836368f460ce1c740de4ed14318068"
B2_FORMAL_CLASSIFICATION = "Negative Transfer"

B2_FORMAL_RUN_ROOT = (
    LINEAR_EXPERIMENTS[B2_FORMAL_EXPERIMENT_ID].output_root / B2_FORMAL_RUN_ID
)
B2_FORMAL_FINAL_TEST_ROOT = B2_FORMAL_RUN_ROOT / "final_test"
B2_FORMAL_COMPARISON_PATH = B2_FORMAL_FINAL_TEST_ROOT / "comparison.json"
B2_TARGET_PROFILE_PATH = LINEAR_EXPERIMENTS[B2_FORMAL_EXPERIMENT_ID].target_profile_path
B2_WOTL_BASELINE_PATH = (
    B2_FORMAL_FINAL_TEST_ROOT / "wotl_metrics_original_scale.json"
)

B2_LOCKED_SOURCE_CANDIDATE_ID = "SRC_lr1e-4"
B2_LOCKED_SOURCE_EPOCH = 500
B2_LOCKED_SOURCE_SHA256 = (
    "5cd3db4501d6c6720fbdd8ee3cce692b01cecb82619574f8c6944a8795182d91"
)
B2_LOCKED_SOURCE_CHECKPOINT = (
    B2_FORMAL_RUN_ROOT
    / "source_candidates"
    / B2_LOCKED_SOURCE_CANDIDATE_ID
    / f"checkpoint_epoch_{B2_LOCKED_SOURCE_EPOCH:04d}.hdf5"
)

B2_FIXED_WOTL_CANDIDATE_ID = "WOTL_lr1e-4"
B2_FIXED_WOTL_METRICS: Mapping[str, float | int | str] = MappingProxyType(
    {
        "mae": 28769.9078045441,
        "mse": 1392464754.3387341,
        "rmse": 37315.744054470284,
        "r2": 0.8208138626892281,
        "prediction_count": 2607,
        "negative_prediction_count": 21,
        "target_name": "DC_POWER",
        "scale": "original",
        "unit": "DC_POWER (raw dataset scale; documented unit: kW)",
    }
)

B2_DISCLOSURE: Mapping[str, bool] = MappingProxyType(
    {
        "formal_experiment_b_preserved": True,
        "original_plant1_test_previously_revealed": True,
        "plant1_test_reused": True,
        "post_test_supplementary_tuning": True,
        "test_used_for_supplementary_comparison": True,
        "formal_final_test_replacement": False,
    }
)

B2_OUTPUT_BASE = (
    REPOSITORY_ROOT
    / "reports"
    / "Solar Energy Result"
    / "Linear_Supplementary"
    / "Experiment_B2"
)
B2_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z_seed1234$")
B2_AUTOMATIC_NEXT_ROUND = False

B2_TRANSFERRED_LAYER_INDICES = (1, 2, 3, 4, 5)
B2_BATCH_NORMALIZATION_LAYER_INDICES = BATCH_NORMALIZATION_LAYER_INDICES
B2_TOTAL_PARAMS = 46681
B2_LAST_LSTM_OUTPUT_TRAINABLE_PARAMS = 29101
B2_OUTPUT_ONLY_TRAINABLE_PARAMS = 61


@dataclass(frozen=True)
class B2CandidateSpec:
    candidate_id: str
    strategy: str
    learning_rate: float
    transferred_indices: tuple[int, ...]
    trainable_indices: tuple[int, ...]
    frozen_indices: tuple[int, ...]
    expected_trainable_params: int
    batch_normalization_frozen: bool = True


B2_CANDIDATES: Mapping[str, B2CandidateSpec] = MappingProxyType(
    {
        "B2-P1": B2CandidateSpec(
            candidate_id="B2-P1",
            strategy="Last LSTM + Output Fine-tuning",
            learning_rate=1e-5,
            transferred_indices=B2_TRANSFERRED_LAYER_INDICES,
            trainable_indices=(4, 6),
            frozen_indices=(1, 2, 3, 5),
            expected_trainable_params=B2_LAST_LSTM_OUTPUT_TRAINABLE_PARAMS,
        ),
        "B2-P2": B2CandidateSpec(
            candidate_id="B2-P2",
            strategy="Last LSTM + Output Fine-tuning",
            learning_rate=3e-6,
            transferred_indices=B2_TRANSFERRED_LAYER_INDICES,
            trainable_indices=(4, 6),
            frozen_indices=(1, 2, 3, 5),
            expected_trainable_params=B2_LAST_LSTM_OUTPUT_TRAINABLE_PARAMS,
        ),
        "B2-P3": B2CandidateSpec(
            candidate_id="B2-P3",
            strategy="Output Head Only",
            learning_rate=1e-5,
            transferred_indices=B2_TRANSFERRED_LAYER_INDICES,
            trainable_indices=(6,),
            frozen_indices=(1, 2, 3, 4, 5),
            expected_trainable_params=B2_OUTPUT_ONLY_TRAINABLE_PARAMS,
        ),
    }
)

B2_FEATURES = FEATURE_COLUMNS
B2_TARGET_COLUMN = TARGET_COLUMN
B2_WINDOW = WINDOW
B2_HORIZON = HORIZON
B2_FREQUENCY = FREQUENCY
B2_EXPECTED_TEST_ROWS = 2612
B2_EXPECTED_TEST_SEQUENCES = 2607
B2_EXPECTED_X_TEST_SHAPE = (2607, 5, 5)
B2_EXPECTED_Y_TEST_SHAPE = (2607,)
B2_EXPECTED_TARGET_ROW_COUNTS: Mapping[str, int] = MappingProxyType(
    {"training": 521, "validation": 131, "test": 2612}
)
B2_EXPECTED_TARGET_SEQUENCE_COUNTS: Mapping[str, int] = MappingProxyType(
    {"training": 516, "validation": 126, "test": 2607}
)
B2_SCALER_FIT_SPLIT = "training"
B2_VALIDATION_TEST_TRANSFORM_ONLY = True
B2_SEED = LINEAR_SEED
B2_BATCH_SIZE = LINEAR_BATCH_SIZE
B2_MAXIMUM_EPOCHS = LINEAR_MAXIMUM_EPOCHS
B2_LOSS = LINEAR_LOSS
B2_SHUFFLE = False
B2_CALLBACK_POLICY = FORMAL_CALLBACK_POLICY
B2_DEVICE_POLICY = FORMAL_DEVICE_POLICY


def b2_candidate(candidate_id: str) -> B2CandidateSpec:
    try:
        return B2_CANDIDATES[candidate_id]
    except KeyError as exc:
        raise ValueError(f"Unknown or unapproved B2 candidate: {candidate_id!r}") from exc


def validate_b2_config() -> None:
    """Fail if the controlled three-candidate supplementary matrix drifts."""

    if tuple(B2_CANDIDATES) != ("B2-P1", "B2-P2", "B2-P3"):
        raise ValueError("B2 candidate registry must contain only P1/P2/P3")
    if any(
        index in candidate.trainable_indices
        for candidate in B2_CANDIDATES.values()
        for index in B2_BATCH_NORMALIZATION_LAYER_INDICES
    ):
        raise ValueError("B2 BatchNormalization layers must remain frozen")
    for candidate in B2_CANDIDATES.values():
        if tuple(sorted(candidate.trainable_indices + candidate.frozen_indices)) != tuple(
            WEIGHT_BEARING_LAYER_INDICES
        ):
            raise ValueError(f"Incomplete B2 trainable mask: {candidate.candidate_id}")
        if candidate.transferred_indices != B2_TRANSFERRED_LAYER_INDICES:
            raise ValueError(f"B2 transfer scope changed: {candidate.candidate_id}")
    if B2_CALLBACK_POLICY.maximum_epochs != 500:
        raise ValueError("B2 maximum epochs changed")
    if B2_BATCH_SIZE != 128 or B2_SEED != 1234 or B2_SHUFFLE is not False:
        raise ValueError("B2 fit contract changed")
    if B2_LOSS != "mse" or B2_AUTOMATIC_NEXT_ROUND is not False:
        raise ValueError("B2 loss/stop contract changed")
