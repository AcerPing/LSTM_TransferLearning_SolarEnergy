"""Immutable configuration for the Solar A2/B Linear protocol.

This module declares contracts only.  Importing it never creates an output
directory, loads a dataset, builds a model, or chooses a learning rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from r2_config.solar_r2 import (
    EXPECTED_ROW_COUNTS,
    EXPECTED_SEQUENCE_COUNTS,
    FEATURE_COLUMNS,
    FREQUENCY,
    HORIZON,
    TARGET_COLUMN,
    WINDOW,
    profile_path,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

LINEAR_ACTIVATION = "linear"
LINEAR_SEED = 1234
LINEAR_BATCH_SIZE = 128
LINEAR_MAXIMUM_EPOCHS = 500
LINEAR_OPTIMIZER = "Adam"
LINEAR_LOSS = "mse"

PARTIAL_FT_STRATEGY_ID = "partial_target_adapters_last_lstm"
TRANSFERRED_LAYER_INDICES = (2, 3, 4, 5)
TRAINABLE_LAYER_INDICES = (1, 4, 6)
FROZEN_LAYER_INDICES = (2, 3, 5)
BATCH_NORMALIZATION_LAYER_INDICES = (3, 5)
WEIGHT_BEARING_LAYER_INDICES = (1, 2, 3, 4, 5, 6)

EXPECTED_TOTAL_PARAMS = 46681
EXPECTED_TRAINABLE_PARAMS = 29161
EXPECTED_NON_TRAINABLE_PARAMS = 17520

# No LR is selected during Phase I.  Every later lifecycle must receive an
# explicit, validation-approved value before it can build a formal model.
LEARNING_RATE_POLICY = "required_explicit_before_training"
REQUIRED_EXPLICIT_LR_LIFECYCLES = (
    "source_pretrain",
    "without_tl",
    PARTIAL_FT_STRATEGY_ID,
)

LINEAR_FORMAL_BASE = (
    REPOSITORY_ROOT / "reports" / "Solar Energy Result" / "Linear_Formal"
)
PROTECTED_LEGACY_ROOTS = (
    REPOSITORY_ROOT / "reports" / "Solar Energy Result" / "R2" / "Experiment_A",
    REPOSITORY_ROOT / "reports" / "Solar Energy Result" / "R2.5" / "Experiment_A",
)


@dataclass(frozen=True)
class LinearExperimentSpec:
    experiment_id: str
    direction: str
    source_plant: str
    target_plant: str
    source_profile_path: Path
    target_profile_path: Path
    output_root: Path
    activation: str = LINEAR_ACTIVATION
    window: int = WINDOW
    horizon: int = HORIZON
    frequency: str = FREQUENCY
    features: tuple[str, ...] = FEATURE_COLUMNS
    target_column: str = TARGET_COLUMN
    seed: int = LINEAR_SEED
    batch_size: int = LINEAR_BATCH_SIZE
    maximum_epochs: int = LINEAR_MAXIMUM_EPOCHS
    optimizer: str = LINEAR_OPTIMIZER
    loss: str = LINEAR_LOSS
    partial_ft_strategy_id: str = PARTIAL_FT_STRATEGY_ID
    transferred_indices: tuple[int, ...] = TRANSFERRED_LAYER_INDICES
    trainable_indices: tuple[int, ...] = TRAINABLE_LAYER_INDICES
    frozen_indices: tuple[int, ...] = FROZEN_LAYER_INDICES
    batch_normalization_frozen: bool = True
    learning_rate_policy: str = LEARNING_RATE_POLICY


LINEAR_EXPERIMENTS: Mapping[str, LinearExperimentSpec] = MappingProxyType(
    {
        "A2": LinearExperimentSpec(
            experiment_id="A2",
            direction="Plant1_to_Plant2",
            source_plant="Plant1",
            target_plant="Plant2",
            source_profile_path=profile_path("A", "source"),
            target_profile_path=profile_path("A", "target"),
            output_root=LINEAR_FORMAL_BASE / "Experiment_A2",
        ),
        "B": LinearExperimentSpec(
            experiment_id="B",
            direction="Plant2_to_Plant1",
            source_plant="Plant2",
            target_plant="Plant1",
            source_profile_path=profile_path("B", "source"),
            target_profile_path=profile_path("B", "target"),
            output_root=LINEAR_FORMAL_BASE / "Experiment_B",
        ),
    }
)


def linear_experiment(experiment_id: str) -> LinearExperimentSpec:
    """Return one immutable A2/B specification without touching the filesystem."""

    try:
        return LINEAR_EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise ValueError(f"Unknown Linear experiment: {experiment_id!r}") from exc


def expected_counts(role: str) -> tuple[Mapping[str, int], Mapping[str, int]]:
    """Return immutable-source row/sequence count contracts for one role."""

    if role not in ("source", "target"):
        raise ValueError(f"Unknown role: {role!r}")
    profile_name = "source_profile" if role == "source" else "target_profile"
    return EXPECTED_ROW_COUNTS[profile_name], EXPECTED_SEQUENCE_COUNTS[profile_name]
