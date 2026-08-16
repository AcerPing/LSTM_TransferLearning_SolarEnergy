"""Immutable configuration for Solar R2.5 partial fine-tuning.

This module defines provenance, protocol constants, and strategy metadata only.
It intentionally contains no model fitting or result-writing logic.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

R25_EXPERIMENT = "A"
R25_SEED = 1234
R25_BATCH_SIZE = 128
R25_LEARNING_RATE = 1e-5
R25_LOSS = "mse"
R25_MAX_EPOCHS = 500
R25_OUTPUT_BASE = (
    REPOSITORY_ROOT
    / "reports"
    / "Solar Energy Result"
    / "R2.5"
    / "Experiment_A"
    / "Partial_FT"
)
R25_SMOKE_OUTPUT_BASE = (
    REPOSITORY_ROOT
    / "reports"
    / "Solar Energy Result"
    / "R2.5"
    / "_smoke"
)

BASELINE_RUN_ID = "20260814T150304Z_seed1234"
BASELINE_RUN_ROOT = (
    REPOSITORY_ROOT
    / "reports"
    / "Solar Energy Result"
    / "R2"
    / "Experiment_A"
    / BASELINE_RUN_ID
)
BASELINE_MANIFEST_PATH = BASELINE_RUN_ROOT / "run_manifest.json"
BASELINE_SOURCE_CHECKPOINT_RELATIVE = Path(
    "source/pretrain/checkpoint_epoch_0472.hdf5"
)
BASELINE_SOURCE_CHECKPOINT_SHA256 = (
    "8aa299e09c58c2ce77473ba317bdaf9fe374310a453b61718b605022356ad6aa"
)
BASELINE_SOURCE_BEST_EPOCH = 472

EXPECTED_LAYER_CLASSES = (
    "InputLayer",
    "TimeDistributed",
    "LSTM",
    "BatchNormalization",
    "LSTM",
    "BatchNormalization",
    "Dense",
)
TRANSFERRED_LAYER_INDICES = (2, 3, 4, 5)
TARGET_ADAPTER_INDICES = (1, 6)
BATCH_NORMALIZATION_INDICES = (3, 5)
WEIGHT_BEARING_LAYER_INDICES = (1, 2, 3, 4, 5, 6)

PRIMARY_STRATEGY_ID = "partial_target_adapters_last_lstm"
PARTIAL_STRATEGY_ORDER = (PRIMARY_STRATEGY_ID,)

PARTIAL_STRATEGY_SPECS = MappingProxyType(
    {
        PRIMARY_STRATEGY_ID: MappingProxyType(
            {
                "trainable_indices": (1, 4, 6),
                "frozen_indices": (2, 3, 5),
                "purpose": (
                    "Adapt both target-specific Dense interfaces and the final "
                    "LSTM while preserving the first transferred LSTM and both BN layers."
                ),
            }
        ),
    }
)

EXPECTED_PARAMETER_COUNTS = MappingProxyType(
    {
        "tl_freeze": 121,
        PRIMARY_STRATEGY_ID: 29161,
        "tl_full_finetune": 46441,
        "total": 46681,
    }
)
