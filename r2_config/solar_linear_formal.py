"""Immutable Phase-III policy for Solar Linear A2/B formal experiments.

The module defines candidates, callbacks, namespaces, and disclosure policy.
Importing it performs no data access and creates no directory.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from r2_config.solar_linear import (
    LINEAR_EXPERIMENTS,
    LINEAR_FORMAL_BASE,
    LINEAR_MAXIMUM_EPOCHS,
    LINEAR_SEED,
    PROTECTED_LEGACY_ROOTS,
    REPOSITORY_ROOT,
)


FORMAL_PROTOCOL_VERSION = "solar-linear-v1.0"
FORMAL_LIFECYCLES = ("source", "wotl", "partial_ft")
FORMAL_LINEAR_LAYER_CLASSES = (
    "InputLayer",
    "TimeDistributed",
    "LSTM",
    "BatchNormalization",
    "LSTM",
    "BatchNormalization",
    "Dense",
)

LEARNING_RATE_CANDIDATES: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "source": (1e-4, 3e-5),
        "wotl": (1e-4, 3e-5),
        "partial_ft": (1e-5, 3e-5),
    }
)

CANDIDATE_PREFIXES: Mapping[str, str] = MappingProxyType(
    {"source": "SRC", "wotl": "WOTL", "partial_ft": "PFT"}
)

VALIDATION_LOSS_TIE_RTOL = 1e-6
VALIDATION_DIAGNOSTIC_TIE_RTOL = 1e-9
VALIDATION_TIE_ATOL = 1e-12


@dataclass(frozen=True)
class ModelCheckpointPolicy:
    monitor: str = "val_loss"
    save_best_only: bool = True
    save_weights_only: bool = False


@dataclass(frozen=True)
class ReduceLROnPlateauPolicy:
    monitor: str = "val_loss"
    factor: float = 0.1
    patience: int = 20
    min_lr: float = 1e-7
    cooldown: int = 0


@dataclass(frozen=True)
class EarlyStoppingPolicy:
    monitor: str = "val_loss"
    patience: int = 50
    min_delta: float = 0.0
    restore_best_weights: bool = False


@dataclass(frozen=True)
class FormalCallbackPolicy:
    maximum_epochs: int
    checkpoint: ModelCheckpointPolicy
    reduce_lr: ReduceLROnPlateauPolicy
    early_stopping: EarlyStoppingPolicy
    terminate_on_nan: bool = True


FORMAL_CALLBACK_POLICY = FormalCallbackPolicy(
    maximum_epochs=LINEAR_MAXIMUM_EPOCHS,
    checkpoint=ModelCheckpointPolicy(),
    reduce_lr=ReduceLROnPlateauPolicy(),
    early_stopping=EarlyStoppingPolicy(),
)


@dataclass(frozen=True)
class FormalDevicePolicy:
    device: str = "CPU"
    cuda_visible_devices: str = "-1"
    priority: str = "reproducibility_over_speed"
    seed: int = LINEAR_SEED
    require_pythonhashseed_before_start: bool = True
    deterministic_ops: bool = True


FORMAL_DEVICE_POLICY = FormalDevicePolicy()

LINEAR_SMOKE_BASE = (
    REPOSITORY_ROOT / "reports" / "Solar Energy Result" / "_smoke_linear"
)
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z_seed1234$")
HISTORICAL_TARGET_TEST_PREVIOUSLY_REVEALED: Mapping[str, bool] = MappingProxyType(
    {"A2": True, "B": False}
)


@dataclass(frozen=True)
class FormalPathContract:
    experiment_id: str
    run_id: str
    experiment_root: Path
    run_root: Path
    protocol_manifest: Path
    source_candidates: Path
    target_wotl_candidates: Path
    target_partial_ft_candidates: Path
    selection: Path
    final: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _lr_token(learning_rate: float) -> str:
    _require(math.isfinite(learning_rate) and learning_rate > 0, "Invalid LR")
    mantissa, exponent = f"{learning_rate:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def candidate_id(lifecycle: str, learning_rate: float) -> str:
    """Return a stable ID only for a pre-registered LR candidate."""

    _require(lifecycle in FORMAL_LIFECYCLES, f"Unknown lifecycle: {lifecycle}")
    _require(
        any(math.isclose(learning_rate, item, rel_tol=0.0, abs_tol=1e-15) for item in LEARNING_RATE_CANDIDATES[lifecycle]),
        f"LR {learning_rate} is not registered for {lifecycle}",
    )
    return f"{CANDIDATE_PREFIXES[lifecycle]}_lr{_lr_token(learning_rate)}"


def candidate_registry() -> Mapping[str, tuple[tuple[str, float], ...]]:
    return MappingProxyType(
        {
            lifecycle: tuple(
                (candidate_id(lifecycle, learning_rate), learning_rate)
                for learning_rate in LEARNING_RATE_CANDIDATES[lifecycle]
            )
            for lifecycle in FORMAL_LIFECYCLES
        }
    )


def formal_path_contract(experiment_id: str, run_id: str) -> FormalPathContract:
    """Resolve the formal namespace without creating it."""

    _require(experiment_id in LINEAR_EXPERIMENTS, "Unknown Linear experiment")
    _require(bool(RUN_ID_PATTERN.fullmatch(run_id)), "Invalid formal run_id")
    experiment_root = LINEAR_EXPERIMENTS[experiment_id].output_root
    run_root = experiment_root / run_id
    resolved = run_root.resolve()
    _require(resolved.is_relative_to(LINEAR_FORMAL_BASE.resolve()), "Formal root escaped")
    _require(not resolved.is_relative_to(LINEAR_SMOKE_BASE.resolve()), "Formal/smoke overlap")
    _require(
        not any(resolved.is_relative_to(root.resolve()) for root in PROTECTED_LEGACY_ROOTS),
        "Formal root overlaps protected Legacy output",
    )
    return FormalPathContract(
        experiment_id=experiment_id,
        run_id=run_id,
        experiment_root=experiment_root,
        run_root=run_root,
        protocol_manifest=run_root / "protocol_manifest.json",
        source_candidates=run_root / "source_candidates",
        target_wotl_candidates=run_root / "target_wotl_candidates",
        target_partial_ft_candidates=run_root / "target_partial_ft_candidates",
        selection=run_root / "selection",
        final=run_root / "final",
    )


def validate_new_run_destination(paths: FormalPathContract) -> None:
    """Reject collisions; this function never creates a destination."""

    _require(not paths.run_root.exists(), f"Formal run already exists: {paths.run_root}")


def validate_formal_policy() -> None:
    """Validate cross-field invariants once without selecting a candidate."""

    _require(FORMAL_CALLBACK_POLICY.maximum_epochs == 500, "Maximum epochs changed")
    _require(
        FORMAL_CALLBACK_POLICY.early_stopping.patience
        > FORMAL_CALLBACK_POLICY.reduce_lr.patience,
        "EarlyStopping patience must exceed ReduceLR patience",
    )
    _require(
        all(len(values) == 2 for values in LEARNING_RATE_CANDIDATES.values()),
        "Each lifecycle must retain exactly two LR candidates",
    )
    ids = [item[0] for values in candidate_registry().values() for item in values]
    _require(len(ids) == len(set(ids)), "Formal candidate IDs are not unique")
