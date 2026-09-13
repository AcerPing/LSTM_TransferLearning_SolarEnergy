"""Configuration-only contract for the Corrected R1 data protocol.

This module contains paths and immutable data expectations only.  It has no
model-building, training, checkpoint, or result-writing logic.
"""

from pathlib import Path


CORRECTED_R1_ROOT = Path(
    r"D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量"
    r"\Code程式碼\PersonalNote\DataSet_For_TransferLearning"
    r"\★★ Solar Power Generation Data ★★\preprocessed_profiles_corrected_r1"
)

CORRECTED_R1_EXPERIMENTS = {
    "A": {
        "name": "Plant1_to_Plant2",
        "source": {
            "plant_dir": "plant1",
            "profile_dir": "source_profile",
            "manifest_plant": "Plant1",
            "manifest_profile": "source",
        },
        "target": {
            "plant_dir": "plant2",
            "profile_dir": "target_profile",
            "manifest_plant": "Plant2",
            "manifest_profile": "target",
        },
    },
    "B": {
        "name": "Plant2_to_Plant1",
        "source": {
            "plant_dir": "plant2",
            "profile_dir": "source_profile",
            "manifest_plant": "Plant2",
            "manifest_profile": "source",
        },
        "target": {
            "plant_dir": "plant1",
            "profile_dir": "target_profile",
            "manifest_plant": "Plant1",
            "manifest_profile": "target",
        },
    },
}

CORRECTED_R1_EXPECTED_ROWS = {
    "source": {"training": 2088, "validation": 523, "test": 653},
    "target": {"training": 521, "validation": 131, "test": 2612},
}

CORRECTED_R1_EXPECTED_SEQUENCES = {
    "source": {"training": 2083, "validation": 518, "test": 648},
    "target": {"training": 516, "validation": 126, "test": 2607},
}

CORRECTED_R1_FEATURE_ORDER = (
    "TIME_SIN",
    "TIME_COS",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
)
CORRECTED_R1_SCALED_FEATURES = (
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
)
CORRECTED_R1_TARGET = "DC_POWER"
CORRECTED_R1_TIMESTAMP = "DATE_TIME"
CORRECTED_R1_SPLITS = ("training", "validation", "test")
CORRECTED_R1_FREQUENCY = "15min"
CORRECTED_R1_WINDOW = 5
CORRECTED_R1_HORIZON = 1

# Phase A records, but does not instantiate, the approved Legacy source model
# contract.  Transfer-learning learning rates are deliberately absent.
R2_SOURCE_MODEL_CONTRACT = {
    "output_activation": "sigmoid",
    "optimizer": "Adam",
    "loss": "MSE",
    "learning_rate": 1e-4,
}
R2_SOURCE_DATA_USAGE = {
    "training": ("training",),
    "validation": ("validation",),
}
R2_SOURCE_SMOKE_CONTRACT = {
    "seed": 1234,
    "device": "/CPU:0",
    "batch_size": 128,
    "epochs": 1,
    "shuffle": False,
}
R2_SOURCE_SMOKE_IDENTITIES = {
    "A": {
        "experiment": "A",
        "source": "Plant1/source_profile",
    },
    "B": {
        "experiment": "B",
        "source": "Plant2/source_profile",
    },
}


def resolve_r2_source_smoke_contract(experiment: str) -> dict:
    """Layer one guarded A/B Source identity on the shared smoke protocol."""

    try:
        identity = R2_SOURCE_SMOKE_IDENTITIES[experiment]
    except KeyError as exc:
        raise ValueError(
            f"Source smoke experiment must be A or B, got {experiment!r}"
        ) from exc
    return {**R2_SOURCE_SMOKE_CONTRACT, **identity}

# Phase B2 formal Source pre-training contract.  The maximum epoch count is
# supported independently by the historical Experiment-A Plant1 params/logs
# and the sealed R2 formal helper; it is therefore a locked value, not a
# default inferred by this runner.
R2_SOURCE_FORMAL_RUN_ID = (
    "R2_A_Plant1_SourcePretrain_Sigmoid_seed1234_run01"
)
R2_SOURCE_FORMAL_RUN_IDS = {
    "A": R2_SOURCE_FORMAL_RUN_ID,
    "B": "R2_B_Plant2_SourcePretrain_Sigmoid_seed1234_run01",
}
R2_SOURCE_FORMAL_OUTPUT_ROOT = Path("reports") / "r2"
R2_SOURCE_FORMAL_OUTPUTS = (
    "run_manifest.json",
    "params.json",
    "environment.json",
    "git_commit.txt",
    "split_manifest.json",
    "scaler_manifest.json",
    "feature_scaler.joblib",
    "target_scaler.joblib",
    "epoch_log.csv",
    "training_log.txt",
    "checkpoints/best_model.hdf5",
    "architecture.png",
    "learning_curve.png",
)
R2_SOURCE_FORMAL_EVALUATION_OUTPUTS = (
    "predictions_normalized.csv",
    "predictions_original_scale.csv",
    "metrics_normalized.json",
    "metrics_original_scale.json",
)
R2_SOURCE_FORMAL_CONTRACT = {
    "experiment": "A",
    "source": "Plant1/source_profile",
    "seed": 1234,
    "device": "/CPU:0",
    "window": 5,
    "horizon": 1,
    "input_shape": (5, 5),
    "total_params": 46681,
    "output_activation": "sigmoid",
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "loss": "mse",
    "batch_size": 128,
    "training_shuffle": True,
    "validation_shuffle": False,
    "maximum_epochs": 500,
    "allow_overwrite": False,
    "callbacks": {
        "ReduceLROnPlateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 4,
            "min_lr": 1e-7,
        },
        "ModelCheckpoint": {
            "monitor": "val_loss",
            "save_best_only": True,
        },
        "EarlyStopping": {
            "monitor": "val_loss",
            "patience": 10,
            "restore_best_weights": True,
        },
        "CSVLogger": {
            "filename": "epoch_log.csv",
        },
    },
}

# Phase D1 keeps one shared Source training contract and selects only the
# immutable experiment identity.  The Experiment-A aliases above remain
# unchanged for compatibility with its sealed run and earlier audit tooling.
R2_SOURCE_FORMAL_IDENTITIES = {
    "A": {
        "experiment": "A",
        "source": "Plant1/source_profile",
        "source_plant": "Plant1",
        "source_profile": "source_profile",
        "run_id": R2_SOURCE_FORMAL_RUN_IDS["A"],
    },
    "B": {
        "experiment": "B",
        "source": "Plant2/source_profile",
        "source_plant": "Plant2",
        "source_profile": "source_profile",
        "run_id": R2_SOURCE_FORMAL_RUN_IDS["B"],
    },
}


def resolve_r2_source_formal_contract(experiment: str) -> dict:
    """Return one guarded A/B identity layered on the shared Source contract."""

    try:
        identity = R2_SOURCE_FORMAL_IDENTITIES[experiment]
    except KeyError as exc:
        raise ValueError(
            f"Formal Source experiment must be A or B, got {experiment!r}"
        ) from exc
    return {
        **R2_SOURCE_FORMAL_CONTRACT,
        **identity,
        "callbacks": {
            name: dict(settings)
            for name, settings in R2_SOURCE_FORMAL_CONTRACT["callbacks"].items()
        },
    }

# Phase C0 locks the approved R2 Corrected Legacy Experiment-A transfer
# learning differences.  These are intentionally not the common settings of
# the later R3 fair-comparison protocol.
R2_EXPERIMENT_A_TL_CONTRACT = {
    "experiment": "A",
    "source": "Plant1/source_profile",
    "target": "Plant2/target_profile",
    "source_run_id": R2_SOURCE_FORMAL_RUN_ID,
    "source_checkpoint": "checkpoints/best_model.hdf5",
    "source_checkpoint_sha256": (
        "8aa299e09c58c2ce77473ba317bdaf9fe"
        "374310a453b61718b605022356ad6aa"
    ),
    "input_shape": (5, 5),
    "total_params": 46681,
    "output_activation": "sigmoid",
    "optimizer": "Adam",
    "learning_rate": 3e-5,
    "loss": "mse",
    "maximum_epochs": 500,
    "seed": 1234,
    "device": "/CPU:0",
    "training_shuffle": True,
    "validation_shuffle": False,
    "transfer_layer_indices": (2, 3, 4, 5),
    "freeze": {
        "batch_size": 64,
        "trainable_params": 121,
        "non_trainable_params": 46560,
    },
    "unfreeze": {
        "batch_size": 128,
        "trainable_params": 46441,
        "non_trainable_params": 240,
    },
    "callbacks": {
        "ReduceLROnPlateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 4,
            "min_lr": 1e-7,
        },
        "ModelCheckpoint": {
            "monitor": "val_loss",
            "save_best_only": True,
        },
        "EarlyStopping": {
            "monitor": "val_loss",
            "patience": 10,
            "restore_best_weights": True,
        },
        "CSVLogger": {
            "filename": "epoch_log.csv",
        },
    },
    "planned_run_ids": {
        "freeze": "R2_A_Plant1_to_Plant2_TL_Freeze_Sigmoid_seed1234_run01",
        "unfreeze": "R2_A_Plant1_to_Plant2_TL_Unfreeze_Sigmoid_seed1234_run01",
    },
    "allow_overwrite": False,
}

# Phase D4 adds only the zero-epoch Experiment-B transfer-learning contract.
# Its smoke and formal execution paths remain deliberately unavailable until a
# later phase grants explicit authorization.
R2_EXPERIMENT_B_TL_CONTRACT = {
    **R2_EXPERIMENT_A_TL_CONTRACT,
    "experiment": "B",
    "source": "Plant2/source_profile",
    "target": "Plant1/target_profile",
    "source_run_id": R2_SOURCE_FORMAL_RUN_IDS["B"],
    "source_checkpoint_sha256": (
        "02b512327becfe28d685f97b1b855d450"
        "ce65877e34218256c78f7bceeea241d"
    ),
    "learning_rate": 1e-5,
    "freeze": {
        "batch_size": 128,
        "trainable_params": 121,
        "non_trainable_params": 46560,
    },
    "unfreeze": {
        "batch_size": 128,
        "trainable_params": 46441,
        "non_trainable_params": 240,
    },
    "callbacks": {
        name: dict(settings)
        for name, settings in R2_EXPERIMENT_A_TL_CONTRACT["callbacks"].items()
    },
    "planned_run_ids": {
        "freeze": "R2_B_Plant2_to_Plant1_TL_Freeze_Sigmoid_seed1234_run01",
        "unfreeze": "R2_B_Plant2_to_Plant1_TL_Unfreeze_Sigmoid_seed1234_run01",
    },
}

R2_TRANSFER_LEARNING_CONTRACTS = {
    "A": R2_EXPERIMENT_A_TL_CONTRACT,
    "B": R2_EXPERIMENT_B_TL_CONTRACT,
}


def resolve_r2_transfer_learning_contract(experiment: str) -> dict:
    """Return an isolated copy of the approved A/B TL contract."""

    try:
        contract = R2_TRANSFER_LEARNING_CONTRACTS[experiment]
    except KeyError as exc:
        raise ValueError(
            f"Transfer-learning experiment must be A or B, got {experiment!r}"
        ) from exc
    return {
        **contract,
        "freeze": dict(contract["freeze"]),
        "unfreeze": dict(contract["unfreeze"]),
        "callbacks": {
            name: dict(settings)
            for name, settings in contract["callbacks"].items()
        },
        "planned_run_ids": dict(contract["planned_run_ids"]),
    }

R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR = {
    "canonical_run_id": "R2_A_Plant2_WithoutTL_Sigmoid_seed1234_run01",
    "evidence_manifest": (
        Path("reports")
        / "Solar Energy Result"
        / "R2"
        / "Experiment_A"
        / "20260814T150304Z_seed1234"
        / "run_manifest.json"
    ),
}

R2_EXPERIMENT_B_WITHOUT_TL_ANCHOR = {
    "canonical_run_id": "R2_B_Plant1_WithoutTL_Sigmoid_seed1234_run01",
    "status": "Deferred",
    "note": (
        "Canonical Experiment-B Plant1 Without-TL artifact is not present; "
        "comparison remains deferred and no alias may be synthesized."
    ),
}

# Phase C2 formal Freeze runner identity and immutable output inventory.  The
# execution entry point is deliberately not exposed by the Phase C2 CLI.
R2_EXPERIMENT_A_TL_FREEZE_FORMAL_RUN_ID = (
    "R2_A_Plant1_to_Plant2_TL_Freeze_Sigmoid_seed1234_run01"
)
R2_EXPERIMENT_A_TL_FREEZE_FORMAL_OUTPUTS = (
    "run_manifest.json",
    "params.json",
    "environment.json",
    "git_commit.txt",
    "split_manifest.json",
    "scaler_manifest.json",
    "feature_scaler.joblib",
    "target_scaler.joblib",
    "source_checkpoint_reference.json",
    "epoch_log.csv",
    "training_log.txt",
    "checkpoints/best_model.hdf5",
    "architecture.png",
    "learning_curve.png",
    "predictions_normalized.csv",
    "predictions_original_scale.csv",
    "metrics_normalized.json",
    "metrics_original_scale.json",
    "prediction_plot_original_scale.png",
    "yy_plot_original_scale.png",
    "residual_plot_original_scale.png",
    "error_histogram_original_scale.png",
)
R2_EXPERIMENT_A_TL_FREEZE_OUTPUT_UNITS = {
    "target": "kW",
    "MAE": "kW",
    "MSE": "kW^2",
    "RMSE": "kW",
    "R2": "unitless",
}

# Phase C5 formal Unfreeze uses the same immutable 22-item evidence schema as
# the sealed Freeze run, but has a distinct non-overwritable run identity.
R2_EXPERIMENT_A_TL_UNFREEZE_FORMAL_RUN_ID = (
    "R2_A_Plant1_to_Plant2_TL_Unfreeze_Sigmoid_seed1234_run01"
)
R2_EXPERIMENT_A_TL_UNFREEZE_FORMAL_OUTPUTS = (
    R2_EXPERIMENT_A_TL_FREEZE_FORMAL_OUTPUTS
)


def corrected_r1_profile_path(
    root: Path | str,
    experiment: str,
    role: str,
) -> Path:
    """Resolve a configured profile below an existing Corrected R1 root."""

    try:
        mapping = CORRECTED_R1_EXPERIMENTS[experiment][role]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Corrected R1 experiment/role: {experiment!r}/{role!r}"
        ) from exc
    return Path(root) / mapping["plant_dir"] / mapping["profile_dir"]
