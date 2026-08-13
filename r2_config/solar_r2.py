"""Fixed data contract for the corrected R1 Solar R2 experiments.

This module contains configuration only.  It deliberately contains no model,
training, output-directory, or data-mutation logic.
"""

from pathlib import Path


CORRECTED_R1_ROOT = Path(
    r"D:\HoChePing\北科大_碩班_AI學程\期刊研究\使用LSTM模型預測福壽雞隻重量"
    r"\Code程式碼\PersonalNote\DataSet_For_TransferLearning"
    r"\★★ Solar Power Generation Data ★★\preprocessed_profiles_corrected_r1"
)

WINDOW = 5
HORIZON = 1
FREQUENCY = "15min"

FEATURE_COLUMNS = (
    "TIME_SIN",
    "TIME_COS",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
)
SCALED_FEATURE_COLUMNS = (
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
)
TARGET_COLUMN = "DC_POWER"
TIMESTAMP_COLUMN = "DATE_TIME"
SPLIT_NAMES = ("training", "validation", "test")

EXPECTED_ROW_COUNTS = {
    "source_profile": {
        "training": 2088,
        "validation": 523,
        "test": 653,
    },
    "target_profile": {
        "training": 521,
        "validation": 131,
        "test": 2612,
    },
}

EXPECTED_SEQUENCE_COUNTS = {
    "source_profile": {
        "training": 2083,
        "validation": 518,
        "test": 648,
    },
    "target_profile": {
        "training": 516,
        "validation": 126,
        "test": 2607,
    },
}

EXPERIMENTS = {
    "A": {
        "name": "plant1_to_plant2",
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
        "name": "plant2_to_plant1",
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


def profile_path(experiment: str, role: str) -> Path:
    """Return the configured corrected-R1 profile path without touching it."""

    try:
        mapping = EXPERIMENTS[experiment][role]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment/role combination: {experiment!r}/{role!r}"
        ) from exc
    return CORRECTED_R1_ROOT / mapping["plant_dir"] / mapping["profile_dir"]

