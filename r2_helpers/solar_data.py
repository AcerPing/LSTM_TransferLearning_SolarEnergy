"""Read-only corrected-R1 data contract and aligned sequence construction."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from r2_config.solar_r2 import (
    EXPECTED_ROW_COUNTS,
    FEATURE_COLUMNS,
    FREQUENCY,
    HORIZON,
    SCALED_FEATURE_COLUMNS,
    SPLIT_NAMES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    WINDOW,
)


REQUIRED_PROFILE_FILES = (
    "normalized_scale_training.csv",
    "normalized_scale_validation.csv",
    "normalized_scale_test.csv",
    "original_scale_training.csv",
    "original_scale_validation.csv",
    "original_scale_test.csv",
    "feature_scaler.joblib",
    "target_scaler.joblib",
    "split_manifest.json",
    "scaler_manifest.json",
    "file_checksums.csv",
)

UNSCALED_CYCLIC_FEATURE_COLUMNS = ("TIME_SIN", "TIME_COS")


class DataContractError(AssertionError):
    """Raised when corrected R1 violates the fixed R2 data contract."""


@dataclass(frozen=True)
class ProfileMetadata:
    profile_dir: Path
    split_manifest: Mapping[str, Any]
    scaler_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class ScalerBundle:
    feature_scaler: Any
    target_scaler: Any


@dataclass(frozen=True)
class SplitData:
    name: str
    timestamps: np.ndarray
    X_normalized: np.ndarray
    y_normalized: np.ndarray
    y_original: np.ndarray
    normalized_frame: pd.DataFrame
    original_frame: pd.DataFrame

    @property
    def row_count(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class SequenceData:
    X_seq: np.ndarray
    y_seq: np.ndarray
    sample_index: np.ndarray
    target_row_index: np.ndarray
    target_timestamp: np.ndarray

    @property
    def count(self) -> int:
        return len(self.sample_index)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataContractError(message)


def _load_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"Required JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Cannot parse JSON file {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_required_files(profile_dir: Path | str) -> tuple[Path, ...]:
    """Require every corrected-R1 contract file; never create missing files."""

    directory = Path(profile_dir)
    _require(directory.is_dir(), f"Profile directory not found: {directory}")
    paths = tuple(directory / name for name in REQUIRED_PROFILE_FILES)
    missing = [path.name for path in paths if not path.is_file()]
    _require(not missing, f"Missing required profile files in {directory}: {missing}")
    return paths


def load_profile_metadata(profile_dir: Path | str) -> ProfileMetadata:
    """Load the two manifests without changing profile contents."""

    directory = Path(profile_dir)
    validate_required_files(directory)
    return ProfileMetadata(
        profile_dir=directory,
        split_manifest=_load_json(directory / "split_manifest.json"),
        scaler_manifest=_load_json(directory / "scaler_manifest.json"),
    )


def validate_checksums(profile_dir: Path | str) -> Mapping[str, str]:
    """Validate every SHA-256 entry recorded by file_checksums.csv."""

    directory = Path(profile_dir)
    validate_required_files(directory)
    checksum_path = directory / "file_checksums.csv"
    try:
        with checksum_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DataContractError(f"Cannot read checksum manifest: {checksum_path}") from exc

    _require(rows, f"Checksum manifest is empty: {checksum_path}")
    _require(
        set(rows[0]) == {"file_name", "sha256"},
        f"Unexpected checksum columns in {checksum_path}: {list(rows[0])}",
    )

    results: dict[str, str] = {}
    for row in rows:
        name = row["file_name"]
        expected = row["sha256"].lower()
        _require(name not in results, f"Duplicate checksum entry: {name}")
        file_path = directory / name
        _require(file_path.is_file(), f"Checksummed file not found: {file_path}")
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        _require(
            actual == expected,
            f"SHA-256 mismatch for {file_path}: expected {expected}, got {actual}",
        )
        results[name] = actual

    files_requiring_checksums = set(REQUIRED_PROFILE_FILES) - {"file_checksums.csv"}
    _require(
        files_requiring_checksums.issubset(results),
        "Checksum manifest does not cover every required artifact: "
        f"{sorted(files_requiring_checksums - set(results))}",
    )
    return results


def validate_manifest(
    metadata: ProfileMetadata,
    *,
    expected_plant: str,
    expected_profile: str,
    expected_profile_dir_name: str,
) -> None:
    """Validate identity, schema, frequency, and fixed split row counts."""

    split = metadata.split_manifest
    scaler = metadata.scaler_manifest
    expected_rows = EXPECTED_ROW_COUNTS[expected_profile_dir_name]

    _require(split.get("plant") == expected_plant, "Split manifest plant mismatch")
    _require(split.get("profile") == expected_profile, "Split manifest profile mismatch")
    _require(split.get("frequency") == FREQUENCY, "Split manifest frequency mismatch")
    _require(
        tuple(split.get("feature_columns", ())) == FEATURE_COLUMNS,
        "Split manifest feature order mismatch",
    )
    _require(split.get("target_column") == TARGET_COLUMN, "Target column mismatch")
    _require(split.get("date_time_index_saved") is True, "DATE_TIME index is not saved")
    _require(
        split.get("date_time_index_label") == TIMESTAMP_COLUMN,
        "Timestamp column mismatch",
    )

    manifest_splits = split.get("splits")
    _require(isinstance(manifest_splits, dict), "Split manifest has no splits object")
    for split_name in SPLIT_NAMES:
        entry = manifest_splits.get(split_name)
        _require(isinstance(entry, dict), f"Missing manifest split: {split_name}")
        _require(
            entry.get("rows") == expected_rows[split_name],
            f"Unexpected row count for {split_name}: {entry.get('rows')}",
        )
        _require(entry.get("nan_total_original") == 0, f"Original NaNs in {split_name}")
        _require(entry.get("nan_total_normalized") == 0, f"Normalized NaNs in {split_name}")
        _require(
            entry.get("normalized_scale_file") == f"normalized_scale_{split_name}.csv",
            f"Unexpected normalized filename for {split_name}",
        )
        _require(
            entry.get("original_scale_file") == f"original_scale_{split_name}.csv",
            f"Unexpected original filename for {split_name}",
        )

    _require(scaler.get("plant") == expected_plant, "Scaler manifest plant mismatch")
    _require(scaler.get("profile") == expected_profile, "Scaler manifest profile mismatch")


def _sample_count(value: Any, label: str) -> int:
    array = np.asarray(value)
    _require(array.size == 1, f"{label} n_samples_seen_ must be scalar")
    return int(array.reshape(-1)[0])


def validate_scalers(metadata: ProfileMetadata) -> ScalerBundle:
    """Load existing scalers and require a training-only fit contract."""

    manifest = metadata.scaler_manifest
    profile_name = metadata.profile_dir.name
    expected_training_rows = EXPECTED_ROW_COUNTS[profile_name]["training"]

    _require(manifest.get("feature_scaler_fit_split") == "training", "Feature scaler was not fit on training only")
    _require(manifest.get("target_scaler_fit_split") == "training", "Target scaler was not fit on training only")
    _require(
        tuple(manifest.get("feature_scaler_columns", ())) == SCALED_FEATURE_COLUMNS,
        "Feature scaler columns mismatch",
    )
    _require(manifest.get("target_scaler_column") == TARGET_COLUMN, "Target scaler column mismatch")
    _require(manifest.get("time_columns_scaled") is False, "TIME_SIN/TIME_COS must remain unscaled")
    _require(manifest.get("clipping_applied") is False, "Corrected R1 must not apply clipping")
    _require(manifest.get("feature_scaler_n_samples_seen") == expected_training_rows, "Feature scaler manifest sample count mismatch")
    _require(manifest.get("target_scaler_n_samples_seen") == expected_training_rows, "Target scaler manifest sample count mismatch")

    feature_path = metadata.profile_dir / manifest.get("feature_scaler_file", "")
    target_path = metadata.profile_dir / manifest.get("target_scaler_file", "")
    _require(feature_path.is_file(), f"Feature scaler not found: {feature_path}")
    _require(target_path.is_file(), f"Target scaler not found: {target_path}")
    try:
        feature_scaler = joblib.load(feature_path)
        target_scaler = joblib.load(target_path)
    except Exception as exc:
        raise DataContractError(f"Cannot load scaler artifact: {exc}") from exc

    _require(_sample_count(feature_scaler.n_samples_seen_, "Feature scaler") == expected_training_rows, "Feature scaler object sample count mismatch")
    _require(_sample_count(target_scaler.n_samples_seen_, "Target scaler") == expected_training_rows, "Target scaler object sample count mismatch")
    _require(int(feature_scaler.n_features_in_) == len(SCALED_FEATURE_COLUMNS), "Feature scaler dimension mismatch")
    _require(int(target_scaler.n_features_in_) == 1, "Target scaler dimension mismatch")
    _require(tuple(feature_scaler.feature_range) == (0, 1), "Feature scaler range mismatch")
    _require(tuple(target_scaler.feature_range) == (0, 1), "Target scaler range mismatch")
    _require(feature_scaler.clip is False, "Feature scaler clipping must be disabled")
    _require(target_scaler.clip is False, "Target scaler clipping must be disabled")
    _require(np.allclose(feature_scaler.data_min_, manifest["feature_data_min"]), "Feature scaler minima mismatch")
    _require(np.allclose(feature_scaler.data_max_, manifest["feature_data_max"]), "Feature scaler maxima mismatch")
    _require(np.allclose(target_scaler.data_min_, manifest["target_data_min"]), "Target scaler minima mismatch")
    _require(np.allclose(target_scaler.data_max_, manifest["target_data_max"]), "Target scaler maxima mismatch")
    return ScalerBundle(feature_scaler=feature_scaler, target_scaler=target_scaler)


def validate_normalized_original_alignment(
    normalized: pd.DataFrame,
    original: pd.DataFrame,
    *,
    split_name: str,
) -> pd.DatetimeIndex:
    """Require exact schema/order and identical timestamps across both scales."""

    expected_columns = (TIMESTAMP_COLUMN,) + FEATURE_COLUMNS + (TARGET_COLUMN,)
    _require(tuple(normalized.columns) == expected_columns, f"Normalized columns/order mismatch in {split_name}")
    _require(tuple(original.columns) == expected_columns, f"Original columns/order mismatch in {split_name}")
    _require(len(normalized) == len(original), f"Scale row-count mismatch in {split_name}")
    _require(not normalized.isna().any().any(), f"NaN in normalized {split_name}")
    _require(not original.isna().any().any(), f"NaN in original {split_name}")

    try:
        normalized_time = pd.to_datetime(normalized[TIMESTAMP_COLUMN], errors="raise")
        original_time = pd.to_datetime(original[TIMESTAMP_COLUMN], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataContractError(f"Invalid timestamp in {split_name}: {exc}") from exc
    _require(normalized_time.equals(original_time), f"Timestamp mismatch between scales in {split_name}")
    _require(normalized_time.is_monotonic_increasing, f"Timestamps not increasing in {split_name}")
    _require(not normalized_time.duplicated().any(), f"Duplicate timestamps in {split_name}")
    return pd.DatetimeIndex(normalized_time)


def load_split(metadata: ProfileMetadata, split_name: str) -> SplitData:
    """Read and validate one fixed split from both normalized and original CSVs."""

    _require(split_name in SPLIT_NAMES, f"Unknown split: {split_name}")
    entry = metadata.split_manifest["splits"][split_name]
    normalized_path = metadata.profile_dir / entry["normalized_scale_file"]
    original_path = metadata.profile_dir / entry["original_scale_file"]
    try:
        normalized = pd.read_csv(normalized_path)
        original = pd.read_csv(original_path)
    except Exception as exc:
        raise DataContractError(f"Cannot read {split_name} CSV files: {exc}") from exc

    timestamps = validate_normalized_original_alignment(
        normalized, original, split_name=split_name
    )
    expected_rows = EXPECTED_ROW_COUNTS[metadata.profile_dir.name][split_name]
    _require(len(normalized) == expected_rows, f"Actual row count mismatch in {split_name}")
    _require(len(normalized) == entry["rows"], f"CSV/manifest row count mismatch in {split_name}")
    _require(
        timestamps[0] == pd.Timestamp(entry["start_time"]),
        f"Start timestamp mismatch in {split_name}",
    )
    _require(
        timestamps[-1] == pd.Timestamp(entry["end_time"]),
        f"End timestamp mismatch in {split_name}",
    )
    if len(timestamps) > 1:
        expected_delta = pd.Timedelta(FREQUENCY)
        actual_deltas = timestamps[1:] - timestamps[:-1]
        _require(
            bool((actual_deltas == expected_delta).all()),
            f"Timestamp frequency mismatch in {split_name}",
        )

    X = normalized.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    y_normalized = normalized.loc[:, TARGET_COLUMN].to_numpy(dtype=np.float64, copy=True)
    y_original = original.loc[:, TARGET_COLUMN].to_numpy(dtype=np.float64, copy=True)
    _require(np.isfinite(X).all(), f"Non-finite feature value in {split_name}")
    _require(np.isfinite(y_normalized).all(), f"Non-finite normalized target in {split_name}")
    _require(np.isfinite(y_original).all(), f"Non-finite original target in {split_name}")
    return SplitData(
        name=split_name,
        timestamps=timestamps.to_numpy(copy=True),
        X_normalized=X,
        y_normalized=y_normalized,
        y_original=y_original,
        normalized_frame=normalized,
        original_frame=original,
    )


def validate_target_round_trip(
    split_data: SplitData,
    target_scaler: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-6,
) -> None:
    """Require stored normalized targets to invert to stored original targets."""

    inverted = target_scaler.inverse_transform(
        split_data.y_normalized.reshape(-1, 1)
    ).reshape(-1)
    _require(
        np.allclose(inverted, split_data.y_original, rtol=rtol, atol=atol),
        f"Target scaler round-trip mismatch in {split_data.name}",
    )


def validate_feature_transform_consistency(
    split_data: SplitData,
    feature_scaler: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> None:
    """Require stored features to match the corrected-R1 scaling contract."""

    original_environmental = split_data.original_frame.loc[
        :, SCALED_FEATURE_COLUMNS
    ].astype(np.float64, copy=True)
    normalized_environmental = split_data.normalized_frame.loc[
        :, SCALED_FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64, copy=True)
    try:
        transformed_environmental = feature_scaler.transform(
            original_environmental
        )
    except Exception as exc:
        raise DataContractError(
            f"Cannot transform original features in {split_data.name}: {exc}"
        ) from exc
    _require(
        np.allclose(
            transformed_environmental,
            normalized_environmental,
            rtol=rtol,
            atol=atol,
        ),
        f"Environmental feature transform mismatch in {split_data.name}",
    )

    original_cyclic = split_data.original_frame.loc[
        :, UNSCALED_CYCLIC_FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64, copy=True)
    normalized_cyclic = split_data.normalized_frame.loc[
        :, UNSCALED_CYCLIC_FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64, copy=True)
    _require(
        np.allclose(
            original_cyclic,
            normalized_cyclic,
            rtol=rtol,
            atol=atol,
        ),
        f"TIME_SIN/TIME_COS changed between scales in {split_data.name}",
    )


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Sequence[Any],
    *,
    window: int = WINDOW,
    horizon: int = HORIZON,
) -> SequenceData:
    """Build aligned sequences: X[i:i+window] -> y[i+window+horizon-1]."""

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    timestamp_array = np.asarray(timestamps)
    _require(X_array.ndim == 2, f"X must be 2-D, got shape {X_array.shape}")
    _require(y_array.ndim == 1, f"y must be 1-D, got shape {y_array.shape}")
    _require(window > 0, "window must be positive")
    _require(horizon > 0, "horizon must be positive")
    _require(len(X_array) == len(y_array) == len(timestamp_array), "X/y/timestamp lengths differ")

    n_sequences = len(X_array) - window - horizon + 1
    _require(n_sequences > 0, "Not enough rows to build one sequence")
    sample_index = np.arange(n_sequences, dtype=np.int64)
    target_row_index = sample_index + window + horizon - 1
    X_seq = np.stack(
        [X_array[i : i + window] for i in sample_index], axis=0
    )
    y_seq = y_array[target_row_index].copy()
    target_timestamp = timestamp_array[target_row_index].copy()

    _require(len(X_seq) == n_sequences, "Sequence builder produced an extra window")
    _require(target_row_index[0] == window + horizon - 1, "First target index is misaligned")
    _require(target_row_index[-1] == len(X_array) - 1, "Last target index is misaligned")
    return SequenceData(
        X_seq=X_seq,
        y_seq=y_seq,
        sample_index=sample_index,
        target_row_index=target_row_index,
        target_timestamp=target_timestamp,
    )
