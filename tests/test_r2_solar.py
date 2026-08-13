"""Read-only contract tests for corrected-R1 Solar R2 data."""

import unittest

import numpy as np

from r2_config.solar_r2 import (
    EXPECTED_ROW_COUNTS,
    EXPECTED_SEQUENCE_COUNTS,
    EXPERIMENTS,
    FEATURE_COLUMNS,
    HORIZON,
    WINDOW,
    profile_path,
)
from r2_helpers.solar_data import (
    REQUIRED_PROFILE_FILES,
    UNSCALED_CYCLIC_FEATURE_COLUMNS,
    build_sequences,
    load_profile_metadata,
    load_split,
    validate_checksums,
    validate_feature_transform_consistency,
    validate_manifest,
    validate_required_files,
    validate_scalers,
    validate_target_round_trip,
)


class SolarR2DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = {}
        for experiment_name, experiment in EXPERIMENTS.items():
            for role in ("source", "target"):
                mapping = experiment[role]
                path = profile_path(experiment_name, role)
                key = (mapping["plant_dir"], mapping["profile_dir"])
                if key not in cls.profiles:
                    metadata = load_profile_metadata(path)
                    splits = {
                        split_name: load_split(metadata, split_name)
                        for split_name in ("training", "validation", "test")
                    }
                    cls.profiles[key] = {
                        "path": path,
                        "metadata": metadata,
                        "splits": splits,
                        "mapping": mapping,
                    }

    def test_01_experiment_a_mapping(self):
        self.assertEqual(EXPERIMENTS["A"]["source"]["plant_dir"], "plant1")
        self.assertEqual(EXPERIMENTS["A"]["source"]["profile_dir"], "source_profile")
        self.assertEqual(EXPERIMENTS["A"]["target"]["plant_dir"], "plant2")
        self.assertEqual(EXPERIMENTS["A"]["target"]["profile_dir"], "target_profile")

    def test_02_experiment_b_mapping(self):
        self.assertEqual(EXPERIMENTS["B"]["source"]["plant_dir"], "plant2")
        self.assertEqual(EXPERIMENTS["B"]["source"]["profile_dir"], "source_profile")
        self.assertEqual(EXPERIMENTS["B"]["target"]["plant_dir"], "plant1")
        self.assertEqual(EXPERIMENTS["B"]["target"]["profile_dir"], "target_profile")

    def test_03_required_files(self):
        for profile in self.profiles.values():
            paths = validate_required_files(profile["path"])
            self.assertEqual({path.name for path in paths}, set(REQUIRED_PROFILE_FILES))

    def test_04_checksums(self):
        for profile in self.profiles.values():
            checksums = validate_checksums(profile["path"])
            self.assertTrue(checksums)

    def test_05_manifests(self):
        for profile in self.profiles.values():
            mapping = profile["mapping"]
            validate_manifest(
                profile["metadata"],
                expected_plant=mapping["manifest_plant"],
                expected_profile=mapping["manifest_profile"],
                expected_profile_dir_name=mapping["profile_dir"],
            )

    def test_06_training_only_scaler_fit(self):
        for profile in self.profiles.values():
            scalers = validate_scalers(profile["metadata"])
            expected = EXPECTED_ROW_COUNTS[profile["mapping"]["profile_dir"]]["training"]
            self.assertEqual(int(scalers.feature_scaler.n_samples_seen_), expected)
            self.assertEqual(int(scalers.target_scaler.n_samples_seen_), expected)

    def test_07_feature_order(self):
        for profile in self.profiles.values():
            manifest_columns = tuple(profile["metadata"].split_manifest["feature_columns"])
            self.assertEqual(manifest_columns, FEATURE_COLUMNS)
            for split_data in profile["splits"].values():
                actual = tuple(split_data.normalized_frame.columns[1:-1])
                self.assertEqual(actual, FEATURE_COLUMNS)

    def test_08_timestamps_alignment(self):
        for profile in self.profiles.values():
            previous_end = None
            for split_name in ("training", "validation", "test"):
                split_data = profile["splits"][split_name]
                normalized = split_data.normalized_frame["DATE_TIME"].tolist()
                original = split_data.original_frame["DATE_TIME"].tolist()
                self.assertEqual(normalized, original)
                self.assertEqual(len(normalized), len(set(normalized)))
                if previous_end is not None:
                    delta = split_data.timestamps[0] - previous_end
                    self.assertEqual(delta, np.timedelta64(15, "m"))
                previous_end = split_data.timestamps[-1]

    def test_09_scaler_round_trip(self):
        for profile in self.profiles.values():
            scalers = validate_scalers(profile["metadata"])
            for split_data in profile["splits"].values():
                validate_target_round_trip(split_data, scalers.target_scaler)

    def test_10_environmental_feature_transform_consistency(self):
        for profile in self.profiles.values():
            scalers = validate_scalers(profile["metadata"])
            for split_data in profile["splits"].values():
                validate_feature_transform_consistency(
                    split_data, scalers.feature_scaler
                )

    def test_11_time_sin_time_cos_unchanged(self):
        for profile in self.profiles.values():
            for split_data in profile["splits"].values():
                original = split_data.original_frame.loc[
                    :, UNSCALED_CYCLIC_FEATURE_COLUMNS
                ].to_numpy(dtype=np.float64)
                normalized = split_data.normalized_frame.loc[
                    :, UNSCALED_CYCLIC_FEATURE_COLUMNS
                ].to_numpy(dtype=np.float64)
                np.testing.assert_allclose(
                    normalized, original, rtol=1e-10, atol=1e-10
                )

    def test_12_synthetic_sequence_semantics(self):
        N = 12
        X = np.arange(N * 2).reshape(N, 2)
        y = np.arange(N)
        timestamps = np.arange(N)
        sequences = build_sequences(X, y, timestamps, window=WINDOW, horizon=HORIZON)
        np.testing.assert_array_equal(sequences.X_seq[0], X[0:5])
        self.assertEqual(sequences.y_seq[0], y[5])
        np.testing.assert_array_equal(sequences.X_seq[-1], X[N - 6 : N - 1])
        self.assertEqual(sequences.y_seq[-1], y[N - 1])

    def test_13_real_sequence_counts(self):
        for profile in self.profiles.values():
            profile_name = profile["mapping"]["profile_dir"]
            for split_name, split_data in profile["splits"].items():
                sequences = build_sequences(
                    split_data.X_normalized,
                    split_data.y_normalized,
                    split_data.timestamps,
                )
                self.assertEqual(
                    sequences.count,
                    EXPECTED_SEQUENCE_COUNTS[profile_name][split_name],
                )

    def test_14_first_and_last_target_indices(self):
        for profile in self.profiles.values():
            for split_data in profile["splits"].values():
                sequences = build_sequences(
                    split_data.X_normalized,
                    split_data.y_normalized,
                    split_data.timestamps,
                )
                self.assertEqual(sequences.target_row_index[0], 5)
                self.assertEqual(sequences.target_row_index[-1], split_data.row_count - 1)
                self.assertEqual(sequences.target_timestamp[0], split_data.timestamps[5])
                self.assertEqual(sequences.target_timestamp[-1], split_data.timestamps[-1])

    def test_15_no_extra_window(self):
        for profile in self.profiles.values():
            for split_data in profile["splits"].values():
                sequences = build_sequences(
                    split_data.X_normalized,
                    split_data.y_normalized,
                    split_data.timestamps,
                )
                expected = split_data.row_count - WINDOW - HORIZON + 1
                self.assertEqual(sequences.count, expected)
                self.assertEqual(sequences.sample_index[-1], expected - 1)
                self.assertLess(sequences.target_row_index[-1], split_data.row_count)


if __name__ == "__main__":
    unittest.main()
