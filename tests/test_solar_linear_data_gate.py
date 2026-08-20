from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from r2_config.solar_linear import linear_experiment
from r2_helpers import solar_data


FORBIDDEN_TEST_FILES = {
    "normalized_scale_test.csv",
    "original_scale_test.csv",
}


class SolarLinearDataGateTests(unittest.TestCase):
    def _load_with_open_spy(self, experiment_id: str, role: str):
        spec = linear_experiment(experiment_id)
        profile_path = (
            spec.source_profile_path if role == "source" else spec.target_profile_path
        )
        expected_plant = spec.source_plant if role == "source" else spec.target_plant
        expected_profile = role
        expected_profile_dir = f"{role}_profile"
        opened: list[Path] = []

        original_path_open = Path.open
        original_read_csv = solar_data.pd.read_csv
        original_joblib_load = solar_data.joblib.load

        def path_open(path, *args, **kwargs):
            opened.append(Path(path).resolve())
            return original_path_open(path, *args, **kwargs)

        def read_csv(path, *args, **kwargs):
            opened.append(Path(path).resolve())
            return original_read_csv(path, *args, **kwargs)

        def joblib_load(path, *args, **kwargs):
            opened.append(Path(path).resolve())
            return original_joblib_load(path, *args, **kwargs)

        with patch.object(Path, "open", path_open), patch.object(
            solar_data.pd, "read_csv", read_csv
        ), patch.object(solar_data.joblib, "load", joblib_load):
            profile = solar_data.load_training_validation_profile(
                profile_path,
                expected_plant=expected_plant,
                expected_profile=expected_profile,
                expected_profile_dir_name=expected_profile_dir,
            )
        return profile, tuple(opened)

    def test_01_a2_training_validation_only(self):
        for role in ("source", "target"):
            with self.subTest(role=role):
                profile, opened = self._load_with_open_spy("A2", role)
                self._assert_gate(role, profile, opened)

    def test_02_b_training_validation_only(self):
        for role in ("source", "target"):
            with self.subTest(role=role):
                profile, opened = self._load_with_open_spy("B", role)
                self._assert_gate(role, profile, opened)

    def _assert_gate(self, role, profile, opened):
        opened_names = {path.name for path in opened}
        self.assertTrue(FORBIDDEN_TEST_FILES.isdisjoint(opened_names))
        self.assertIn("normalized_scale_training.csv", opened_names)
        self.assertIn("original_scale_training.csv", opened_names)
        self.assertIn("normalized_scale_validation.csv", opened_names)
        self.assertIn("original_scale_validation.csv", opened_names)
        self.assertIn("feature_scaler.joblib", opened_names)
        self.assertIn("target_scaler.joblib", opened_names)
        self.assertIn("split_manifest.json", opened_names)
        self.assertIn("scaler_manifest.json", opened_names)
        self.assertIn("file_checksums.csv", opened_names)

        expected_rows = (2088, 523) if role == "source" else (521, 131)
        expected_sequences = (2083, 518) if role == "source" else (516, 126)
        self.assertEqual(
            (profile.training.split_data.row_count, profile.validation.split_data.row_count),
            expected_rows,
        )
        self.assertEqual(
            (profile.training.sequences.count, profile.validation.sequences.count),
            expected_sequences,
        )
        self.assertEqual(len(profile.checksums), 8)
        self.assertTrue(FORBIDDEN_TEST_FILES.isdisjoint(profile.checksums))

    def test_03_scoped_checksum_rejects_invalid_scope(self):
        path = linear_experiment("A2").source_profile_path
        with self.assertRaises(solar_data.DataContractError):
            solar_data.validate_checksums_for_splits(path, splits=("training", "training"))
        with self.assertRaises(solar_data.DataContractError):
            solar_data.validate_checksums_for_splits(path, splits=("unknown",))


if __name__ == "__main__":
    unittest.main()
