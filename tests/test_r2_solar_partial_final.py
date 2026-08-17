"""Zero-Test-access and zero-training contracts for R2.5 Phase E."""

from __future__ import annotations

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from r2_helpers.solar_partial_final import (
    SELECTED_CHECKPOINT_SHA256,
    WITHOUT_TL_BASELINE,
    compare_and_classify,
    run_final_target_test,
    validate_phase_e_preflight,
)


class SolarR25PhaseEContracts(unittest.TestCase):
    def test_01_preflight_without_test_access(self):
        result = validate_phase_e_preflight()
        self.assertEqual(result.selected_checkpoint_sha256, SELECTED_CHECKPOINT_SHA256)

    def test_02_runner_has_no_training_call(self):
        source = inspect.getsource(run_final_target_test)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("fit_train_validation", source)

    def test_03_positive_classification(self):
        partial = {"MAE": 2000.0, "MSE": 8_000_000.0, "RMSE": 3000.0, "R2": 0.8}
        _, result = compare_and_classify(partial, WITHOUT_TL_BASELINE)
        self.assertEqual(result["classification"], "observed_positive")

    def test_04_partial_positive_classification(self):
        partial = {"MAE": 2000.0, "MSE": 8_000_000.0, "RMSE": 3000.0, "R2": 0.6}
        _, result = compare_and_classify(partial, WITHOUT_TL_BASELINE)
        self.assertEqual(result["classification"], "observed_partial_positive")

    def test_05_negative_classification(self):
        partial = {"MAE": 4000.0, "MSE": 20_000_000.0, "RMSE": 4500.0, "R2": 0.4}
        _, result = compare_and_classify(partial, WITHOUT_TL_BASELINE)
        self.assertEqual(result["classification"], "observed_negative")

    def test_06_mixed_classification(self):
        partial = {"MAE": 2000.0, "MSE": 20_000_000.0, "RMSE": 4500.0, "R2": 0.4}
        _, result = compare_and_classify(partial, WITHOUT_TL_BASELINE)
        self.assertEqual(result["classification"], "observed_mixed")

    def test_07_delta_direction(self):
        partial = {"MAE": 2800.0, "MSE": 13_000_000.0, "RMSE": 3600.0, "R2": 0.7}
        comparison, _ = compare_and_classify(partial, WITHOUT_TL_BASELINE)
        self.assertLess(comparison["delta_rmse"], 0)
        self.assertGreater(comparison["improvement_percent_rmse"], 0)
        self.assertGreater(comparison["delta_r2"], 0)


if __name__ == "__main__":
    unittest.main()
