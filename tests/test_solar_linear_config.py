from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from r2_config.solar_linear import (
    EXPECTED_NON_TRAINABLE_PARAMS,
    EXPECTED_TOTAL_PARAMS,
    EXPECTED_TRAINABLE_PARAMS,
    LEARNING_RATE_POLICY,
    LINEAR_EXPERIMENTS,
    LINEAR_FORMAL_BASE,
    PROTECTED_LEGACY_ROOTS,
    linear_experiment,
)


class SolarLinearConfigTests(unittest.TestCase):
    def test_01_a2_mapping(self):
        spec = linear_experiment("A2")
        self.assertEqual((spec.source_plant, spec.target_plant), ("Plant1", "Plant2"))
        self.assertEqual(spec.source_profile_path.parts[-2:], ("plant1", "source_profile"))
        self.assertEqual(spec.target_profile_path.parts[-2:], ("plant2", "target_profile"))

    def test_02_b_mapping(self):
        spec = linear_experiment("B")
        self.assertEqual((spec.source_plant, spec.target_plant), ("Plant2", "Plant1"))
        self.assertEqual(spec.source_profile_path.parts[-2:], ("plant2", "source_profile"))
        self.assertEqual(spec.target_profile_path.parts[-2:], ("plant1", "target_profile"))

    def test_03_fixed_protocol(self):
        for spec in LINEAR_EXPERIMENTS.values():
            self.assertEqual(spec.activation, "linear")
            self.assertEqual((spec.window, spec.horizon, spec.frequency), (5, 1, "15min"))
            self.assertEqual(spec.seed, 1234)
            self.assertEqual((spec.batch_size, spec.maximum_epochs), (128, 500))
            self.assertEqual((spec.optimizer, spec.loss), ("Adam", "mse"))
            self.assertEqual(spec.transferred_indices, (2, 3, 4, 5))
            self.assertEqual(spec.trainable_indices, (1, 4, 6))
            self.assertEqual(spec.frozen_indices, (2, 3, 5))
            self.assertTrue(spec.batch_normalization_frozen)

    def test_04_config_is_immutable(self):
        with self.assertRaises(TypeError):
            LINEAR_EXPERIMENTS["X"] = linear_experiment("A2")
        with self.assertRaises(FrozenInstanceError):
            linear_experiment("A2").activation = "sigmoid"

    def test_05_output_namespaces_are_new_and_not_created(self):
        for spec in LINEAR_EXPERIMENTS.values():
            resolved = spec.output_root.resolve()
            self.assertTrue(resolved.is_relative_to(LINEAR_FORMAL_BASE.resolve()))
            self.assertFalse(any(resolved.is_relative_to(root.resolve()) for root in PROTECTED_LEGACY_ROOTS))
            self.assertFalse(spec.output_root.exists())

    def test_06_learning_rate_is_not_selected(self):
        self.assertEqual(LEARNING_RATE_POLICY, "required_explicit_before_training")
        for spec in LINEAR_EXPERIMENTS.values():
            self.assertFalse(hasattr(spec, "learning_rate"))

    def test_07_parameter_contract(self):
        self.assertEqual(EXPECTED_TOTAL_PARAMS, 46681)
        self.assertEqual(EXPECTED_TRAINABLE_PARAMS, 29161)
        self.assertEqual(EXPECTED_NON_TRAINABLE_PARAMS, 17520)


if __name__ == "__main__":
    unittest.main()
