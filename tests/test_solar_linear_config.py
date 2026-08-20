from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

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


def _filesystem_snapshot(path: Path):
    root = path.resolve()
    if not root.exists():
        return None
    return tuple(
        sorted(
            (
                item.relative_to(root).as_posix(),
                item.stat().st_size,
                item.stat().st_mtime_ns,
            )
            for item in root.rglob("*")
            if item.is_file()
        )
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

    def test_05_output_namespaces_are_isolated_and_not_modified(self):
        formal_before = _filesystem_snapshot(LINEAR_FORMAL_BASE)
        output_before = {
            experiment_id: _filesystem_snapshot(spec.output_root)
            for experiment_id, spec in LINEAR_EXPERIMENTS.items()
        }
        expected_names = {"A2": "Experiment_A2", "B": "Experiment_B"}
        resolved_outputs = set()

        for experiment_id in expected_names:
            spec = linear_experiment(experiment_id)
            resolved = spec.output_root.resolve()
            resolved_outputs.add(resolved)
            self.assertEqual(
                resolved,
                (LINEAR_FORMAL_BASE / expected_names[experiment_id]).resolve(),
            )
            self.assertTrue(resolved.is_relative_to(LINEAR_FORMAL_BASE.resolve()))
            for root in PROTECTED_LEGACY_ROOTS:
                protected = root.resolve()
                self.assertFalse(resolved.is_relative_to(protected))
                self.assertFalse(protected.is_relative_to(resolved))

        self.assertEqual(len(resolved_outputs), len(expected_names))
        self.assertEqual(_filesystem_snapshot(LINEAR_FORMAL_BASE), formal_before)
        self.assertEqual(
            {
                experiment_id: _filesystem_snapshot(spec.output_root)
                for experiment_id, spec in LINEAR_EXPERIMENTS.items()
            },
            output_before,
        )

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
