from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from r2_config.solar_linear import LINEAR_EXPERIMENTS
from r2_config.solar_linear_formal import (
    FORMAL_CALLBACK_POLICY,
    FORMAL_DEVICE_POLICY,
    FORMAL_PROTOCOL_VERSION,
    HISTORICAL_TARGET_TEST_PREVIOUSLY_REVEALED,
    LEARNING_RATE_CANDIDATES,
    candidate_id,
    candidate_registry,
    formal_path_contract,
    validate_formal_policy,
    validate_new_run_destination,
)


class SolarLinearFormalConfigTests(unittest.TestCase):
    def test_01_lr_candidate_policy_is_locked(self):
        self.assertEqual(LEARNING_RATE_CANDIDATES["source"], (1e-4, 3e-5))
        self.assertEqual(LEARNING_RATE_CANDIDATES["wotl"], (1e-4, 3e-5))
        self.assertEqual(LEARNING_RATE_CANDIDATES["partial_ft"], (1e-5, 3e-5))
        self.assertEqual(sum(len(values) for values in LEARNING_RATE_CANDIDATES.values()), 6)

    def test_02_candidate_ids_are_deterministic(self):
        self.assertEqual(candidate_id("source", 1e-4), "SRC_lr1e-4")
        self.assertEqual(candidate_id("source", 3e-5), "SRC_lr3e-5")
        self.assertEqual(candidate_id("wotl", 1e-4), "WOTL_lr1e-4")
        self.assertEqual(candidate_id("wotl", 3e-5), "WOTL_lr3e-5")
        self.assertEqual(candidate_id("partial_ft", 1e-5), "PFT_lr1e-5")
        self.assertEqual(candidate_id("partial_ft", 3e-5), "PFT_lr3e-5")
        identifiers = [item[0] for values in candidate_registry().values() for item in values]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_03_unregistered_lr_is_rejected(self):
        with self.assertRaises(ValueError):
            candidate_id("source", 5e-5)

    def test_04_callback_policy_is_locked(self):
        policy = FORMAL_CALLBACK_POLICY
        self.assertEqual(policy.maximum_epochs, 500)
        self.assertEqual(
            (policy.reduce_lr.factor, policy.reduce_lr.patience, policy.reduce_lr.min_lr, policy.reduce_lr.cooldown),
            (0.1, 20, 1e-7, 0),
        )
        self.assertEqual(
            (policy.early_stopping.patience, policy.early_stopping.min_delta, policy.early_stopping.restore_best_weights),
            (50, 0.0, False),
        )
        self.assertGreater(policy.early_stopping.patience, policy.reduce_lr.patience)
        self.assertTrue(policy.terminate_on_nan)
        validate_formal_policy()

    def test_05_device_policy_is_cpu_reproducibility_first(self):
        self.assertEqual(FORMAL_DEVICE_POLICY.device, "CPU")
        self.assertEqual(FORMAL_DEVICE_POLICY.cuda_visible_devices, "-1")
        self.assertEqual(FORMAL_DEVICE_POLICY.seed, 1234)
        self.assertTrue(FORMAL_DEVICE_POLICY.deterministic_ops)

    def test_06_a2_b_mapping_and_disclosure(self):
        self.assertEqual(
            (LINEAR_EXPERIMENTS["A2"].source_plant, LINEAR_EXPERIMENTS["A2"].target_plant),
            ("Plant1", "Plant2"),
        )
        self.assertEqual(
            (LINEAR_EXPERIMENTS["B"].source_plant, LINEAR_EXPERIMENTS["B"].target_plant),
            ("Plant2", "Plant1"),
        )
        self.assertTrue(HISTORICAL_TARGET_TEST_PREVIOUSLY_REVEALED["A2"])
        self.assertFalse(HISTORICAL_TARGET_TEST_PREVIOUSLY_REVEALED["B"])

    def test_07_formal_namespace_isolated_and_not_created(self):
        for experiment_id, expected_name in (("A2", "Experiment_A2"), ("B", "Experiment_B")):
            paths = formal_path_contract(experiment_id, "20991231T235959Z_seed1234")
            self.assertEqual(paths.experiment_root.name, expected_name)
            self.assertEqual(paths.run_root.parent, paths.experiment_root)
            self.assertFalse(paths.run_root.exists())
            self.assertEqual(paths.protocol_manifest.name, "protocol_manifest.json")

    def test_08_run_id_and_collision_are_rejected(self):
        with self.assertRaises(ValueError):
            formal_path_contract("A2", "bad-run-id")
        paths = formal_path_contract("A2", "20991231T235959Z_seed1234")
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary)
            collision = replace(paths, run_root=existing)
            with self.assertRaises(ValueError):
                validate_new_run_destination(collision)

    def test_09_policy_objects_are_immutable(self):
        self.assertEqual(FORMAL_PROTOCOL_VERSION, "solar-linear-v1.0")
        with self.assertRaises(FrozenInstanceError):
            FORMAL_DEVICE_POLICY.device = "GPU"
        with self.assertRaises(TypeError):
            LEARNING_RATE_CANDIDATES["source"] = (9e-4,)


if __name__ == "__main__":
    unittest.main()
