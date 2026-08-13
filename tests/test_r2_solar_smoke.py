"""One-process, eight-epoch integration gate for Solar R2.3.

Running this module intentionally performs exactly one disposable smoke run.
It is not part of the zero-epoch R2.1/R2.2 regression commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


class SolarR2SmokeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.root = (
            Path(tempfile.gettempdir())
            / "solar_r2_smoke"
            / f"unittest_{uuid.uuid4().hex}"
        )
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "1234"
        command = [
            sys.executable,
            "-B",
            "r2_solar.py",
            "smoke",
            "--device",
            "cpu",
            "--seed",
            "1234",
            "--keep-smoke",
            "--smoke-root",
            str(cls.root),
        ]
        cls.process = subprocess.run(
            command,
            cwd=cls.repo,
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if cls.process.returncode != 0:
            raise AssertionError(
                "R2.3 integration subprocess failed\nSTDOUT:\n"
                + cls.process.stdout
                + "\nSTDERR:\n"
                + cls.process.stderr
            )
        cls.manifest_path = cls.root / "smoke_manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "root", None) and cls.root.exists():
            from r2_helpers.solar_smoke import finalize_smoke_root

            finalize_smoke_root(cls.root, success=True, keep_smoke=False)

    def _event_order(self, name):
        return next(
            event["order"]
            for event in self.manifest["events"]
            if event["name"] == name
        )

    def test_01_process_device_bootstrap(self):
        self.assertEqual(self.manifest["device"], "cpu")
        self.assertEqual(self.manifest["visible_gpu_count"], 0)

    def test_02_pythonhashseed_gate(self):
        self.assertEqual(self.manifest["pythonhashseed"], "1234")

    def test_03_smoke_root_safety(self):
        resolved_root = self.root.resolve()
        resolved_temp = Path(tempfile.gettempdir()).resolve()
        resolved_repo = self.repo.resolve()

        self.assertTrue(resolved_root.is_relative_to(resolved_temp))
        self.assertFalse(resolved_root.is_relative_to(resolved_repo))

    def test_04_experiment_a_only_mapping(self):
        self.assertEqual(self.manifest["experiment"], "A")
        self.assertEqual(self.manifest["mapping"]["source"], "Plant1/source_profile")
        self.assertEqual(self.manifest["mapping"]["target"], "Plant2/target_profile")

    def test_05_subset_counts(self):
        for role in ("source", "target"):
            self.assertEqual(self.manifest["subsets"][role]["training"]["count"], 256)
            self.assertEqual(self.manifest["subsets"][role]["validation"]["count"], 64)
            self.assertEqual(self.manifest["subsets"][role]["test"]["count"], 32)

    def test_06_sequence_shapes(self):
        for role in ("source", "target"):
            self.assertEqual(self.manifest["subsets"][role]["training"]["shape"], [256, 5, 5])
            self.assertEqual(self.manifest["subsets"][role]["validation"]["shape"], [64, 5, 5])
            self.assertEqual(self.manifest["subsets"][role]["test"]["shape"], [32, 5, 5])

    def test_07_source_real_two_epoch_smoke(self):
        source = self.manifest["lifecycles"]["source_pretrain"]
        self.assertEqual(source["epochs_completed"], 2)
        self.assertTrue(source["val_loss_received"])

    def test_08_source_checkpoint_reload(self):
        self.assertTrue(self.manifest["lifecycles"]["source_pretrain"]["checkpoint_reloaded"])
        self.assertIn("source_pretrain", self.manifest["checkpoints"])

    def test_09_without_tl_real_two_epoch_smoke(self):
        method = self.manifest["lifecycles"]["without_tl"]
        self.assertEqual(method["epochs_completed"], 2)
        self.assertEqual(method["status"], "PASS")

    def test_10_without_tl_never_reads_source_before_fit(self):
        self.assertEqual(
            self.manifest["lifecycles"]["without_tl"]["source_checkpoint_reads_before_training"],
            0,
        )

    def test_11_tl_freeze_real_two_epoch_smoke(self):
        self.assertEqual(self.manifest["lifecycles"]["tl_freeze"]["epochs_completed"], 2)

    def test_12_frozen_weights_unchanged(self):
        freeze = self.manifest["lifecycles"]["tl_freeze"]
        self.assertTrue(freeze["transferable_weights_unchanged"])
        self.assertEqual(freeze["checked_layer_indices"], [2, 3, 4, 5])

    def test_13_full_finetune_real_two_epoch_smoke(self):
        self.assertEqual(self.manifest["lifecycles"]["tl_full_finetune"]["epochs_completed"], 2)

    def test_14_full_finetune_transferred_weight_changed(self):
        self.assertTrue(
            self.manifest["lifecycles"]["tl_full_finetune"]["transferred_trainable_weight_changed"]
        )

    def test_15_optimizer_iterations_increased(self):
        full = self.manifest["lifecycles"]["tl_full_finetune"]
        self.assertGreater(full["optimizer_iterations_after"], full["optimizer_iterations_before"])

    def test_16_method_checkpoints_unique(self):
        paths = [entry["path"] for entry in self.manifest["checkpoints"].values()]
        self.assertEqual(len(paths), 4)
        self.assertEqual(len(set(paths)), 4)

    def test_17_delayed_test_loading(self):
        for method in ("source_pretrain", "without_tl", "tl_freeze", "tl_full_finetune"):
            self.assertGreater(self._event_order(f"load_test:{method}"), self._event_order(f"reload:{method}"))

    def test_18_reload_before_test(self):
        for method in ("source_pretrain", "without_tl", "tl_freeze", "tl_full_finetune"):
            self.assertTrue(self.manifest["lifecycles"][method]["test_loaded_after_reload"])

    def test_19_prediction_counts(self):
        for lifecycle in self.manifest["lifecycles"].values():
            self.assertEqual(lifecycle["prediction_count"], 32)

    def test_20_predictions_finite(self):
        for lifecycle in self.manifest["lifecycles"].values():
            self.assertTrue(lifecycle["predictions_finite"])

    def test_21_smoke_manifest_flags(self):
        self.assertEqual(self.manifest["run_type"], "smoke_only")
        self.assertFalse(self.manifest["formal_result"])
        self.assertFalse(self.manifest["performance_comparison_allowed"])
        self.assertEqual(self.manifest["status"], "PASS")

    def test_22_successful_cleanup(self):
        from r2_helpers.solar_smoke import finalize_smoke_root

        disposable = self.root.parent / f"cleanup_{uuid.uuid4().hex}"
        disposable.mkdir(parents=True)
        self.assertTrue(finalize_smoke_root(disposable, success=True, keep_smoke=False))
        self.assertFalse(disposable.exists())

    def test_23_failure_preserves_temp_directory(self):
        from r2_helpers.solar_smoke import finalize_smoke_root

        disposable = self.root.parent / f"failure_{uuid.uuid4().hex}"
        disposable.mkdir(parents=True)
        try:
            self.assertFalse(finalize_smoke_root(disposable, success=False, keep_smoke=False))
            self.assertTrue(disposable.exists())
        finally:
            finalize_smoke_root(disposable, success=True, keep_smoke=False)


if __name__ == "__main__":
    unittest.main()
