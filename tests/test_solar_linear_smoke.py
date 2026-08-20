from __future__ import annotations

import json
import os
import unittest


RUN_SMOKE = os.environ.get("SOLAR_RUN_SMOKE") == "1"


@unittest.skipUnless(
    RUN_SMOKE,
    "Explicit integration only: set SOLAR_RUN_SMOKE=1 to execute six 2-epoch lifecycles",
)
class SolarLinearSmokeIntegrationTests(unittest.TestCase):
    def test_phase_ii_a2_then_b(self):
        # Delayed import ensures ordinary unit-test discovery neither imports
        # TensorFlow through this module nor runs training.
        from r2_helpers.solar_linear_smoke import run_phase_ii_smoke

        result = run_phase_ii_smoke()
        self.assertEqual(tuple(result.experiments), ("A2", "B"))
        self.assertTrue(result.protected_roots_unchanged)
        self.assertFalse(result.linear_formal_created)
        self.assertFalse(
            any(path.name.endswith("_test.csv") for path in result.accessed_data_files)
        )
        summary = {"run_root": str(result.run_root), "experiments": {}}
        for experiment_id, experiment in result.experiments.items():
            summary["experiments"][experiment_id] = {}
            for method in ("source", "wotl", "partial_ft"):
                lifecycle = getattr(experiment, method)
                self.assertEqual(lifecycle.epochs, 2)
                self.assertTrue(lifecycle.reload_passed)
                self.assertFalse(lifecycle.test_accessed)
                self.assertTrue(lifecycle.checkpoint_path.is_file())
                manifest = json.loads(lifecycle.manifest_path.read_text(encoding="utf-8"))
                self.assertTrue(manifest["smoke_only"])
                self.assertFalse(manifest["formal_eligible"])
                self.assertFalse(manifest["test_accessed"])
                self.assertFalse(manifest["test_metrics_computed"])
                summary["experiments"][experiment_id][method] = {
                    "best_epoch": lifecycle.best_epoch,
                    "best_val_loss": lifecycle.best_val_loss,
                    "checkpoint": str(lifecycle.checkpoint_path),
                    "sha256": lifecycle.checkpoint_sha256,
                }
            self.assertTrue(all(experiment.partial_state_audit.__dict__.values()))
        print("PHASE_II_SMOKE_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
