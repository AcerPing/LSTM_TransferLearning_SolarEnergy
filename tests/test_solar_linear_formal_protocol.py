from __future__ import annotations

import ast
import inspect
import unittest
from dataclasses import replace
from pathlib import Path

from r2_config.solar_linear import LINEAR_EXPERIMENTS, LINEAR_FORMAL_BASE
from r2_config.solar_linear_formal import FORMAL_PROTOCOL_VERSION
from r2_config.solar_linear_formal import FORMAL_LINEAR_LAYER_CLASSES
from r2_helpers import solar_linear_formal as formal


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


class SolarLinearFormalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.git = formal.GitIdentity(head="1" * 40, branch="unit-test", dirty=False)
        cls.a2_run_root = (
            LINEAR_EXPERIMENTS["A2"].output_root / "20991231T235959Z_seed1234"
        )
        cls.b_run_root = (
            LINEAR_EXPERIMENTS["B"].output_root / "20991230T235959Z_seed1234"
        )
        cls.formal_snapshot_before = _filesystem_snapshot(LINEAR_FORMAL_BASE)
        cls.a2_run_snapshot_before = _filesystem_snapshot(cls.a2_run_root)
        cls.b_run_snapshot_before = _filesystem_snapshot(cls.b_run_root)
        cls.a2 = formal.prepare_formal_dry_run(
            "A2", "20991231T235959Z_seed1234", git_identity=cls.git
        )
        cls.b = formal.prepare_formal_dry_run(
            "B", "20991230T235959Z_seed1234", git_identity=cls.git
        )
        cls.formal_snapshot_after = _filesystem_snapshot(LINEAR_FORMAL_BASE)
        cls.a2_run_snapshot_after = _filesystem_snapshot(cls.a2_run_root)
        cls.b_run_snapshot_after = _filesystem_snapshot(cls.b_run_root)

    def _source_provenance(self, experiment_id="A2"):
        spec = LINEAR_EXPERIMENTS[experiment_id]
        run_id = "20991231T235959Z_seed1234"
        identifier = "SRC_lr1e-4"
        return formal.SourceCheckpointProvenance(
            protocol_version=FORMAL_PROTOCOL_VERSION,
            experiment_id=experiment_id,
            source_run_id=run_id,
            source_candidate_id=identifier,
            source_activation="linear",
            source_learning_rate=1e-4,
            source_best_epoch=17,
            source_validation_loss=0.125,
            source_checkpoint_path=(
                spec.output_root
                / run_id
                / "source_candidates"
                / identifier
                / "checkpoint_epoch_0017.hdf5"
            ),
            source_checkpoint_sha256="a" * 64,
            source_checkpoint_sha256_verified=True,
            source_git_head="1" * 40,
            source_training_profile=spec.source_profile_path / "normalized_scale_training.csv",
            source_validation_profile=spec.source_profile_path / "normalized_scale_validation.csv",
            source_layer_classes=FORMAL_LINEAR_LAYER_CLASSES,
            source_input_shape=(5, 5),
            source_output_shape=(1,),
            source_total_params=46681,
            validation_selected=True,
            formal_eligible=True,
        )

    def test_01_dry_run_does_not_create_or_modify_formal_output(self):
        self.assertIsNone(self.a2_run_snapshot_before)
        self.assertIsNone(self.b_run_snapshot_before)
        self.assertFalse(self.a2.paths.run_root.exists())
        self.assertFalse(self.b.paths.run_root.exists())
        self.assertIsNone(self.a2_run_snapshot_after)
        self.assertIsNone(self.b_run_snapshot_after)
        self.assertEqual(self.formal_snapshot_after, self.formal_snapshot_before)
        self.assertEqual(_filesystem_snapshot(LINEAR_FORMAL_BASE), self.formal_snapshot_before)
        for dry_run in (self.a2, self.b):
            self.assertFalse(dry_run.formal_root_created)
            self.assertEqual(dry_run.training_epochs_executed, 0)

    def test_02_dry_run_has_all_six_candidates(self):
        self.assertEqual(len(self.a2.candidate_checkpoint_patterns), 6)
        self.assertEqual(len(self.a2.callbacks), 6)
        self.assertEqual(
            set(self.a2.candidate_checkpoint_patterns),
            {"SRC_lr1e-4", "SRC_lr3e-5", "WOTL_lr1e-4", "WOTL_lr3e-5", "PFT_lr1e-5", "PFT_lr3e-5"},
        )
        self.assertTrue(all(not path.parent.exists() for path in self.a2.candidate_checkpoint_patterns.values()))

    def test_03_callback_classes_order_and_parameters(self):
        callbacks = self.a2.callbacks["SRC_lr1e-4"]
        self.assertEqual(
            tuple(type(callback).__name__ for callback in callbacks),
            ("ModelCheckpoint", "ReduceLROnPlateau", "EarlyStopping", "TerminateOnNaN"),
        )
        checkpoint, reduce_lr, early_stopping, _ = callbacks
        self.assertEqual(checkpoint.monitor, "val_loss")
        self.assertTrue(checkpoint.save_best_only)
        self.assertFalse(checkpoint.save_weights_only)
        self.assertEqual((reduce_lr.factor, reduce_lr.patience, reduce_lr.min_lr, reduce_lr.cooldown), (0.1, 20, 1e-7, 0))
        self.assertEqual(early_stopping.patience, 50)
        self.assertEqual(early_stopping.min_delta, 0.0)
        self.assertFalse(early_stopping.restore_best_weights)

    def test_04_manifest_schema_complete_and_immutable(self):
        for dry_run in (self.a2, self.b):
            formal.validate_protocol_manifest(dry_run.manifest)
            self.assertTrue(set(formal.REQUIRED_PROTOCOL_MANIFEST_FIELDS).issubset(dry_run.manifest))
            self.assertTrue(dry_run.manifest["config_locked"])
            self.assertFalse(dry_run.manifest["formal_eligible"])
            self.assertFalse(dry_run.manifest["test_authorized"])
            with self.assertRaises(TypeError):
                dry_run.manifest["activation"] = "sigmoid"

    def test_05_manifest_mapping_and_historical_disclosure(self):
        self.assertEqual((self.a2.manifest["source_plant"], self.a2.manifest["target_plant"]), ("Plant1", "Plant2"))
        self.assertEqual((self.b.manifest["source_plant"], self.b.manifest["target_plant"]), ("Plant2", "Plant1"))
        self.assertTrue(self.a2.manifest["historical_target_test_previously_revealed"])
        self.assertFalse(self.b.manifest["historical_target_test_previously_revealed"])

    def test_06_manifest_counts_and_partial_contract(self):
        manifest = self.a2.manifest
        self.assertEqual(manifest["training_rows"], {"source": 2088, "target": 521})
        self.assertEqual(manifest["validation_sequences"], {"source": 518, "target": 126})
        self.assertEqual(manifest["test_sequences"], {"source": 648, "target": 2607})
        self.assertEqual(manifest["transferred_layers"], (2, 3, 4, 5))
        self.assertEqual(manifest["trainable_layers"], (1, 4, 6))
        self.assertEqual(manifest["frozen_layers"], (2, 3, 5))

    def test_07_scaler_provenance_reads_no_test_csv(self):
        for dry_run in (self.a2, self.b):
            names = {path.name for path in dry_run.accessed_files}
            self.assertEqual(names, {"feature_scaler.joblib", "target_scaler.joblib", "scaler_manifest.json"})
            self.assertEqual(len(dry_run.accessed_files), 6)
            for digest in dry_run.manifest["feature_scaler_sha256"].values():
                self.assertEqual(len(digest), 64)
            for digest in dry_run.manifest["target_scaler_sha256"].values():
                self.assertEqual(len(digest), 64)

    def test_08_valid_source_provenance(self):
        provenance = self._source_provenance()
        formal.validate_source_checkpoint_provenance(provenance, expected_git_head="1" * 40)

    def test_09_sigmoid_source_rejected(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_source_checkpoint_provenance(
                replace(self._source_provenance(), source_activation="sigmoid"),
                expected_git_head="1" * 40,
            )

    def test_10_smoke_source_checkpoint_rejected(self):
        provenance = replace(
            self._source_provenance(),
            source_checkpoint_path=(
                Path("reports/Solar Energy Result/_smoke_linear/run/A2/source/checkpoint_epoch_0002.hdf5")
            ),
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_source_checkpoint_provenance(provenance, expected_git_head="1" * 40)

    def test_11_wrong_source_profile_rejected(self):
        provenance = replace(
            self._source_provenance(),
            source_validation_profile=Path("wrong/validation.csv"),
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_source_checkpoint_provenance(provenance, expected_git_head="1" * 40)

    def test_11a_source_architecture_and_sha_proof_required(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_source_checkpoint_provenance(
                replace(self._source_provenance(), source_total_params=46680),
                expected_git_head="1" * 40,
            )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_source_checkpoint_provenance(
                replace(self._source_provenance(), source_checkpoint_sha256_verified=False),
                expected_git_head="1" * 40,
            )

    def test_12_phase_iii_source_contains_no_training_or_inference_call(self):
        for module in (formal, __import__("r2_config.solar_linear_formal", fromlist=["*"])):
            source = inspect.getsource(module)
            tree = ast.parse(source)
            called_attributes = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            self.assertTrue({"fit", "predict"}.isdisjoint(called_attributes))
            self.assertNotIn("normalized_scale_test.csv", source)
            self.assertNotIn("original_scale_test.csv", source)


if __name__ == "__main__":
    unittest.main()
