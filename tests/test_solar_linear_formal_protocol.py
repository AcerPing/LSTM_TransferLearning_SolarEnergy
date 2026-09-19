from __future__ import annotations

import ast
import inspect
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from r2_config.solar_linear import LINEAR_EXPERIMENTS, LINEAR_FORMAL_BASE
from r2_config.solar_linear_formal import (
    A2_APPROVED_UNTRACKED_BASELINE_ID,
    A2_APPROVED_UNTRACKED_BASELINE_PROVENANCE_ANCHOR,
    A2_APPROVED_UNTRACKED_BASELINE_RELATIVE_PATH,
    A2_APPROVED_UNTRACKED_BASELINE_ROW_COUNT,
    A2_APPROVED_UNTRACKED_BASELINE_SCHEMA_VERSION,
    A2_APPROVED_UNTRACKED_BASELINE_SHA256,
    A2_APPROVED_UNTRACKED_BASELINE_SOURCE_HEAD,
    FORMAL_LINEAR_LAYER_CLASSES,
    FORMAL_PROTOCOL_VERSION,
)
from r2_helpers import solar_linear_formal as formal
from r2_helpers import solar_linear_formal_train as formal_train


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

    def _step10a_manifest(self, *, completed):
        manifest = dict(self.a2.manifest)
        expected_ids = tuple(
            identifier
            for lifecycle in formal.FORMAL_LIFECYCLES
            for identifier, _ in formal.candidate_registry()[lifecycle]
        )
        manifest.update(
            {
                "dry_run": False,
                "formal_eligible": True,
                "step10a_scope": formal.FormalExecutionScope.TRAIN_VALIDATION_ONLY.value,
                "maximum_epochs": formal.FORMAL_CALLBACK_POLICY.maximum_epochs,
                "shuffle": False,
                "source_test_authorized": False,
                "source_test_accessed": False,
                "target_test_authorized": False,
                "target_test_accessed": False,
                "final_test_authorized": False,
                "final_test_executed": False,
                "training_validation_completed": completed,
                "executed_candidate_ids": expected_ids if completed else (),
                "source_dependency_selection_path": str(
                    self.a2.paths.run_root / "source_dependency_selection.json"
                ),
                "source_dependency_selection_sha256": "a" * 64 if completed else None,
                "approved_untracked_baseline_path": (
                    A2_APPROVED_UNTRACKED_BASELINE_RELATIVE_PATH
                ),
                "approved_untracked_baseline_sha256": (
                    A2_APPROVED_UNTRACKED_BASELINE_SHA256
                ),
                "approved_untracked_baseline_schema_version": (
                    A2_APPROVED_UNTRACKED_BASELINE_SCHEMA_VERSION
                ),
                "approved_untracked_baseline_id": A2_APPROVED_UNTRACKED_BASELINE_ID,
                "approved_untracked_baseline_source_head": (
                    A2_APPROVED_UNTRACKED_BASELINE_SOURCE_HEAD
                ),
                "approved_untracked_baseline_provenance_anchor": (
                    A2_APPROVED_UNTRACKED_BASELINE_PROVENANCE_ANCHOR
                ),
                "approved_untracked_baseline_row_count": (
                    A2_APPROVED_UNTRACKED_BASELINE_ROW_COUNT
                ),
                "approved_untracked_current_count": (
                    A2_APPROVED_UNTRACKED_BASELINE_ROW_COUNT
                ),
                "approved_untracked_identity_matches": (
                    A2_APPROVED_UNTRACKED_BASELINE_ROW_COUNT
                ),
                "approved_untracked_extra_paths": 0,
                "approved_untracked_missing_paths": 0,
                "approved_untracked_mutated_paths": 0,
                "approved_untracked_verified": True,
                "selection_locked": False,
                "checkpoint_locked": False,
                "test_accessed": False,
                "test_authorized": False,
                "test_metrics_used_for_selection": False,
                "protocol_stage": (
                    formal.ProtocolStage.FORMAL_TRAINING_VALIDATION_COMPLETE_AWAITING_STEP_10B.value
                    if completed
                    else formal.ProtocolStage.CONFIG_LOCKED_AWAITING_TRAINING.value
                ),
            }
        )
        return manifest

    def test_11b_step10a_manifest_intermediate_stage_is_conditional(self):
        initial = self._step10a_manifest(completed=False)
        completed = self._step10a_manifest(completed=True)
        formal.validate_protocol_manifest(initial)
        formal.validate_protocol_manifest(completed)
        formal.validate_protocol_manifest(self.b.manifest)
        self.assertNotIn("step10a_scope", self.b.manifest)
        bad = dict(completed)
        bad["target_test_accessed"] = True
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_protocol_manifest(bad)

        for field_name, value in (
            ("approved_untracked_verified", False),
            ("approved_untracked_identity_matches", 3466),
            ("approved_untracked_extra_paths", 1),
            ("approved_untracked_baseline_sha256", "0" * 64),
        ):
            with self.subTest(field_name=field_name):
                tampered = dict(completed)
                tampered[field_name] = value
                with self.assertRaises(formal.FormalProtocolError):
                    formal.validate_protocol_manifest(tampered)

    def test_11c_step10a_intermediate_stage_cannot_authorize_final_test(self):
        manifest = self._step10a_manifest(completed=True)
        with patch.object(formal, "validate_selection_record"):
            with self.assertRaisesRegex(
                formal.FormalProtocolError,
                "Manifest protocol stage cannot authorize Final Test",
            ):
                formal.authorize_final_test(
                    manifest,
                    object(),
                    human_authorized=True,
                )

    def test_11d_step10a_locked_stage_and_canonical_transition_validate(self):
        completed = self._step10a_manifest(completed=True)
        formal.validate_protocol_manifest(completed)
        selection = SimpleNamespace(
            experiment_id="A2",
            run_id=completed["run_id"],
        )
        with patch.object(formal_train, "validate_selection_record"):
            locked = formal_train.lock_manifest_after_selection(completed, selection)
        formal.validate_protocol_manifest(locked)
        self.assertEqual(
            locked["protocol_stage"],
            formal.ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        )
        self.assertTrue(locked["config_locked"])
        self.assertTrue(locked["selection_locked"])
        self.assertTrue(locked["checkpoint_locked"])

    def test_11e_step10a_locked_stage_rejects_lock_and_test_flag_tampering(self):
        locked = self._step10a_manifest(completed=True)
        locked.update(
            {
                "protocol_stage": (
                    formal.ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value
                ),
                "selection_locked": True,
                "checkpoint_locked": True,
            }
        )
        formal.validate_protocol_manifest(locked)
        for field_name, value in (
            ("config_locked", False),
            ("selection_locked", False),
            ("checkpoint_locked", False),
            ("test_accessed", True),
            ("test_metrics_used_for_selection", True),
            ("test_authorized", True),
            ("post_test_tuning_allowed", True),
            ("source_test_accessed", True),
            ("target_test_accessed", True),
            ("final_test_authorized", True),
            ("final_test_executed", True),
        ):
            with self.subTest(field_name=field_name):
                tampered = dict(locked)
                tampered[field_name] = value
                with self.assertRaises(formal.FormalProtocolError):
                    formal.validate_protocol_manifest(tampered)

    def test_11f_step10a_locked_stage_rejects_baseline_proof_tampering(self):
        locked = self._step10a_manifest(completed=True)
        locked.update(
            {
                "protocol_stage": (
                    formal.ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value
                ),
                "selection_locked": True,
                "checkpoint_locked": True,
            }
        )
        for field_name, value in (
            ("approved_untracked_verified", False),
            ("approved_untracked_identity_matches", 3466),
            ("approved_untracked_missing_paths", 1),
            ("approved_untracked_baseline_sha256", "0" * 64),
        ):
            with self.subTest(field_name=field_name):
                tampered = dict(locked)
                tampered[field_name] = value
                with self.assertRaises(formal.FormalProtocolError):
                    formal.validate_protocol_manifest(tampered)
        missing = dict(locked)
        del missing["approved_untracked_verified"]
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_protocol_manifest(missing)

    def test_11g_step10a_scope_rejects_unapproved_future_stage(self):
        manifest = self._step10a_manifest(completed=True)
        manifest.update(
            {
                "protocol_stage": formal.ProtocolStage.TEST_AUTHORIZED.value,
                "selection_locked": True,
                "checkpoint_locked": True,
            }
        )
        with self.assertRaisesRegex(
            formal.FormalProtocolError,
            "Step10A manifest has an invalid lifecycle stage",
        ):
            formal.validate_protocol_manifest(manifest)

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
