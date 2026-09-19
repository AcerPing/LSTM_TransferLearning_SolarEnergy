from __future__ import annotations

import ast
import csv
import inspect
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from r2_helpers import solar_linear_a2_final_test as final
from r2_helpers.solar_linear_formal import ProtocolStage, target_protocol_fingerprint


class FakeScaler:
    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float64) * 100.0


class FakeModel:
    def __init__(self, prediction, *, fail=False):
        self.prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 1)
        self.fail = fail
        self.predict_calls = 0

    def predict(self, X, **kwargs):
        self.predict_calls += 1
        if self.fail:
            raise RuntimeError("synthetic prediction failure")
        return self.prediction.copy()


class SolarLinearA2FinalTestTests(unittest.TestCase):
    EXECUTION_HEAD = "1" * 40

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_root = Path(self.temporary.name) / final.LOCKED_RUN_ID
        self.run_root.mkdir()
        self.paths = {
            "source": self.run_root / "source_candidates" / final.LOCKED_SOURCE_CANDIDATE_ID / "checkpoint_epoch_0500.hdf5",
            "wotl": self.run_root / "target_wotl_candidates" / final.LOCKED_WOTL_CANDIDATE_ID / "checkpoint_epoch_0500.hdf5",
            "partial_ft": self.run_root / "target_partial_ft_candidates" / final.LOCKED_PARTIAL_FT_CANDIDATE_ID / "checkpoint_epoch_0500.hdf5",
        }
        for name, path in self.paths.items():
            path.parent.mkdir(parents=True)
            path.write_bytes(f"synthetic-{name}".encode("ascii"))
        (self.run_root / "selection").mkdir()
        self.selection = self._selection()
        self.manifest = self._manifest()
        self._write_contracts()
        self.good_git = final.GitState(self.EXECUTION_HEAD, False)

    def _score(self, lifecycle, candidate, lr, loss, sha, path):
        return {
            "protocol_version": final.LOCKED_PROTOCOL_VERSION,
            "experiment_id": final.LOCKED_EXPERIMENT_ID,
            "run_id": final.LOCKED_RUN_ID,
            "git_head": final.SELECTION_GIT_HEAD,
            "target_protocol_fingerprint": target_protocol_fingerprint("A2"),
            "lifecycle": lifecycle,
            "candidate_id": candidate,
            "learning_rate": lr,
            "best_epoch": 500,
            "validation_loss": loss,
            "validation_original_mae": loss * 100.0,
            "validation_original_rmse": loss * 120.0,
            "validation_original_r2": 1.0 - loss,
            "trainable_params": 29161 if lifecycle == "partial_ft" else 46681,
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha,
            "checkpoint_sha256_verified": True,
            "validation_only": True,
            "test_accessed": False,
            "test_metrics_used": False,
        }

    def _selection(self):
        return {
            "protocol_version": final.LOCKED_PROTOCOL_VERSION,
            "experiment_id": "A2",
            "run_id": final.LOCKED_RUN_ID,
            "git_head": final.SELECTION_GIT_HEAD,
            "target_protocol_fingerprint": target_protocol_fingerprint("A2"),
            "source_selected_candidate": final.LOCKED_SOURCE_CANDIDATE_ID,
            "source_checkpoint_sha256": final.LOCKED_SOURCE_SHA256,
            "source_checkpoint_sha256_verified": True,
            "wotl_selected_candidate": final.LOCKED_WOTL_CANDIDATE_ID,
            "wotl_selected_checkpoint_sha256": final.LOCKED_WOTL_SHA256,
            "wotl_selected_checkpoint_sha256_verified": True,
            "partial_ft_selected_candidate": final.LOCKED_PARTIAL_FT_CANDIDATE_ID,
            "partial_ft_selected_checkpoint_sha256": final.LOCKED_PARTIAL_FT_SHA256,
            "partial_ft_selected_checkpoint_sha256_verified": True,
            "comparison_pair_locked": True,
            "source_validation_comparison": [
                self._score("source", final.LOCKED_SOURCE_CANDIDATE_ID, 1e-4, 0.1, final.LOCKED_SOURCE_SHA256, self.paths["source"])
            ],
            "wotl_validation_comparison": [
                self._score("wotl", final.LOCKED_WOTL_CANDIDATE_ID, 1e-4, 0.2, final.LOCKED_WOTL_SHA256, self.paths["wotl"])
            ],
            "partial_ft_validation_comparison": [
                self._score("partial_ft", final.LOCKED_PARTIAL_FT_CANDIDATE_ID, 3e-5, 0.15, final.LOCKED_PARTIAL_FT_SHA256, self.paths["partial_ft"])
            ],
            "config_locked": True,
            "selection_locked": True,
            "checkpoint_locked": True,
            "test_metrics_used_for_selection": False,
            "test_accessed": False,
            "test_authorized": False,
            "post_test_tuning_allowed": False,
            "protocol_stage": ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        }

    def _manifest(self):
        return {
            "protocol_version": final.LOCKED_PROTOCOL_VERSION,
            "experiment_id": "A2",
            "run_id": final.LOCKED_RUN_ID,
            "direction": final.LOCKED_DIRECTION,
            "source_plant": final.LOCKED_SOURCE_PLANT,
            "target_plant": final.LOCKED_TARGET_PLANT,
            "activation": "linear",
            "git": {"head": final.SELECTION_GIT_HEAD},
            "target_protocol_fingerprint": target_protocol_fingerprint("A2"),
            "formal_eligible": True,
            "dry_run": False,
            "config_locked": True,
            "selection_locked": True,
            "checkpoint_locked": True,
            "test_accessed": False,
            "test_metrics_used_for_selection": False,
            "test_authorized": False,
            "post_test_tuning_allowed": False,
            "source_test_authorized": False,
            "source_test_accessed": False,
            "target_test_authorized": False,
            "target_test_accessed": False,
            "final_test_authorized": False,
            "final_test_executed": False,
            "historical_target_test_previously_revealed": True,
            "protocol_stage": ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
            "formal_paths": {
                "run_root": str(self.run_root),
                "final": str(self.run_root / "final"),
            },
        }

    def _write_contracts(self):
        (self.run_root / "selection/selection.json").write_text(json.dumps(self.selection), encoding="utf-8")
        (self.run_root / "protocol_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def _rewrite_selection(self):
        (self.run_root / "selection/selection.json").write_text(json.dumps(self.selection), encoding="utf-8")

    def _rewrite_manifest(self):
        (self.run_root / "protocol_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def _sha(self, path):
        return {
            final.LOCKED_SOURCE_CANDIDATE_ID: final.LOCKED_SOURCE_SHA256,
            final.LOCKED_WOTL_CANDIDATE_ID: final.LOCKED_WOTL_SHA256,
            final.LOCKED_PARTIAL_FT_CANDIDATE_ID: final.LOCKED_PARTIAL_FT_SHA256,
        }[Path(path).parent.name]

    def _scaler(self, **changes):
        values = {
            "target_plant": "Plant2",
            "target_profile": "target",
            "feature_scaler_path": Path(self.temporary.name) / "feature.joblib",
            "target_scaler_path": Path(self.temporary.name) / "target.joblib",
            "feature_scaler_sha256": "a" * 64,
            "target_scaler_sha256": "b" * 64,
            "feature_fit_split": "training",
            "target_fit_split": "training",
            "feature_fit_rows": 521,
            "target_fit_rows": 521,
            "separate_artifacts": True,
            "time_columns_scaled": False,
            "clipping_applied": False,
        }
        values.update(changes)
        return final.ScalerAudit(**values)

    def _contract(self, experiment_id, run_id):
        return SimpleNamespace(final=self.run_root / "final")

    def _preflight(self, **overrides):
        arguments = {
            "run_root": self.run_root,
            "git_state": self.good_git,
            "sha256_func": self._sha,
            "selection_validator": lambda selection: None,
            "scaler_validator": lambda manifest: self._scaler(),
            "path_contract_resolver": self._contract,
            "_allow_test_root": True,
        }
        arguments.update(overrides)
        return final.preflight_a2_final_test(**arguments)

    def _authorize(self, **overrides):
        arguments = {
            "human_authorized": True,
            "expected_execution_git_head": self.EXECUTION_HEAD,
            "run_root": self.run_root,
            "git_state": self.good_git,
            "authorization_timestamp": "2099-12-31T23:59:59Z",
            "sha256_func": self._sha,
            "selection_validator": lambda selection: None,
            "scaler_validator": lambda manifest: self._scaler(),
            "path_contract_resolver": self._contract,
            "_allow_test_root": True,
        }
        arguments.update(overrides)
        return final.prepare_a2_final_test_authorization(**arguments)

    def _data(self):
        y = np.linspace(0.1, 0.9, final.EXPECTED_TEST_SEQUENCES)
        return final.TargetTestData(
            X_test=np.zeros(final.EXPECTED_X_TEST_SHAPE, dtype=np.float32),
            y_test=y,
            target_timestamp=np.arange(final.EXPECTED_TEST_SEQUENCES),
            target_scaler=FakeScaler(),
            accessed_files=(Path(self.temporary.name) / "synthetic.csv",),
        )

    def _execute(self, authorization, data=None, models=None, **overrides):
        data = data or self._data()
        models = models or [FakeModel(data.y_test + 0.02), FakeModel(data.y_test + 0.01)]
        arguments = {
            "execute_test": True,
            "output_root": self.run_root / "final",
            "test_loader": lambda: data,
            "model_loader": lambda path: models.pop(0),
            "model_validator": lambda model: None,
            "sha256_func": self._sha,
            "git_state_provider": lambda: self.good_git,
            "_allow_test_dependencies": True,
        }
        arguments.update(overrides)
        with patch.dict(os.environ, {final.ENVIRONMENT_OPT_IN: "1"}):
            return final.execute_a2_final_test(authorization, **arguments)

    def test_01_valid_read_only_preflight(self):
        result = self._preflight()
        self.assertTrue(result.structurally_ready)
        self.assertEqual(result.target_plant, "Plant2")
        self.assertFalse((self.run_root / "final").exists())

    def test_02_identity_rejections(self):
        cases = [
            ("experiment_id", "B"),
            ("run_id", "20990101T000000Z_seed1234"),
        ]
        for field, value in cases:
            with self.subTest(field=field), self.assertRaises(final.A2FinalTestError):
                self._preflight(**{field: value})
        for field, value in (
            ("direction", "Plant2_to_Plant1"),
            ("source_plant", "Plant2"),
            ("target_plant", "Plant1"),
            ("protocol_version", "wrong"),
            ("activation", "sigmoid"),
        ):
            with self.subTest(field=field):
                self.manifest[field] = value
                self._rewrite_manifest()
                with self.assertRaises(final.A2FinalTestError):
                    self._preflight()
                self.manifest = self._manifest()
                self._write_contracts()

    def test_03_formal_root_mismatch_rejected(self):
        self.manifest["formal_paths"]["final"] = str(self.run_root / "final_test")
        self._rewrite_manifest()
        with self.assertRaises(final.A2FinalTestError):
            self._preflight()

    def test_04_contract_root_mismatch_rejected(self):
        with self.assertRaises(final.A2FinalTestError):
            self._preflight(path_contract_resolver=lambda e, r: SimpleNamespace(final=self.run_root / "other"))

    def test_05_selected_candidate_rejections(self):
        fields = {
            "source_selected_candidate": "SRC_lr3e-5",
            "wotl_selected_candidate": "WOTL_lr3e-5",
            "partial_ft_selected_candidate": "PFT_lr1e-5",
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                self.selection[field] = value
                self._rewrite_selection()
                with self.assertRaises(final.A2FinalTestError):
                    self._preflight()
                self.selection = self._selection()
                self._write_contracts()

    def test_06_selected_sha_rejections(self):
        for field in ("source_checkpoint_sha256", "wotl_selected_checkpoint_sha256", "partial_ft_selected_checkpoint_sha256"):
            with self.subTest(field=field):
                self.selection[field] = "0" * 64
                self._rewrite_selection()
                with self.assertRaises(final.A2FinalTestError):
                    self._preflight()
                self.selection = self._selection()
                self._write_contracts()

    def test_07_selected_epoch_rejections(self):
        for field in ("source_validation_comparison", "wotl_validation_comparison", "partial_ft_validation_comparison"):
            with self.subTest(field=field):
                self.selection[field][0]["best_epoch"] = 499
                self._rewrite_selection()
                with self.assertRaises(final.A2FinalTestError):
                    self._preflight()
                self.selection = self._selection()
                self._write_contracts()

    def test_08_actual_checkpoint_sha_rejected(self):
        with self.assertRaises(final.A2FinalTestError):
            self._preflight(sha256_func=lambda path: "0" * 64)

    def test_09_lock_rejections(self):
        for target, field in (
            ("manifest", "config_locked"),
            ("manifest", "selection_locked"),
            ("manifest", "checkpoint_locked"),
            ("selection", "config_locked"),
            ("selection", "selection_locked"),
            ("selection", "checkpoint_locked"),
            ("selection", "comparison_pair_locked"),
        ):
            with self.subTest(target=target, field=field):
                getattr(self, target)[field] = False
                getattr(self, f"_rewrite_{target}")()
                with self.assertRaises(final.A2FinalTestError):
                    self._preflight()
                self.selection = self._selection()
                self.manifest = self._manifest()
                self._write_contracts()

    def test_10_test_state_rejections(self):
        manifest_fields = (
            "test_accessed",
            "test_metrics_used_for_selection",
            "test_authorized",
            "post_test_tuning_allowed",
            "source_test_authorized",
            "source_test_accessed",
            "target_test_authorized",
            "target_test_accessed",
            "final_test_authorized",
            "final_test_executed",
        )
        for field in manifest_fields:
            with self.subTest(field=field):
                self.manifest[field] = True
                self._rewrite_manifest()
                with self.assertRaises(final.A2FinalTestError):
                    self._preflight()
                self.manifest = self._manifest()
                self._write_contracts()

    def test_11_selection_test_state_rejections(self):
        for field in ("test_metrics_used_for_selection", "test_accessed", "test_authorized", "post_test_tuning_allowed"):
            with self.subTest(field=field):
                self.selection[field] = True
                self._rewrite_selection()
                with self.assertRaises(final.A2FinalTestError):
                    self._preflight()
                self.selection = self._selection()
                self._write_contracts()

    def test_12_historical_exposure_true_accepted_and_false_rejected(self):
        self.assertTrue(self._preflight().historical_target_test_previously_revealed)
        self.manifest["historical_target_test_previously_revealed"] = False
        self._rewrite_manifest()
        with self.assertRaises(final.A2FinalTestError):
            self._preflight()

    def test_13_selection_provenance_rejected(self):
        self.selection["git_head"] = "0" * 40
        self._rewrite_selection()
        with self.assertRaises(final.A2FinalTestError):
            self._preflight()

    def test_14_manifest_provenance_rejected(self):
        self.manifest["git"]["head"] = "0" * 40
        self._rewrite_manifest()
        with self.assertRaises(final.A2FinalTestError):
            self._preflight()

    def test_15_scaler_provenance_rejections(self):
        cases = (
            {"target_plant": "Plant1"},
            {"target_fit_split": "test"},
            {"target_fit_rows": 2612},
            {"separate_artifacts": False},
            {"clipping_applied": True},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(final.A2FinalTestError):
                self._preflight(scaler_validator=lambda manifest, c=changes: self._scaler(**c))

    def test_16_preexisting_final_rejected(self):
        (self.run_root / "final").mkdir()
        with self.assertRaises(final.A2FinalTestError):
            self._preflight()

    def test_17_preexisting_legacy_final_test_rejected(self):
        (self.run_root / "final_test").mkdir()
        with self.assertRaises(final.A2FinalTestError):
            self._preflight()

    def test_18_authorization_requires_human_and_explicit_head(self):
        with self.assertRaises(final.A2FinalTestError):
            self._authorize(human_authorized=False)
        with self.assertRaises(final.A2FinalTestError):
            self._authorize(expected_execution_git_head="bad")

    def test_19_wrong_execution_head_and_dirty_tree_rejected(self):
        with self.assertRaises(final.A2FinalTestError):
            self._authorize(git_state=final.GitState("2" * 40, False))
        with self.assertRaises(final.A2FinalTestError):
            self._authorize(git_state=final.GitState(self.EXECUTION_HEAD, True))

    def test_20_authorization_reads_no_test_and_writes_nothing(self):
        before = sorted(str(p.relative_to(self.run_root)) for p in self.run_root.rglob("*") if p.is_file())
        authorization = self._authorize()
        after = sorted(str(p.relative_to(self.run_root)) for p in self.run_root.rglob("*") if p.is_file())
        self.assertEqual(before, after)
        self.assertFalse(authorization.test_accessed)
        self.assertFalse((self.run_root / "final").exists())

    def test_21_execution_requires_authorization(self):
        with self.assertRaises(final.A2FinalTestError):
            final.execute_a2_final_test(object(), execute_test=True)

    def test_22_execute_false_does_not_execute(self):
        calls = []
        result = final.execute_a2_final_test(self._authorize(), execute_test=False, test_loader=lambda: calls.append(1))
        self.assertIsNone(result)
        self.assertEqual(calls, [])

    def test_23_missing_environment_opt_in_rejected(self):
        calls = []
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(final.ENVIRONMENT_OPT_IN, None)
            with self.assertRaises(final.A2FinalTestError):
                final.execute_a2_final_test(self._authorize(), execute_test=True, test_loader=lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_24_one_time_loader_rejects_second_access(self):
        access = final.OneTimeTestAccess(self._data)
        access.load()
        with self.assertRaises(final.A2FinalTestError):
            access.load()
        self.assertEqual(access.access_count, 1)

    def test_25_successful_synthetic_execution_and_artifacts(self):
        result = self._execute(self._authorize())
        self.assertEqual(result.output_root.name, "final")
        self.assertEqual({p.name for p in result.output_root.iterdir()}, final.CANONICAL_ARTIFACT_FILENAMES)
        state = json.loads((result.output_root / "final_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["state"], final.FinalTestState.TEST_COMPLETED.value)
        self.assertEqual(state["test_access_count"], 1)
        comparison = json.loads((result.output_root / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(comparison["unit"], "kW")
        self.assertEqual(comparison["target_plant"], "Plant2")

    def test_26_started_state_is_durable_before_loader(self):
        output = self.run_root / "final"
        calls = []
        def loader():
            calls.append(1)
            state = json.loads((output / "final_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["state"], final.FinalTestState.TEST_ACCESS_STARTED.value)
            raise RuntimeError("loader failure")
        with self.assertRaisesRegex(final.A2FinalTestError, "TEST_ACCESS_STARTED"):
            self._execute(self._authorize(), test_loader=loader)
        self.assertEqual(calls, [1])

    def test_27_incomplete_state_is_durable_after_prediction_failure(self):
        data = self._data()
        models = [FakeModel(data.y_test, fail=True), FakeModel(data.y_test)]
        with self.assertRaisesRegex(final.A2FinalTestError, "durable Test access"):
            self._execute(self._authorize(), data=data, models=models)
        state = json.loads((self.run_root / "final/final_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["state"], final.FinalTestState.TEST_ACCESSED_INCOMPLETE.value)
        self.assertEqual(state["test_access_count"], 1)

    def test_28_rerun_rejected_after_started_state(self):
        output = self.run_root / "final"
        output.mkdir()
        (output / "final_state.json").write_text(json.dumps({"state": final.FinalTestState.TEST_ACCESS_STARTED.value}), encoding="utf-8")
        with self.assertRaises(final.A2FinalTestError):
            self._authorize()

    def test_29_rerun_rejected_after_incomplete_state(self):
        output = self.run_root / "final"
        output.mkdir()
        (output / "final_state.json").write_text(json.dumps({"state": final.FinalTestState.TEST_ACCESSED_INCOMPLETE.value}), encoding="utf-8")
        with self.assertRaises(final.A2FinalTestError):
            self._authorize()

    def test_30_rerun_rejected_after_completed_state(self):
        output = self.run_root / "final"
        output.mkdir()
        (output / "final_state.json").write_text(json.dumps({"state": final.FinalTestState.TEST_COMPLETED.value}), encoding="utf-8")
        with self.assertRaises(final.A2FinalTestError):
            self._authorize()

    def test_31_shape_contract(self):
        final.validate_target_test_data(self._data())
        bad = replace(self._data(), X_test=np.zeros((1, 5, 5)))
        with self.assertRaises(final.A2FinalTestError):
            final.validate_target_test_data(bad)

    def test_32_inverse_transform_and_original_metrics(self):
        result = final.evaluate_raw_prediction(np.array([0.0, 0.5, 1.0]), np.array([-0.1, 0.4, 0.8]), FakeScaler())
        np.testing.assert_array_equal(result.y_true_original, [0.0, 50.0, 100.0])
        np.testing.assert_array_equal(result.prediction_original, [-10.0, 40.0, 80.0])
        self.assertAlmostEqual(result.metrics.mae, 40.0 / 3.0)
        self.assertAlmostEqual(result.metrics.mse, 200.0)
        self.assertAlmostEqual(result.metrics.rmse, np.sqrt(200.0))
        self.assertAlmostEqual(result.metrics.r2, 0.88)

    def test_33_negative_prediction_not_clipped(self):
        result = final.evaluate_raw_prediction(np.array([0.0, 0.5, 1.0]), np.array([-0.1, 0.4, 0.8]), FakeScaler())
        self.assertEqual(result.prediction_original[0], -10.0)
        self.assertEqual(result.metrics.negative_prediction_count, 1)

    def test_34_residual_is_actual_minus_predicted(self):
        result = self._execute(self._authorize())
        with (result.output_root / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertAlmostEqual(float(row["wotl_residual"]), float(row["y_true_original"]) - float(row["wotl_pred_original"]))
        self.assertAlmostEqual(float(row["partial_ft_residual"]), float(row["y_true_original"]) - float(row["partial_ft_pred_original"]))

    def _metric(self, mae, mse, rmse, r2):
        return final.OriginalScaleMetrics(mae, mse, rmse, r2, 3, 0)

    def test_35_positive_classification(self):
        self.assertIs(final.classify_transfer(self._metric(10, 100, 10, 0.7), self._metric(9, 81, 9, 0.8)), final.TransferClassification.POSITIVE_TRANSFER)

    def test_36_partial_positive_classification(self):
        self.assertIs(final.classify_transfer(self._metric(10, 100, 10, 0.7), self._metric(9, 81, 9, 0.7)), final.TransferClassification.PARTIAL_POSITIVE_TRANSFER)

    def test_37_mixed_classification(self):
        self.assertIs(final.classify_transfer(self._metric(10, 100, 10, 0.7), self._metric(9, 121, 11, 0.8)), final.TransferClassification.MIXED_RESULT)

    def test_38_negative_classification(self):
        self.assertIs(final.classify_transfer(self._metric(10, 100, 10, 0.7), self._metric(11, 121, 11, 0.6)), final.TransferClassification.NEGATIVE_TRANSFER)

    def test_39_metadata_contract(self):
        metrics = final.compute_original_scale_metrics(np.array([0.0, 1.0]), np.array([-1.0, 2.0]))
        self.assertEqual(metrics.target_name, "DC_POWER")
        self.assertEqual(metrics.unit, "kW")

    def test_40_authorization_provenance_is_separate(self):
        authorization = self._authorize()
        self.assertEqual(authorization.selection_git_head, final.SELECTION_GIT_HEAD)
        self.assertEqual(authorization.execution_git_head, self.EXECUTION_HEAD)
        self.assertNotEqual(authorization.selection_git_head, authorization.execution_git_head)

    def test_41_tampered_authorization_rejected(self):
        authorization = replace(self._authorize(), wotl_checkpoint_sha256="0" * 64)
        with self.assertRaises(final.A2FinalTestError):
            final.execute_a2_final_test(authorization, execute_test=False)

    def test_42_module_contains_no_fit_or_clipping(self):
        source = inspect.getsource(final)
        tree = ast.parse(source)
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("fit", called)
        self.assertNotIn("fit_transform", called)
        self.assertNotIn("clip", called)

    def test_43_real_loader_is_plant2_only(self):
        source = inspect.getsource(final.load_locked_a2_target_test)
        self.assertIn("LOCKED_TARGET_PLANT", source)
        self.assertIn('LINEAR_EXPERIMENTS[LOCKED_EXPERIMENT_ID]', source)
        self.assertNotIn('LINEAR_EXPERIMENTS["B"]', source)

    def test_44_output_root_is_manifest_authoritative(self):
        preflight = self._preflight()
        self.assertEqual(preflight.final_root.resolve(), Path(self.manifest["formal_paths"]["final"]).resolve())
        self.assertEqual(preflight.final_root.name, "final")

    def test_45_execution_git_rechecked_before_actions(self):
        calls = []
        with patch.dict(os.environ, {final.ENVIRONMENT_OPT_IN: "1"}):
            with self.assertRaises(final.A2FinalTestError):
                final.execute_a2_final_test(
                    self._authorize(),
                    execute_test=True,
                    output_root=self.run_root / "final",
                    test_loader=lambda: calls.append(1),
                    git_state_provider=lambda: final.GitState("2" * 40, False),
                    _allow_test_dependencies=True,
                )
        self.assertEqual(calls, [])
        self.assertFalse((self.run_root / "final").exists())


if __name__ == "__main__":
    unittest.main()
