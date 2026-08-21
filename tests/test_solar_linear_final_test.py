from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from r2_config.solar_linear_formal import FORMAL_PROTOCOL_VERSION
from r2_helpers import solar_linear_final_test as final
from r2_helpers.solar_linear_formal import (
    ProtocolStage,
    target_protocol_fingerprint,
)


class FakeScaler:
    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float64) * 100.0


class FakeModel:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 1)
        self.predict_calls = []

    def predict(self, X, **kwargs):
        self.predict_calls.append((X, kwargs))
        return self.prediction.copy()


class FailingPredictionModel(FakeModel):
    def predict(self, X, **kwargs):
        self.predict_calls.append((X, kwargs))
        raise RuntimeError("simulated prediction failure")


class SolarLinearFinalTestTests(unittest.TestCase):
    EXECUTION_GIT_HEAD = "1" * 40

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_root = Path(self.temporary.name) / final.LOCKED_RUN_ID
        self.run_root.mkdir()
        self.paths = {
            "source": self.run_root
            / "source_candidates"
            / final.LOCKED_SOURCE_CANDIDATE_ID
            / f"checkpoint_epoch_{final.LOCKED_SOURCE_EPOCH:04d}.hdf5",
            "wotl": self.run_root
            / "target_wotl_candidates"
            / final.LOCKED_WOTL_CANDIDATE_ID
            / f"checkpoint_epoch_{final.LOCKED_WOTL_EPOCH:04d}.hdf5",
            "partial_ft": self.run_root
            / "target_partial_ft_candidates"
            / final.LOCKED_PARTIAL_FT_CANDIDATE_ID
            / f"checkpoint_epoch_{final.LOCKED_PARTIAL_FT_EPOCH:04d}.hdf5",
        }
        for name, path in self.paths.items():
            path.parent.mkdir(parents=True)
            path.write_bytes(f"fake-{name}-checkpoint".encode("ascii"))
        (self.run_root / "selection").mkdir()
        self.selection = self._selection()
        self.manifest = self._manifest()
        self._write_contracts()
        self.good_git = final.GitState(
            head=self.EXECUTION_GIT_HEAD,
            tracked_dirty=False,
        )

    @staticmethod
    def _filesystem_snapshot(root):
        path = Path(root)
        if not path.exists():
            return None
        return tuple(
            sorted(
                (
                    str(item.relative_to(path)),
                    item.stat().st_size,
                    item.stat().st_mtime_ns,
                )
                for item in path.rglob("*")
                if item.is_file()
            )
        )

    def _score(
        self,
        lifecycle,
        candidate_id,
        learning_rate,
        loss,
        epoch,
        sha,
        path,
    ):
        return {
            "protocol_version": FORMAL_PROTOCOL_VERSION,
            "experiment_id": final.LOCKED_EXPERIMENT_ID,
            "run_id": final.LOCKED_RUN_ID,
            "git_head": final.SELECTION_GIT_HEAD,
            "target_protocol_fingerprint": target_protocol_fingerprint("B"),
            "lifecycle": lifecycle,
            "candidate_id": candidate_id,
            "learning_rate": learning_rate,
            "best_epoch": epoch,
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
        source = [
            self._score(
                "source",
                final.LOCKED_SOURCE_CANDIDATE_ID,
                1e-4,
                0.10,
                final.LOCKED_SOURCE_EPOCH,
                final.LOCKED_SOURCE_SHA256,
                self.paths["source"],
            ),
            self._score(
                "source",
                "SRC_lr3e-5",
                3e-5,
                0.20,
                500,
                "a" * 64,
                self.run_root / "source_candidates/SRC_lr3e-5/checkpoint_epoch_0500.hdf5",
            ),
        ]
        wotl = [
            self._score(
                "wotl",
                final.LOCKED_WOTL_CANDIDATE_ID,
                1e-4,
                0.11,
                final.LOCKED_WOTL_EPOCH,
                final.LOCKED_WOTL_SHA256,
                self.paths["wotl"],
            ),
            self._score(
                "wotl",
                "WOTL_lr3e-5",
                3e-5,
                0.21,
                500,
                "b" * 64,
                self.run_root / "target_wotl_candidates/WOTL_lr3e-5/checkpoint_epoch_0500.hdf5",
            ),
        ]
        partial = [
            self._score(
                "partial_ft",
                "PFT_lr1e-5",
                1e-5,
                0.22,
                16,
                "c" * 64,
                self.run_root / "target_partial_ft_candidates/PFT_lr1e-5/checkpoint_epoch_0016.hdf5",
            ),
            self._score(
                "partial_ft",
                final.LOCKED_PARTIAL_FT_CANDIDATE_ID,
                3e-5,
                0.12,
                final.LOCKED_PARTIAL_FT_EPOCH,
                final.LOCKED_PARTIAL_FT_SHA256,
                self.paths["partial_ft"],
            ),
        ]
        return {
            "protocol_version": FORMAL_PROTOCOL_VERSION,
            "experiment_id": final.LOCKED_EXPERIMENT_ID,
            "run_id": final.LOCKED_RUN_ID,
            "git_head": final.SELECTION_GIT_HEAD,
            "target_protocol_fingerprint": target_protocol_fingerprint("B"),
            "selection_timestamp": "2099-12-31T23:59:59Z",
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
            "selection_basis": ["lowest_validation_loss"],
            "source_validation_comparison": source,
            "wotl_validation_comparison": wotl,
            "partial_ft_validation_comparison": partial,
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
            "protocol_version": FORMAL_PROTOCOL_VERSION,
            "experiment_id": final.LOCKED_EXPERIMENT_ID,
            "run_id": final.LOCKED_RUN_ID,
            "git": {"head": final.SELECTION_GIT_HEAD, "branch": "unit-test", "dirty": False},
            "target_protocol_fingerprint": target_protocol_fingerprint("B"),
            "formal_eligible": True,
            "dry_run": False,
            "config_locked": True,
            "selection_locked": True,
            "checkpoint_locked": True,
            "test_accessed": False,
            "test_authorized": False,
            "test_metrics_used_for_selection": False,
            "post_test_tuning_allowed": False,
            "protocol_stage": ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        }

    def _write_contracts(self):
        (self.run_root / "selection/selection.json").write_text(
            json.dumps(self.selection), encoding="utf-8"
        )
        (self.run_root / "protocol_manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def _rewrite_selection(self):
        (self.run_root / "selection/selection.json").write_text(
            json.dumps(self.selection), encoding="utf-8"
        )

    def _rewrite_manifest(self):
        (self.run_root / "protocol_manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def _valid_sha(self, path):
        parent = Path(path).parent.name
        return {
            final.LOCKED_SOURCE_CANDIDATE_ID: final.LOCKED_SOURCE_SHA256,
            final.LOCKED_WOTL_CANDIDATE_ID: final.LOCKED_WOTL_SHA256,
            final.LOCKED_PARTIAL_FT_CANDIDATE_ID: final.LOCKED_PARTIAL_FT_SHA256,
        }[parent]

    def _authorize(self, **overrides):
        arguments = {
            "human_authorized": True,
            "expected_execution_git_head": self.EXECUTION_GIT_HEAD,
            "run_root": self.run_root,
            "git_state": self.good_git,
            "authorization_timestamp": "2099-12-31T23:59:59Z",
            "sha256_func": self._valid_sha,
            "selection_validator": lambda selection: None,
            "_allow_test_root": True,
        }
        arguments.update(overrides)
        return final.prepare_final_test_authorization(**arguments)

    def _test_data(self):
        y = np.linspace(0.1, 0.9, final.EXPECTED_TEST_SEQUENCES)
        return final.TargetTestData(
            X_test=np.zeros(final.EXPECTED_X_TEST_SHAPE, dtype=np.float32),
            y_test=y,
            target_timestamp=np.arange(final.EXPECTED_TEST_SEQUENCES),
            target_scaler=FakeScaler(),
            accessed_files=(Path(self.temporary.name) / "fake_target_data",),
        )

    def _metric(self, mae, mse, rmse, r2):
        return final.OriginalScaleMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            prediction_count=3,
            negative_prediction_count=0,
        )

    def _assert_tampered_authorization_blocked(self, authorization):
        model_calls = []
        loader_calls = []
        output = self.run_root / "final_test"
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaises(final.FinalTestError):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=lambda: loader_calls.append(1),
                    model_loader=lambda path: model_calls.append(path),
                    model_validator=lambda model: None,
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertEqual(model_calls, [])
        self.assertEqual(loader_calls, [])
        self.assertFalse(output.exists())

    def test_01_import_does_not_touch_real_formal_run(self):
        before = self._filesystem_snapshot(final.LOCKED_RUN_ROOT)
        importlib.reload(final)
        after = self._filesystem_snapshot(final.LOCKED_RUN_ROOT)
        self.assertEqual(after, before)

    def test_02_prepare_authorization_does_not_load_test_or_create_output(self):
        before = self._filesystem_snapshot(self.run_root)
        authorization = self._authorize()
        self.assertEqual(self._filesystem_snapshot(self.run_root), before)
        self.assertEqual(authorization.test_access_count, 0)
        self.assertFalse(authorization.test_accessed)
        self.assertFalse((self.run_root / "final_test").exists())

    def test_02a_selection_and_execution_git_provenance_are_separate(self):
        self.assertEqual(
            final.SELECTION_GIT_HEAD,
            "ec6e881cf95f03bedd81adff10e55151a8b26ff5",
        )
        self.assertNotEqual(self.EXECUTION_GIT_HEAD, final.SELECTION_GIT_HEAD)
        authorization = self._authorize()
        self.assertEqual(authorization.selection_git_head, final.SELECTION_GIT_HEAD)
        self.assertEqual(authorization.execution_git_head, self.EXECUTION_GIT_HEAD)

    def test_02b_execution_git_head_is_an_explicit_parameter(self):
        signature = inspect.signature(final.prepare_final_test_authorization)
        parameter = signature.parameters["expected_execution_git_head"]
        self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(final.FinalTestError):
            self._authorize(expected_execution_git_head="2" * 40)
        source = inspect.getsource(final.prepare_final_test_authorization)
        self.assertIn("git_state is None", source)
        self.assertIn("collect_git_state()", source)

    def test_03_wrong_experiment_blocked(self):
        with self.assertRaises(final.FinalTestError):
            self._authorize(experiment_id="A2")

    def test_04_wrong_run_id_blocked(self):
        with self.assertRaises(final.FinalTestError):
            self._authorize(run_id="20991231T235959Z_seed1234")

    def test_05_wrong_git_head_blocked(self):
        with self.assertRaises(final.FinalTestError):
            self._authorize(git_state=final.GitState("0" * 40, False))

    def test_06_tracked_dirty_blocked(self):
        with self.assertRaises(final.FinalTestError):
            self._authorize(git_state=final.GitState(self.EXECUTION_GIT_HEAD, True))

    def test_06a_execute_time_git_head_mutation_is_blocked_before_all_actions(self):
        authorization = self._authorize()
        model_calls = []
        loader_calls = []
        output = self.run_root / "final_test"
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(final.FinalTestError, "execute-time Git HEAD"):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=lambda: loader_calls.append(1),
                    model_loader=lambda path: model_calls.append(path),
                    git_state_provider=lambda: final.GitState("2" * 40, False),
                    _allow_test_dependencies=True,
                )
        self.assertEqual(model_calls, [])
        self.assertEqual(loader_calls, [])
        self.assertFalse(output.exists())

    def test_06b_execute_time_tracked_dirty_is_blocked_before_all_actions(self):
        authorization = self._authorize()
        model_calls = []
        loader_calls = []
        output = self.run_root / "final_test"
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(final.FinalTestError, "became dirty"):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=lambda: loader_calls.append(1),
                    model_loader=lambda path: model_calls.append(path),
                    git_state_provider=lambda: final.GitState(
                        self.EXECUTION_GIT_HEAD, True
                    ),
                    _allow_test_dependencies=True,
                )
        self.assertEqual(model_calls, [])
        self.assertEqual(loader_calls, [])
        self.assertFalse(output.exists())

    def test_07_selection_not_locked_blocked(self):
        self.selection["selection_locked"] = False
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_08_comparison_pair_unlocked_blocked(self):
        self.selection["comparison_pair_locked"] = False
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_09_previous_test_access_blocked(self):
        self.selection["test_accessed"] = True
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_10_already_authorized_blocked(self):
        self.selection["test_authorized"] = True
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_11_wotl_candidate_mismatch_blocked(self):
        self.selection["wotl_selected_candidate"] = "WOTL_lr3e-5"
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_12_partial_candidate_mismatch_blocked(self):
        self.selection["partial_ft_selected_candidate"] = "PFT_lr1e-5"
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_13_source_candidate_mismatch_blocked(self):
        self.selection["source_selected_candidate"] = "SRC_lr3e-5"
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_14_wotl_sha_mismatch_blocked(self):
        self.selection["wotl_selected_checkpoint_sha256"] = "d" * 64
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_15_partial_sha_mismatch_blocked(self):
        self.selection["partial_ft_selected_checkpoint_sha256"] = "e" * 64
        self._rewrite_selection()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_16_actual_checkpoint_sha_mismatch_blocked(self):
        with self.assertRaises(final.FinalTestError):
            self._authorize(sha256_func=lambda path: "f" * 64)

    def test_16a_tampered_source_authorization_is_blocked(self):
        tampered = replace(
            self._authorize(),
            source_candidate_id="SRC_lr3e-5",
            source_checkpoint_sha256="a" * 64,
        )
        self._assert_tampered_authorization_blocked(tampered)

    def test_16b_tampered_wotl_candidate_is_blocked(self):
        tampered = replace(self._authorize(), wotl_candidate_id="WOTL_lr3e-5")
        self._assert_tampered_authorization_blocked(tampered)

    def test_16c_tampered_wotl_path_and_matching_sha_are_blocked(self):
        path = Path(self.temporary.name) / "tampered_wotl.hdf5"
        path.write_bytes(b"self-consistent-tampered-wotl")
        tampered = replace(
            self._authorize(),
            wotl_checkpoint_path=path,
            wotl_checkpoint_sha256="b" * 64,
        )
        self._assert_tampered_authorization_blocked(tampered)

    def test_16d_tampered_partial_candidate_is_blocked(self):
        tampered = replace(self._authorize(), partial_ft_candidate_id="PFT_lr1e-5")
        self._assert_tampered_authorization_blocked(tampered)

    def test_16e_tampered_partial_path_and_matching_sha_are_blocked(self):
        path = Path(self.temporary.name) / "tampered_partial.hdf5"
        path.write_bytes(b"self-consistent-tampered-partial")
        tampered = replace(
            self._authorize(),
            partial_ft_checkpoint_path=path,
            partial_ft_checkpoint_sha256="c" * 64,
        )
        self._assert_tampered_authorization_blocked(tampered)

    def test_17_execute_false_reads_nothing(self):
        authorization = self._authorize()
        calls = []
        result = final.execute_final_test(
            authorization,
            execute_test=False,
            test_loader=lambda: calls.append("loaded"),
        )
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertFalse((self.run_root / "final_test").exists())

    def test_18_missing_environment_opt_in_reads_nothing(self):
        authorization = self._authorize()
        calls = []
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOLAR_RUN_FINAL_TEST", None)
            with self.assertRaises(final.FinalTestError):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=self.run_root / "final_test",
                    test_loader=lambda: calls.append("loaded"),
                )
        self.assertEqual(calls, [])

    def test_19_fake_test_load_exactly_once_and_shared_arrays(self):
        authorization = self._authorize()
        data = self._test_data()
        loader_calls = []
        y = data.y_test
        wotl_model = FakeModel(y + 0.02)
        partial_model = FakeModel(y + 0.01)
        loaded_paths = []

        def loader():
            authorization_payload = json.loads(
                (self.run_root / "final_test/authorization.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                authorization_payload["state"], final.FinalTestState.TEST_AUTHORIZED.value
            )
            self.assertFalse(authorization_payload["test_accessed"])
            self.assertEqual(authorization_payload["test_access_count"], 0)
            started = json.loads(
                (self.run_root / "final_test/final_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(started["state"], final.FinalTestState.TEST_ACCESS_STARTED.value)
            self.assertEqual(started["test_access_attempt_count"], 1)
            self.assertFalse(started["test_accessed"])
            self.assertEqual(started["test_access_count"], 0)
            loader_calls.append(1)
            return data

        def model_loader(path):
            self.assertFalse((self.run_root / "final_test").exists())
            loaded_paths.append(path)
            return wotl_model if "WOTL" in str(path) else partial_model

        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            result = final.execute_final_test(
                authorization,
                execute_test=True,
                output_root=self.run_root / "final_test",
                test_loader=loader,
                model_loader=model_loader,
                model_validator=lambda model: None,
                sha256_func=self._valid_sha,
                git_state_provider=lambda: self.good_git,
                _allow_test_dependencies=True,
            )
        self.assertEqual(loader_calls, [1])
        self.assertIs(wotl_model.predict_calls[0][0], data.X_test)
        self.assertIs(partial_model.predict_calls[0][0], data.X_test)
        self.assertEqual(loaded_paths, [self.paths["wotl"], self.paths["partial_ft"]])
        self.assertEqual(result.authorization.test_access_count, 1)

    def test_20_one_time_loader_blocks_second_read(self):
        access = final.OneTimeTestAccess(self._test_data)
        access.load()
        with self.assertRaises(final.FinalTestError):
            access.load()
        self.assertEqual(access.access_count, 1)

    def test_21_raw_predictions_are_not_clipped(self):
        metrics = final.compute_original_scale_metrics(
            np.array([0.0, 1.0, 2.0]),
            np.array([-1.0, 1.0, 3.0]),
        )
        self.assertEqual(metrics.negative_prediction_count, 1)
        self.assertAlmostEqual(metrics.mae, 2.0 / 3.0)
        self.assertAlmostEqual(metrics.mse, 2.0 / 3.0)

    def test_22_original_scale_metric_computation(self):
        evaluated = final.evaluate_raw_prediction(
            np.array([0.0, 0.5, 1.0]),
            np.array([-0.1, 0.4, 0.8]),
            FakeScaler(),
        )
        np.testing.assert_array_equal(evaluated.prediction_original, [-10.0, 40.0, 80.0])
        self.assertEqual(evaluated.metrics.negative_prediction_count, 1)
        self.assertAlmostEqual(evaluated.metrics.mae, 40.0 / 3.0)

    def test_23_positive_transfer_requires_all_four(self):
        wotl = self._metric(10, 100, 10, 0.7)
        partial = self._metric(9, 81, 9, 0.8)
        self.assertIs(
            final.classify_transfer(wotl, partial),
            final.TransferClassification.POSITIVE_TRANSFER,
        )

    def test_24_partial_positive_transfer(self):
        wotl = self._metric(10, 100, 10, 0.7)
        partial = self._metric(9, 81, 9, 0.7)
        self.assertIs(
            final.classify_transfer(wotl, partial),
            final.TransferClassification.PARTIAL_POSITIVE_TRANSFER,
        )

    def test_25_mixed_result(self):
        wotl = self._metric(10, 100, 10, 0.7)
        partial = self._metric(9, 121, 11, 0.8)
        self.assertIs(
            final.classify_transfer(wotl, partial),
            final.TransferClassification.MIXED_RESULT,
        )

    def test_26_negative_transfer(self):
        wotl = self._metric(10, 100, 10, 0.7)
        partial = self._metric(11, 121, 11, 0.6)
        self.assertIs(
            final.classify_transfer(wotl, partial),
            final.TransferClassification.NEGATIVE_TRANSFER,
        )

    def test_27_completed_run_blocked_before_test_load(self):
        authorization = self._authorize()
        output = self.run_root / "final_test"
        output.mkdir()
        (output / "comparison.json").write_text("{}", encoding="utf-8")
        calls = []
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaises(final.FinalTestError):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=lambda: calls.append(1),
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertEqual(calls, [])

    def test_27a_completed_state_blocks_rerun_before_test_load(self):
        authorization = self._authorize()
        output = self.run_root / "final_test"
        output.mkdir()
        (output / "final_state.json").write_text(
            json.dumps({"state": final.FinalTestState.TEST_COMPLETED.value}),
            encoding="utf-8",
        )
        calls = []
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(final.FinalTestError, "already TEST_COMPLETED"):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=lambda: calls.append(1),
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertEqual(calls, [])

    def test_27b_model_reload_failure_precedes_root_creation_and_test_access(self):
        authorization = self._authorize()
        output = self.run_root / "final_test"
        calls = []
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(RuntimeError, "simulated reload failure"):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=lambda: calls.append(1),
                    model_loader=lambda path: (_ for _ in ()).throw(
                        RuntimeError("simulated reload failure")
                    ),
                    model_validator=lambda model: None,
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertEqual(calls, [])
        self.assertFalse(output.exists())
        self.assertFalse(authorization.test_accessed)
        self.assertEqual(authorization.test_access_count, 0)

    def test_28_manifest_test_completed_blocked(self):
        self.manifest["protocol_stage"] = final.FinalTestState.TEST_COMPLETED.value
        self._rewrite_manifest()
        with self.assertRaises(final.FinalTestError):
            self._authorize()

    def test_29_outputs_are_temp_only_and_complete(self):
        real_before = self._filesystem_snapshot(final.LOCKED_RUN_ROOT)
        authorization = self._authorize()
        data = self._test_data()
        models = [FakeModel(data.y_test + 0.02), FakeModel(data.y_test + 0.01)]
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            result = final.execute_final_test(
                authorization,
                execute_test=True,
                output_root=self.run_root / "final_test",
                test_loader=lambda: data,
                model_loader=lambda path: models.pop(0),
                model_validator=lambda model: None,
                sha256_func=self._valid_sha,
                git_state_provider=lambda: self.good_git,
                _allow_test_dependencies=True,
            )
        self.assertTrue(result.output_root.is_relative_to(Path(self.temporary.name)))
        self.assertEqual(
            {path.name for path in result.output_root.iterdir()},
            {
                "authorization.json",
                "final_state.json",
                "wotl_metrics_original_scale.json",
                "partial_ft_metrics_original_scale.json",
                "comparison.json",
                "access_log.json",
                "predictions.csv",
            },
        )
        self.assertEqual(self._filesystem_snapshot(final.LOCKED_RUN_ROOT), real_before)

        authorization_payload = json.loads(
            (result.output_root / "authorization.json").read_text(encoding="utf-8")
        )
        self.assertEqual(authorization_payload["state"], final.FinalTestState.TEST_AUTHORIZED.value)
        self.assertFalse(authorization_payload["test_accessed"])
        self.assertEqual(authorization_payload["test_access_count"], 0)
        completed = json.loads(
            (result.output_root / "final_state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(completed["state"], final.FinalTestState.TEST_COMPLETED.value)
        self.assertTrue(completed["test_accessed"])
        self.assertTrue(completed["test_completed"])
        self.assertEqual(completed["test_access_count"], 1)
        self.assertEqual(completed["test_access_attempt_count"], 1)
        self.assertFalse(completed["post_test_tuning_allowed"])
        comparison = json.loads(
            (result.output_root / "comparison.json").read_text(encoding="utf-8")
        )
        self.assertEqual(comparison["metrics_scale"], "original")
        self.assertEqual(comparison["target_name"], "DC_POWER")
        self.assertIn("documented unit: kW", comparison["unit"])
        self.assertEqual(comparison["selection_git_head"], final.SELECTION_GIT_HEAD)
        self.assertEqual(comparison["execution_git_head"], self.EXECUTION_GIT_HEAD)

    def test_29a_prediction_failure_is_durably_sealed_and_rerun_is_blocked(self):
        authorization = self._authorize()
        data = self._test_data()
        loader_calls = []
        output = self.run_root / "final_test"

        def loader():
            loader_calls.append(1)
            return data

        models = [FailingPredictionModel(data.y_test), FakeModel(data.y_test)]
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(final.FinalTestError, "durable Test access") as raised:
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=loader,
                    model_loader=lambda path: models.pop(0),
                    model_validator=lambda model: None,
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(str(raised.exception.__cause__), "simulated prediction failure")
        self.assertEqual(loader_calls, [1])
        incomplete = json.loads((output / "final_state.json").read_text(encoding="utf-8"))
        self.assertEqual(
            incomplete["state"], final.FinalTestState.TEST_ACCESSED_INCOMPLETE.value
        )
        self.assertTrue(incomplete["test_accessed"])
        self.assertEqual(incomplete["test_access_count"], 1)
        self.assertEqual(incomplete["test_access_attempt_count"], 1)
        self.assertFalse(incomplete["test_completed"])
        self.assertFalse(incomplete["post_test_tuning_allowed"])
        self.assertIn("test_access_timestamp", incomplete)

        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(final.FinalTestError, "re-run prohibited"):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=loader,
                    model_loader=lambda path: FakeModel(data.y_test),
                    model_validator=lambda model: None,
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertEqual(loader_calls, [1])

    def test_29c_loader_failure_preserves_access_started_and_blocks_rerun(self):
        authorization = self._authorize()
        output = self.run_root / "final_test"
        loader_calls = []
        model_calls = []

        def loader():
            loader_calls.append(1)
            started = json.loads((output / "final_state.json").read_text(encoding="utf-8"))
            self.assertEqual(started["state"], final.FinalTestState.TEST_ACCESS_STARTED.value)
            raise RuntimeError("simulated test loader failure")

        def model_loader(path):
            model_calls.append(path)
            return FakeModel(np.zeros(final.EXPECTED_TEST_SEQUENCES))

        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(final.FinalTestError, "TEST_ACCESS_STARTED") as raised:
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=loader,
                    model_loader=model_loader,
                    model_validator=lambda model: None,
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(str(raised.exception.__cause__), "simulated test loader failure")
        self.assertEqual(loader_calls, [1])
        self.assertEqual(model_calls, [self.paths["wotl"], self.paths["partial_ft"]])
        self.assertTrue((output / "authorization.json").is_file())
        started = json.loads((output / "final_state.json").read_text(encoding="utf-8"))
        self.assertEqual(started["state"], final.FinalTestState.TEST_ACCESS_STARTED.value)
        self.assertEqual(started["test_access_attempt_count"], 1)
        self.assertFalse(started["test_accessed"])
        self.assertEqual(started["test_access_count"], 0)
        self.assertFalse(started["test_completed"])
        self.assertFalse(started["post_test_tuning_allowed"])
        self.assertIn("test_access_start_timestamp", started)

        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaisesRegex(final.FinalTestError, "pristine Test status"):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=output,
                    test_loader=loader,
                    model_loader=model_loader,
                    model_validator=lambda model: None,
                    sha256_func=self._valid_sha,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertEqual(loader_calls, [1])
        self.assertEqual(model_calls, [self.paths["wotl"], self.paths["partial_ft"]])

    def test_29b_original_scale_metadata_is_explicit(self):
        metrics = final.compute_original_scale_metrics(
            np.array([0.0, 1.0, 2.0]), np.array([-1.0, 1.0, 3.0])
        )
        self.assertEqual(metrics.target_name, "DC_POWER")
        self.assertEqual(metrics.scale, "original")
        self.assertEqual(
            metrics.unit, "DC_POWER (raw dataset scale; documented unit: kW)"
        )

    def test_30_final_module_contains_no_fit_call(self):
        source = inspect.getsource(final)
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("fit", called_attributes)
        self.assertNotIn(".fit(", source)
        signature = inspect.signature(final.execute_final_test)
        self.assertIs(
            signature.parameters["git_state_provider"].default,
            final.collect_git_state,
        )
        execute_source = inspect.getsource(final.execute_final_test)
        self.assertIn("git_state_provider is collect_git_state", execute_source)

    def test_31_only_target_test_loader_is_exposed(self):
        source = inspect.getsource(final.load_locked_target_test)
        self.assertIn("target_profile_path", source)
        self.assertNotIn("source_profile_path", source)
        self.assertNotIn("source_profile", source)
        module_source = inspect.getsource(final)
        self.assertEqual(module_source.count('solar_data.load_split(metadata, "test")'), 1)

    def test_32_post_test_tuning_is_always_false(self):
        authorization = self._authorize()
        self.assertFalse(authorization.post_test_tuning_allowed)
        self.assertIs(authorization.state, final.FinalTestState.TEST_AUTHORIZED)

    def test_33_checkpoint_revalidated_before_test_load(self):
        authorization = self._authorize()
        calls = []
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            with self.assertRaises(final.FinalTestError):
                final.execute_final_test(
                    authorization,
                    execute_test=True,
                    output_root=self.run_root / "final_test",
                    test_loader=lambda: calls.append(1),
                    sha256_func=lambda path: "0" * 64,
                    git_state_provider=lambda: self.good_git,
                    _allow_test_dependencies=True,
                )
        self.assertEqual(calls, [])

    def test_34_target_test_shape_contract(self):
        bad = final.TargetTestData(
            X_test=np.zeros((1, 5, 5)),
            y_test=np.zeros(1),
            target_timestamp=np.zeros(1),
            target_scaler=FakeScaler(),
        )
        with self.assertRaises(final.FinalTestError):
            final.validate_target_test_data(bad)

    def test_35_human_authorization_is_required(self):
        with self.assertRaises(final.FinalTestError):
            self._authorize(human_authorized=False)

    def test_36_sealed_result_disallows_post_test_tuning(self):
        authorization = self._authorize()
        data = self._test_data()
        models = [FakeModel(data.y_test + 0.02), FakeModel(data.y_test + 0.01)]
        with patch.dict(os.environ, {"SOLAR_RUN_FINAL_TEST": "1"}):
            result = final.execute_final_test(
                authorization,
                execute_test=True,
                output_root=self.run_root / "final_test",
                test_loader=lambda: data,
                model_loader=lambda path: models.pop(0),
                model_validator=lambda model: None,
                sha256_func=self._valid_sha,
                git_state_provider=lambda: self.good_git,
                _allow_test_dependencies=True,
            )
        self.assertTrue(result.authorization.test_accessed)
        self.assertTrue(result.authorization.test_completed)
        self.assertEqual(result.authorization.test_access_count, 1)
        self.assertFalse(result.authorization.post_test_tuning_allowed)
        self.assertIs(result.authorization.state, final.FinalTestState.TEST_COMPLETED)


if __name__ == "__main__":
    unittest.main()
