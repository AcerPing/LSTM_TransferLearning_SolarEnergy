"""Zero-epoch contracts for the R2.4 Experiment-A formal runner."""

from __future__ import annotations

import inspect
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import r2_solar
import r2_helpers.solar_formal as formal
from r2_config.solar_r2 import EXPECTED_ROW_COUNTS, EXPECTED_SEQUENCE_COUNTS, EXPERIMENTS
from r2_helpers.solar_data import ProfileMetadata, ScalerBundle, SequenceData, SplitData
from r2_helpers.solar_runtime import NamedSequenceSplit, checkpoint_paths, make_r2_callbacks


class SolarR2FormalContractTests(unittest.TestCase):
    @staticmethod
    def _minimal_manifest(status="RUNNING"):
        value = {name: {} for name in formal.REQUIRED_MANIFEST_SECTIONS}
        value["residual_definition"] = "y_true_original - y_pred_original"
        value["status"] = status
        value["timestamps"] = {"start": "x", "end": None}
        value["failure"] = None
        return value

    @staticmethod
    def _formal_split(role="source", split_name="training"):
        profile = EXPERIMENTS["A"][role]["profile_dir"]
        rows = EXPECTED_ROW_COUNTS[profile][split_name]
        count = EXPECTED_SEQUENCE_COUNTS[profile][split_name]
        timestamps = np.arange(rows)
        split_data = SplitData(
            name=split_name,
            timestamps=timestamps,
            X_normalized=np.zeros((rows, 5)),
            y_normalized=np.zeros(rows),
            y_original=np.zeros(rows),
            normalized_frame=pd.DataFrame(),
            original_frame=pd.DataFrame(),
        )
        sequences = SequenceData(
            X_seq=np.zeros((count, 5, 5)),
            y_seq=np.zeros(count),
            sample_index=np.arange(count),
            target_row_index=np.arange(count) + 5,
            target_timestamp=timestamps[np.arange(count) + 5],
        )
        return formal.FormalSequenceSplit(
            named=NamedSequenceSplit(split_name, sequences.X_seq, sequences.y_seq),
            split_data=split_data,
            sequences=sequences,
        )

    @staticmethod
    def _prediction_fixture():
        original = np.arange(8, dtype=np.float64) * 10.0
        scaler = MinMaxScaler().fit(original[:5].reshape(-1, 1))
        indices = np.array([5, 6, 7])
        normalized = scaler.transform(original[indices].reshape(-1, 1)).reshape(-1)
        split_data = SplitData(
            name="test",
            timestamps=np.arange(8),
            X_normalized=np.zeros((8, 5)),
            y_normalized=scaler.transform(original.reshape(-1, 1)).reshape(-1),
            y_original=original,
            normalized_frame=pd.DataFrame(),
            original_frame=pd.DataFrame(),
        )
        sequences = SequenceData(
            X_seq=np.zeros((3, 5, 5)),
            y_seq=normalized,
            sample_index=np.arange(3),
            target_row_index=indices,
            target_timestamp=np.array(["t5", "t6", "t7"]),
        )
        test = formal.FormalSequenceSplit(
            named=NamedSequenceSplit("test", sequences.X_seq, sequences.y_seq),
            split_data=split_data,
            sequences=sequences,
        )
        return test, scaler

    @classmethod
    def _pass_result(cls, root):
        run_root = Path(root)
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = cls._minimal_manifest(status="PASS")
        manifest["identity"] = {
            "run_id": "20260814T010203Z_seed1234",
            "source": "Plant1/source_profile",
            "target": "Plant2/target_profile",
        }
        manifest["metrics"] = {
            method: {
                "original": {
                    "MAE": 1.0 + index,
                    "MSE": 4.0 + index,
                    "RMSE": 2.0 + index,
                    "R2": 0.9 - index * 0.1,
                    "n_samples": 2607,
                }
            }
            for index, method in enumerate(formal.R2_METHOD_NAMES)
        }
        manifest["transfer_comparison"] = {
            "tl_freeze": {
                "delta_rmse": -0.5,
                "improvement_percent": 25.0,
                "label": "observed_positive",
            },
            "tl_full_finetune": {
                "delta_rmse": 0.25,
                "improvement_percent": -12.5,
                "label": "observed_negative",
            },
        }
        manifest["lifecycle"] = {
            method: {"best_epoch": index + 1, "epochs_completed": index + 2}
            for index, method in enumerate(formal.FORMAL_METHODS)
        }
        manifest_path = run_root / "run_manifest.json"
        formal.write_manifest_atomic(manifest_path, manifest)
        (run_root / "target_comparison.csv").write_text(
            "Method,MAE,MSE,RMSE,R2,Best_Epoch,Epochs_Completed\n",
            encoding="utf-8",
        )
        return formal.FormalRunResult(run_root=run_root, manifest_path=manifest_path)

    def test_01_experiment_a_mapping(self):
        self.assertEqual(EXPERIMENTS["A"]["source"]["manifest_plant"], "Plant1")
        self.assertEqual(EXPERIMENTS["A"]["target"]["manifest_plant"], "Plant2")
        with patch.dict(os.environ, {"PYTHONHASHSEED": "1234"}):
            formal.validate_formal_request("A", "cpu", 1234)

    def test_02_experiment_b_rejected_by_formal_contract(self):
        with patch.dict(os.environ, {"PYTHONHASHSEED": "1234"}):
            with self.assertRaises(formal.FormalContractError):
                formal.validate_formal_request("B", "cpu", 1234)

    def test_03_cpu_only_bootstrap_contract(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            r2_solar._apply_device_policy("cpu")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
            with self.assertRaises(ValueError):
                r2_solar._apply_device_policy("gpu")

    def test_04_pythonhashseed_gate(self):
        with patch.dict(os.environ, {"PYTHONHASHSEED": "1234"}):
            r2_solar._validate_hash_seed(1234)
        with patch.dict(os.environ, {"PYTHONHASHSEED": "7"}):
            with self.assertRaises(ValueError):
                r2_solar._validate_hash_seed(1234)

    def test_05_clean_git_gate(self):
        formal.require_clean_git({"dirty": False, "critical_dirty": False})
        formal.require_clean_git(
            {"dirty": True, "critical_dirty": False, "unrelated_dirty": True}
        )
        with self.assertRaises(formal.FormalContractError):
            formal.require_clean_git(
                {
                    "dirty": True,
                    "critical_dirty": True,
                    "critical_dirty_paths": ["r2_helpers/solar_formal.py"],
                }
            )

    def test_05a_git_status_clean_classification(self):
        result = formal.classify_git_status([])
        self.assertFalse(result["dirty"])
        self.assertFalse(result["critical_dirty"])
        self.assertFalse(result["unrelated_dirty"])

    def test_05b_unrelated_untracked_file_does_not_block(self):
        lines = ["?? tools/Clean-CodexRefs.bat"]
        result = formal.classify_git_status(lines)
        self.assertTrue(result["dirty"])
        self.assertFalse(result["critical_dirty"])
        self.assertTrue(result["unrelated_dirty"])
        self.assertEqual(result["status_porcelain"], lines)
        formal.require_clean_git(result)

    def test_05c_modified_formal_helper_blocks(self):
        result = formal.classify_git_status([" M r2_helpers/solar_formal.py"])
        self.assertTrue(result["critical_dirty"])
        self.assertEqual(result["critical_dirty_paths"], ["r2_helpers/solar_formal.py"])
        with self.assertRaises(formal.FormalContractError):
            formal.require_clean_git(result)

    def test_05d_modified_entrypoint_blocks(self):
        result = formal.classify_git_status(["M  r2_solar.py"])
        with self.assertRaises(formal.FormalContractError):
            formal.require_clean_git(result)

    def test_05e_modified_legacy_model_blocks(self):
        result = formal.classify_git_status(["MM utils/model.py"])
        with self.assertRaises(formal.FormalContractError):
            formal.require_clean_git(result)

    def test_05f_untracked_formal_helper_blocks(self):
        result = formal.classify_git_status(["?? r2_helpers/solar_formal.py"])
        with self.assertRaises(formal.FormalContractError):
            formal.require_clean_git(result)

    def test_05g_mixed_status_blocks_and_preserves_unrelated(self):
        lines = [
            "?? tools/Clean-CodexRefs.bat",
            " M r2_helpers/solar_runtime.py",
        ]
        result = formal.classify_git_status(lines)
        self.assertTrue(result["critical_dirty"])
        self.assertTrue(result["unrelated_dirty"])
        self.assertEqual(result["status_porcelain"], lines)
        with self.assertRaises(formal.FormalContractError):
            formal.require_clean_git(result)

    def test_05h_staged_renamed_and_quoted_status_parsing(self):
        lines = [
            'R  "notes/old file.md" -> "notes/new file.md"',
            'A  "r2_helpers/solar_formal.py"',
            " D README.md",
        ]
        result = formal.classify_git_status(lines)
        self.assertTrue(result["critical_dirty"])
        self.assertEqual(result["critical_dirty_paths"], ["r2_helpers/solar_formal.py"])
        self.assertEqual(
            result["unrelated_dirty_paths"],
            ["README.md", "notes/new file.md", "notes/old file.md"],
        )

    def test_05i_renamed_critical_path_blocks(self):
        result = formal.classify_git_status(
            ["R  r2_helpers/solar_formal.py -> archive/solar_formal.py"]
        )
        self.assertTrue(result["critical_dirty"])
        with self.assertRaises(formal.FormalContractError):
            formal.require_clean_git(result)

    def test_05j_critical_file_list_and_porcelain_columns(self):
        self.assertEqual(
            formal.EXPERIMENT_CRITICAL_PATHS,
            (
                "r2_solar.py",
                "r2_config/solar_r2.py",
                "r2_helpers/solar_data.py",
                "r2_helpers/solar_runtime.py",
                "r2_helpers/solar_formal.py",
                "utils/model.py",
            ),
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=" M r2_solar.py\n",
            stderr="",
        )
        with patch.object(formal.subprocess, "run", return_value=completed) as run:
            self.assertEqual(formal._git(Path.cwd(), "status"), " M r2_solar.py")
        command = run.call_args.args[0]
        self.assertNotIn("core.quotePath=false", command)
        self.assertIn("core.quotePath=true", command)

    def test_05k_git_octal_utf8_path_round_trip(self):
        expected = "notebook/20260814執行資料紀錄/紀錄.txt"
        quoted = '"' + "".join(
            chr(byte)
            if 32 <= byte < 127 and chr(byte) not in ('"', "\\")
            else f"\\{byte:03o}"
            for byte in expected.encode("utf-8")
        ) + '"'
        self.assertEqual(
            formal.parse_git_status_paths(f"?? {quoted}"),
            (expected,),
        )

    def test_05l_chinese_untracked_path_is_unrelated_dirty(self):
        expected = "notebook/20260814執行資料紀錄/紀錄.txt"
        quoted = '"' + "".join(
            chr(byte)
            if 32 <= byte < 127 and chr(byte) not in ('"', "\\")
            else f"\\{byte:03o}"
            for byte in expected.encode("utf-8")
        ) + '"'
        lines = [f"?? {quoted}", " M r2_solar.py"]
        result = formal.classify_git_status(lines)
        self.assertTrue(result["unrelated_dirty"])
        self.assertIn(expected, result["unrelated_dirty_paths"])
        self.assertTrue(result["critical_dirty"])
        self.assertEqual(result["critical_dirty_paths"], ["r2_solar.py"])
        self.assertEqual(result["status_porcelain"], lines)

    def test_05m_quoted_unicode_rename_and_copy_parsing(self):
        old_path = "notebook/舊紀錄.txt"
        new_path = "notebook/新紀錄.txt"

        def quote(path):
            return '"' + "".join(
                chr(byte)
                if 32 <= byte < 127 and chr(byte) not in ('"', "\\")
                else f"\\{byte:03o}"
                for byte in path.encode("utf-8")
            ) + '"'

        for status in ("R ", "C "):
            with self.subTest(status=status):
                self.assertEqual(
                    formal.parse_git_status_paths(
                        f"{status} {quote(old_path)} -> {quote(new_path)}"
                    ),
                    (old_path, new_path),
                )

    def test_05n_git_missing_capture_has_clear_diagnostic(self):
        completed = SimpleNamespace(returncode=0, stdout=None, stderr=None)
        with patch.object(formal.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(
                formal.FormalContractError,
                "no captured stdout",
            ):
                formal._git(Path.cwd(), "status")

    def test_05o_git_decode_failure_preserves_original_cause(self):
        decode_error = UnicodeDecodeError("cp950", b"\xe5", 0, 1, "illegal byte")
        with patch.object(formal.subprocess, "run", side_effect=decode_error):
            with self.assertRaisesRegex(
                formal.FormalContractError,
                "Git output decoding failed",
            ) as caught:
                formal._git(Path.cwd(), "status")
        self.assertIs(caught.exception.__cause__, decode_error)

    def test_06_run_id_format(self):
        run_id = formal.generate_run_id(
            1234, datetime(2026, 8, 13, 1, 2, 3, tzinfo=timezone.utc)
        )
        self.assertEqual(run_id, "20260813T010203Z_seed1234")
        formal.validate_run_id(run_id)
        with self.assertRaises(formal.FormalContractError):
            formal.validate_run_id("../bad")

    def test_07_output_root_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            approved = Path(temporary) / "R2" / "Experiment_A"
            with patch.object(formal, "FORMAL_OUTPUT_BASE", approved):
                self.assertEqual(formal.validate_output_base(approved), approved.resolve())
                with self.assertRaises(formal.FormalContractError):
                    formal.validate_output_base(Path(temporary) / "outside")

    def test_08_existing_run_root_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            approved = Path(temporary) / "R2" / "Experiment_A"
            run_id = "20260813T010203Z_seed1234"
            with patch.object(formal, "FORMAL_OUTPUT_BASE", approved):
                root = formal.create_formal_run_root(run_id, output_base=approved)
                self.assertTrue(root.is_dir())
                with self.assertRaises(formal.FormalContractError):
                    formal.create_formal_run_root(run_id, output_base=approved)

    def test_09_sequence_expected_count_contract(self):
        split = self._formal_split("source", "training")
        formal.validate_sequence_contract("source", split)
        self.assertEqual(split.sequences.count, 2083)
        self.assertEqual(split.sequences.X_seq.shape, (2083, 5, 5))

    def test_10_formal_method_names(self):
        self.assertEqual(
            formal.FORMAL_METHODS,
            ("source_pretrain", "without_tl", "tl_freeze", "tl_full_finetune"),
        )
        self.assertNotIn("tl_unfreeze", formal.FORMAL_METHODS)

    def test_11_max_epochs_contract(self):
        self.assertEqual(formal.FORMAL_MAX_EPOCHS, 500)
        self.assertEqual(formal.R2_BATCH_SIZE, 128)

    def test_12_callback_protocol_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            callbacks = make_r2_callbacks(Path(temporary) / "best_model.hdf5")
        reduce_lr, checkpoint, early = callbacks
        self.assertEqual((reduce_lr.monitor, reduce_lr.factor, reduce_lr.patience, reduce_lr.min_lr), ("val_loss", 0.5, 4, 1e-7))
        self.assertEqual(checkpoint.monitor, "val_loss")
        self.assertTrue(checkpoint.save_best_only)
        self.assertFalse(checkpoint.save_weights_only)
        self.assertEqual((early.monitor, early.patience, early.restore_best_weights), ("val_loss", 10, True))

    def test_13_checkpoint_paths_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = checkpoint_paths(temporary).as_dict()
        self.assertEqual(len(paths), 4)
        self.assertEqual(len(set(paths.values())), 4)

    def test_14_training_phase_does_not_load_test(self):
        metadata = ProfileMetadata(Path("profile"), {}, {})
        scalers = ScalerBundle(object(), object())
        calls = []

        def fake_build(meta, bundle, name):
            calls.append(name)
            return self._formal_split("source", name)

        with patch.object(formal, "_validate_profile", return_value=(metadata, scalers, {})), patch.object(
            formal, "_build_formal_split", side_effect=fake_build
        ):
            formal.prepare_role_training_validation("source")
        self.assertEqual(calls, ["training", "validation"])
        self.assertNotIn("test", calls)

    def test_15_test_phase_after_all_training(self):
        events = [
            "fit_end:source_pretrain",
            "fit_end:without_tl",
            "fit_end:tl_freeze",
            "fit_end:tl_full_finetune",
            "reload:source_pretrain",
            "reload:without_tl",
            "reload:tl_freeze",
            "reload:tl_full_finetune",
            "load_test:source",
            "load_test:target",
        ]
        formal.validate_test_isolation_events(events)

    def test_16_reload_before_test_contract(self):
        bad = [
            "fit_end:source_pretrain",
            "fit_end:without_tl",
            "fit_end:tl_freeze",
            "fit_end:tl_full_finetune",
            "load_test:source",
            "reload:source_pretrain",
            "reload:without_tl",
            "reload:tl_freeze",
            "reload:tl_full_finetune",
            "load_test:target",
        ]
        with self.assertRaises(formal.FormalContractError):
            formal.validate_test_isolation_events(bad)

    def test_17_normalized_metrics_calculation(self):
        metrics = formal.compute_metrics(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.4, 0.8]))
        self.assertEqual(metrics["n_samples"], 3)
        self.assertAlmostEqual(metrics["MSE"], (0.0 + 0.01 + 0.04) / 3)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(metrics["MSE"]))

    def test_18_original_metrics_calculation(self):
        metrics = formal.compute_metrics(np.array([10.0, 20.0]), np.array([8.0, 23.0]))
        self.assertAlmostEqual(metrics["MAE"], 2.5)
        self.assertAlmostEqual(metrics["MSE"], 6.5)

    def test_19_inverse_transform_validation(self):
        test, scaler = self._prediction_fixture()
        frame = formal.build_prediction_frame(test, test.sequences.y_seq, scaler)
        np.testing.assert_allclose(frame["y_true_original"], [50.0, 60.0, 70.0])
        np.testing.assert_allclose(frame["y_pred_original"], [50.0, 60.0, 70.0])

    def test_20_prediction_csv_schema(self):
        test, scaler = self._prediction_fixture()
        frame = formal.build_prediction_frame(test, test.sequences.y_seq, scaler)
        self.assertEqual(tuple(frame.columns), formal.PREDICTION_COLUMNS)

    def test_21_residual_direction(self):
        test, scaler = self._prediction_fixture()
        prediction = test.sequences.y_seq - 0.1
        frame = formal.build_prediction_frame(test, prediction, scaler)
        expected = frame["y_true_original"] - frame["y_pred_original"]
        np.testing.assert_array_equal(frame["residual_original"], expected)
        self.assertTrue((frame["residual_original"] > 0).all())

    def test_22_history_schema(self):
        history = SimpleNamespace(history={"loss": [2.0, 1.0], "val_loss": [3.0, 2.0]})
        frame = formal.build_history_frame(history, [1e-4, 5e-5])
        self.assertEqual(tuple(frame.columns), formal.HISTORY_COLUMNS)
        self.assertEqual(frame["epoch"].tolist(), [1, 2])

    def test_23_observer_only_lr_logging(self):
        learning_rate = tf.Variable(1e-4, dtype=tf.float32)
        fake_model = SimpleNamespace(optimizer=SimpleNamespace(learning_rate=learning_rate))
        observer = formal.LearningRateObserver()
        observer.set_model(fake_model)
        before = float(learning_rate.numpy())
        observer.on_epoch_begin(0, {})
        self.assertEqual(float(learning_rate.numpy()), before)
        learning_rate.assign(5e-5)
        changed = float(learning_rate.numpy())
        observer.on_epoch_begin(1, {})
        self.assertEqual(float(learning_rate.numpy()), changed)
        self.assertEqual(len(observer.learning_rates), 2)
        self.assertAlmostEqual(observer.learning_rates[0], before)
        self.assertAlmostEqual(observer.learning_rates[1], changed)

    def test_24_best_epoch_calculation(self):
        frame = pd.DataFrame(
            {"epoch": [1, 2, 3], "loss": [3.0, 2.0, 1.0], "val_loss": [2.0, 0.5, 1.0], "learning_rate": [1e-4] * 3},
            columns=formal.HISTORY_COLUMNS,
        )
        summary = formal.summarize_history(frame, SimpleNamespace(stopped_epoch=2), max_epochs=500)
        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["best_val_loss"], 0.5)

    def test_25_stop_reason(self):
        frame = pd.DataFrame(
            {"epoch": [1, 2], "loss": [2.0, 1.0], "val_loss": [2.0, 1.0], "learning_rate": [1e-4, 1e-4]},
            columns=formal.HISTORY_COLUMNS,
        )
        self.assertEqual(formal.summarize_history(frame, SimpleNamespace(stopped_epoch=1))["stop_reason"], "early_stopping")
        full = pd.concat([frame.iloc[[0]]] * 500, ignore_index=True)
        full["epoch"] = np.arange(1, 501)
        self.assertEqual(formal.summarize_history(full, SimpleNamespace(stopped_epoch=0))["stop_reason"], "max_epochs")

    def test_26_target_comparison_schema(self):
        evaluations = {}
        lifecycles = {}
        for index, method in enumerate(formal.R2_METHOD_NAMES):
            metrics = {"MAE": 1.0, "MSE": 2.0, "RMSE": 3.0 + index, "R2": 0.5, "n_samples": 3}
            evaluations[method] = formal.EvaluationResult(method, metrics, metrics, pd.DataFrame(), {})
            lifecycles[method] = SimpleNamespace(summary={"best_epoch": 2, "epochs_completed": 12})
        frame = formal.build_target_comparison(evaluations, lifecycles)
        self.assertEqual(tuple(frame.columns), formal.COMPARISON_COLUMNS)
        self.assertEqual(frame["Method"].tolist(), list(formal.R2_METHOD_NAMES))

    def test_27_rmse_transfer_rule(self):
        self.assertEqual(formal.transfer_interpretation(10.0, 8.0)["label"], "observed_positive")
        self.assertEqual(formal.transfer_interpretation(10.0, 12.0)["label"], "observed_negative")
        self.assertEqual(formal.transfer_interpretation(10.0, 10.0)["label"], "neutral")
        self.assertFalse(formal.transfer_interpretation(10.0, 8.0)["statistical_significance_claimed"])

    def test_28_manifest_required_fields(self):
        manifest = self._minimal_manifest()
        formal.validate_manifest_schema(manifest)
        del manifest["protocol"]
        with self.assertRaises(formal.FormalContractError):
            formal.validate_manifest_schema(manifest)

    def test_29_checkpoint_artifact_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(b"solar-r2")
            self.assertEqual(
                formal.sha256_file(path),
                "683dcf990976ecdb7db3b6a53bf9626dc19aafc5b8e052042b9be6b7f2d99af4",
            )

    def test_30_failure_manifest_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "run_manifest.json"
            manifest = self._minimal_manifest()
            try:
                raise ValueError("planned failure")
            except ValueError as exc:
                formal.mark_manifest_failed(manifest, failed_stage="unit_test", exc=exc)
            formal.write_manifest_atomic(path, manifest)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["status"], "FAIL")
            self.assertEqual(stored["failure"]["failed_stage"], "unit_test")
            self.assertEqual(stored["failure"]["type"], "ValueError")
            self.assertTrue(path.exists())

    def test_31_no_overwrite_contract(self):
        signature = inspect.signature(formal.create_formal_run_root)
        self.assertNotIn("force", signature.parameters)
        self.assertNotIn("overwrite", signature.parameters)

    def test_32_no_auto_resume_contract(self):
        self.assertNotIn("resume", inspect.signature(formal.run_formal).parameters)
        self.assertNotIn("force", inspect.signature(formal.run_formal).parameters)

    def test_33_formal_run_root_only_write_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            run_root = repo / "reports" / "R2" / "run"
            run_root.mkdir(parents=True)
            formal.audit_git_writes(["?? reports/R2/run/file.txt"], run_root, repo)
            formal.audit_git_writes(
                ["?? tools/Clean-CodexRefs.bat", "?? reports/R2/run/file.txt"],
                run_root,
                repo,
                baseline_status_lines=["?? tools/Clean-CodexRefs.bat"],
            )
            with self.assertRaises(formal.FormalContractError):
                formal.audit_git_writes(["?? outside.txt"], run_root, repo)

    def test_34_keyboard_interrupt_lifecycle_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_base = Path(temporary) / "R2" / "Experiment_A"
            run_id = "20260814T010203Z_seed1234"

            def initial_manifest(*args, **kwargs):
                manifest = self._minimal_manifest()
                manifest["identity"] = {
                    "run_id": run_id,
                    "source": "Plant1/source_profile",
                    "target": "Plant2/target_profile",
                }
                return manifest

            def interrupting_build(*args, **kwargs):
                run_root = Path(kwargs["output_dir"])
                checkpoint = run_root / "source" / "pretrain" / "best_model.hdf5"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"partial-checkpoint")
                raise KeyboardInterrupt()

            provenance = {
                "dirty": False,
                "critical_dirty": False,
                "status_porcelain": [],
            }
            create_formal_run_root = formal.create_formal_run_root
            with (
                patch.object(formal, "FORMAL_OUTPUT_BASE", output_base),
                patch.object(
                    formal,
                    "create_formal_run_root",
                    side_effect=lambda value: create_formal_run_root(
                        value,
                        output_base=output_base,
                    ),
                ),
                patch.object(formal, "validate_formal_request"),
                patch.object(formal.tf.config, "list_physical_devices", return_value=[]),
                patch.object(formal, "collect_git_provenance", return_value=provenance),
                patch.object(formal, "require_clean_git"),
                patch.object(formal, "configure_reproducibility"),
                patch.object(
                    formal,
                    "prepare_role_training_validation",
                    return_value=SimpleNamespace(),
                ),
                patch.object(formal, "_initial_manifest", side_effect=initial_manifest),
                patch.object(formal, "build_r2_model", side_effect=interrupting_build),
            ):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    formal.run_formal(
                        experiment="A",
                        device="cpu",
                        seed=1234,
                        run_id=run_id,
                    )

            run_root = output_base / run_id
            manifest_path = run_root / "run_manifest.json"
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["status"], "FAIL")
            self.assertEqual(stored["failure"]["type"], "KeyboardInterrupt")
            self.assertEqual(
                stored["failure"]["message"],
                "Formal run interrupted by user",
            )
            self.assertEqual(stored["failure"]["failed_stage"], "source_pretrain")
            self.assertIsNotNone(stored["timestamps"]["end"])
            self.assertIsNotNone(stored["timestamps"]["elapsed_seconds"])
            self.assertTrue((run_root / "source/pretrain/best_model.hdf5").is_file())
            self.assertTrue((run_root / "failure_traceback.txt").is_file())
            self.assertFalse((run_root / "target_comparison.csv").exists())
            self.assertEqual(caught.exception.run_root.resolve(), run_root.resolve())
            self.assertEqual(
                caught.exception.manifest_path.resolve(),
                manifest_path.resolve(),
            )

    def test_35_interrupted_console_summary_contains_no_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "20260814T010203Z_seed1234"
            run_root.mkdir()
            manifest = self._minimal_manifest(status="FAIL")
            manifest["identity"] = {"run_id": run_root.name}
            manifest["failure"] = {
                "type": "KeyboardInterrupt",
                "message": "Formal run interrupted by user",
                "failed_stage": "source_pretrain",
            }
            manifest_path = run_root / "run_manifest.json"
            formal.write_manifest_atomic(manifest_path, manifest)
            summary = formal.format_formal_interrupted_summary(run_root)
            self.assertIn("R2 Formal Experiment A INTERRUPTED", summary)
            self.assertIn("source_pretrain", summary)
            self.assertIn(str(manifest_path), summary)
            self.assertNotIn("MAE", summary)
            self.assertNotIn("RMSE", summary)
            self.assertNotIn("MAPE", summary)

    def test_36_pass_console_summary_uses_final_artifacts_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._pass_result(Path(temporary) / "formal")
            with (
                patch.object(formal, "evaluate_lifecycle", side_effect=AssertionError),
                patch.object(formal, "load_formal_test", side_effect=AssertionError),
                patch.object(formal, "compute_metrics", side_effect=AssertionError),
            ):
                summary = formal.format_formal_success_summary(result)
            self.assertIn("R2 Formal Experiment A Completed", summary)
            self.assertIn("Target Test Metrics - Original Scale", summary)
            for metric in ("MAE", "MSE", "RMSE", "R²"):
                self.assertIn(metric, summary)
            self.assertNotIn("MAPE", summary)
            self.assertIn("Primary criterion: Original-scale RMSE", summary)
            self.assertIn("Training Summary", summary)
            self.assertIn(str(result.manifest_path), summary)
            self.assertIn(str(result.run_root / "target_comparison.csv"), summary)

    def test_37_cli_propagates_keyboard_interrupt_after_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "20260814T010203Z_seed1234"
            run_root.mkdir()
            manifest = self._minimal_manifest(status="FAIL")
            manifest["identity"] = {"run_id": run_root.name}
            manifest["failure"] = {
                "type": "KeyboardInterrupt",
                "message": "Formal run interrupted by user",
                "failed_stage": "without_tl",
            }
            manifest_path = run_root / "run_manifest.json"
            formal.write_manifest_atomic(manifest_path, manifest)
            interruption = KeyboardInterrupt()
            interruption.run_root = run_root
            interruption.manifest_path = manifest_path
            interruption.failed_stage = "without_tl"
            args = SimpleNamespace(
                command="formal",
                experiment="A",
                device="cpu",
                seed=1234,
                run_id=run_root.name,
            )
            stderr = io.StringIO()
            with (
                patch.object(r2_solar, "_parse_args", return_value=args),
                patch.object(r2_solar, "_apply_device_policy"),
                patch.object(r2_solar, "_validate_hash_seed"),
                patch.object(formal, "run_formal", side_effect=interruption),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    r2_solar.main()
            self.assertIn("R2 Formal Experiment A INTERRUPTED", stderr.getvalue())
            self.assertIn("without_tl", stderr.getvalue())
            self.assertNotIn("RMSE", stderr.getvalue())

    def test_38_cli_prints_pass_summary_from_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._pass_result(Path(temporary) / "formal")
            args = SimpleNamespace(
                command="formal",
                experiment="A",
                device="cpu",
                seed=1234,
                run_id=None,
            )
            stdout = io.StringIO()
            with (
                patch.object(r2_solar, "_parse_args", return_value=args),
                patch.object(r2_solar, "_apply_device_policy"),
                patch.object(r2_solar, "_validate_hash_seed"),
                patch.object(formal, "run_formal", return_value=result),
                redirect_stdout(stdout),
            ):
                self.assertEqual(r2_solar.main(), 0)
            output = stdout.getvalue()
            self.assertIn("R2 Formal Experiment A Completed", output)
            self.assertIn("R2.4_FORMAL_RUN=PASS", output)
            self.assertNotIn("MAPE", output)


if __name__ == "__main__":
    unittest.main()
