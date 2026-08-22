from __future__ import annotations

import inspect
import importlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from r2_config import solar_linear_b2 as config
from r2_helpers import solar_linear_b2 as b2


class FakeScaler:
    def __init__(self):
        self.inverse_transform_calls = 0

    def inverse_transform(self, values):
        self.inverse_transform_calls += 1
        return np.asarray(values, dtype=np.float64) * 100000.0

    def fit(self, values):
        raise AssertionError("B2 must never fit the sealed target scaler")


class FakePredictionModel:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 1)
        self.predict_calls = []

    def predict(self, X, **kwargs):
        self.predict_calls.append((X, kwargs))
        return self.prediction.copy()


class FakeLayer:
    def __init__(self, index, prefix):
        self.name = f"layer_{index}"
        self.trainable = True
        self._weights = [] if index == 0 else [np.array([prefix * index], dtype=np.float64)]

    def get_weights(self):
        return [weight.copy() for weight in self._weights]

    def set_weights(self, values):
        self._weights = [np.asarray(value).copy() for value in values]


class FakeBuildModel:
    def __init__(self, prefix):
        self.layers = [FakeLayer(index, prefix) for index in range(7)]
        self.compile_calls = []

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)
        self.optimizer = kwargs["optimizer"]
        self.loss = kwargs["loss"]


class FakeFitModel:
    def __init__(self, history=None):
        self.fit_calls = []
        self.checkpoint_pattern = None
        self.history = history or {
            "loss": [0.5, 0.3, 0.4],
            "val_loss": [0.4, 0.2, 0.3],
        }

    def fit(self, X, y, **kwargs):
        self.fit_calls.append((X, y, kwargs))
        checkpoint = Path(str(self.checkpoint_pattern).format(epoch=2))
        checkpoint.write_bytes(b"fake-best-validation-checkpoint")
        return SimpleNamespace(history=self.history)


class SolarLinearB2Tests(unittest.TestCase):
    RUN_ID = "20991231T235959Z_seed1234"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp_root = Path(self.temporary.name)

    @staticmethod
    def _snapshot(root):
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

    @staticmethod
    def _metric(mae, mse, rmse, r2, negative=0):
        return b2.B2Metrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            prediction_count=2607,
            negative_prediction_count=negative,
        )

    def _paths(self):
        return b2.b2_path_contract(
            self.RUN_ID,
            output_base=self.temp_root / "Linear_Supplementary/Experiment_B2",
            _allow_test_output=True,
        )

    def _manifest(self):
        return b2.build_run_manifest(
            run_id=self.RUN_ID,
            execution_git_head="1" * 40,
            tracked_dirty=True,
            untracked_paths=("r2_config/solar_linear_b2.py",),
        )

    def _fake_test_data(self):
        y = np.linspace(0.1, 0.9, config.B2_EXPECTED_TEST_SEQUENCES)
        return b2.B2TestData(
            X_test=np.zeros(config.B2_EXPECTED_X_TEST_SHAPE, dtype=np.float32),
            y_test=y,
            target_timestamp=np.arange(config.B2_EXPECTED_TEST_SEQUENCES),
            target_scaler=FakeScaler(),
        )

    def _build_contract(self, candidate_id):
        from r2_helpers import solar_linear_runtime as runtime

        source = FakeBuildModel(prefix=10.0)
        target = FakeBuildModel(prefix=-10.0)
        layer_params = {1: 60, 2: 17040, 3: 240, 4: 29040, 5: 240, 6: 61}

        def counts(model):
            trainable = sum(
                layer_params[index]
                for index in layer_params
                if model.layers[index].trainable
            )
            return SimpleNamespace(
                total=config.B2_TOTAL_PARAMS,
                trainable=trainable,
                non_trainable=config.B2_TOTAL_PARAMS - trainable,
            )

        with patch.object(runtime, "validate_linear_model", lambda *args, **kwargs: None), patch.object(
            runtime, "build_linear_without_tl_model", return_value=target
        ), patch.object(runtime, "parameter_counts", side_effect=counts):
            contract = b2.build_b2_candidate_model(
                source,
                candidate_id,
                output_dir=self.temp_root,
            )
        return source, target, contract

    def _fit_contract(self, candidate_id="B2-P1", history=None):
        candidate = config.B2_CANDIDATES[candidate_id]
        model = FakeFitModel(history=history)
        contract = b2.B2ModelContract(
            candidate=candidate,
            model=model,
            transferred_layer_names=tuple(f"layer_{i}" for i in candidate.transferred_indices),
            trainable_layer_names=tuple(f"layer_{i}" for i in candidate.trainable_indices),
            frozen_layer_names=tuple(f"layer_{i}" for i in candidate.frozen_indices),
            trainable_params=candidate.expected_trainable_params,
            total_params=config.B2_TOTAL_PARAMS,
            trainable_ratio=candidate.expected_trainable_params / config.B2_TOTAL_PARAMS,
        )
        return model, contract

    def _locked_candidate(self, candidate_id="B2-P1", prediction=None):
        paths = self._paths()
        candidate_root = paths.candidate_roots[candidate_id]
        checkpoint_root = candidate_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_root / "checkpoint_epoch_0002.hdf5"
        checkpoint.write_bytes(f"best-{candidate_id}".encode("ascii"))
        _, contract = self._fit_contract(candidate_id)
        fit = b2.B2FitEvidence(
            candidate_id=candidate_id,
            history=MappingProxyType({"loss": (0.5, 0.3, 0.4), "val_loss": (0.4, 0.2, 0.3)}),
            best_epoch=2,
            best_val_loss=0.2,
            checkpoint_path=checkpoint,
            checkpoint_sha256=b2._sha256(checkpoint),
            checkpoint_sha256_verified=True,
        )
        lock = b2.lock_b2_best_checkpoint(
            contract,
            fit,
            execution_git_head="1" * 40,
            run_root=paths.run_root,
        )
        truth = np.array([0.2, 0.4, 0.6], dtype=np.float64)
        predicted = truth + 0.01 if prediction is None else np.asarray(prediction)
        reloaded = b2.reload_b2_best_checkpoint(
            lock,
            model_loader_validator=lambda path, candidate, evidence: FakePredictionModel(predicted),
        )
        evaluation = b2.evaluate_b2_validation_checkpoint(
            reloaded,
            SimpleNamespace(X_seq=np.zeros((3, 5, 5)), y_seq=truth),
            FakeScaler(),
            expected_count=3,
        )
        return lock, reloaded, b2.finalize_b2_locked_candidate(lock, evaluation)

    def _locked_registry(self):
        return MappingProxyType(
            {
                candidate_id: self._locked_candidate(candidate_id)[2]
                for candidate_id in config.B2_CANDIDATES
            }
        )

    def _run_fake_orchestrator(self):
        events = []
        paths = self._paths()
        training = SimpleNamespace(
            X_seq=np.zeros((4, 5, 5)), y_seq=np.linspace(0.1, 0.4, 4)
        )
        validation = SimpleNamespace(
            X_seq=np.zeros((126, 5, 5)), y_seq=np.linspace(0.1, 0.9, 126)
        )
        profile = SimpleNamespace(
            training=SimpleNamespace(sequences=training),
            validation=SimpleNamespace(sequences=validation),
            scalers=SimpleNamespace(target_scaler=FakeScaler()),
        )

        def initialize(run_paths, manifest, baseline):
            events.append("initialize")
            b2.initialize_b2_run(run_paths, manifest, baseline)

        def builder(source, candidate_id, *, output_dir):
            del source, output_dir
            events.append(f"{candidate_id}:build")
            return self._fit_contract(candidate_id)[1]

        def fitter(contract, actual_training, actual_validation, *, checkpoint_directory):
            self.assertIs(actual_training, training)
            self.assertIs(actual_validation, validation)
            candidate_id = contract.candidate.candidate_id
            events.append(f"{candidate_id}:fit")
            root = Path(checkpoint_directory)
            root.mkdir(parents=True, exist_ok=False)
            checkpoint = root / "checkpoint_epoch_0002.hdf5"
            checkpoint.write_bytes(f"runner-{candidate_id}".encode("ascii"))
            return b2.B2FitEvidence(
                candidate_id=candidate_id,
                history=MappingProxyType({"loss": (0.5, 0.3), "val_loss": (0.4, 0.2)}),
                best_epoch=2,
                best_val_loss=0.2,
                checkpoint_path=checkpoint,
                checkpoint_sha256=b2._sha256(checkpoint),
                checkpoint_sha256_verified=True,
            )

        def reloader(lock):
            events.append(f"{lock.candidate_id}:reload")
            return b2.reload_b2_best_checkpoint(
                lock,
                model_loader_validator=lambda path, candidate, evidence: FakePredictionModel(
                    validation.y_seq + 0.01
                ),
            )

        def evaluator(reloaded, actual_validation, scaler):
            events.append(f"{reloaded.locked_candidate.candidate_id}:validation")
            return b2.evaluate_b2_validation_checkpoint(
                reloaded, actual_validation, scaler
            )

        def writer(run_paths, locked, history):
            events.append(f"{locked.candidate_id}:write")
            b2.write_locked_candidate_validation_results(run_paths, locked, history)

        dependencies = b2.B2RunnerDependencies(
            git_collector=lambda: b2.B2ExecutionGitProvenance(
                head="1" * 40,
                branch="AcerPing",
                tracked_dirty=False,
                untracked_paths=("notebook/unrelated.txt",),
                critical_files_committed=True,
            ),
            runtime_gate=lambda: MappingProxyType(
                {
                    "device": "CPU",
                    "cuda_visible_devices": "-1",
                    "pythonhashseed": "1234",
                    "seed": 1234,
                    "deterministic_ops": True,
                }
            ),
            path_factory=lambda run_id, output_base: paths,
            baseline_loader=b2.load_fixed_wotl_baseline,
            source_checkpoint_validator=lambda: events.append("source_sha") or self.temp_root,
            source_model_loader=lambda: events.append("source_load") or object(),
            profile_loader=lambda accessed: events.append("profile") or profile,
            run_initializer=initialize,
            candidate_builder=builder,
            candidate_fitter=fitter,
            checkpoint_reloader=reloader,
            validation_evaluator=evaluator,
            candidate_writer=writer,
        )
        result = b2.run_b2_training_validation(
            self.RUN_ID,
            "1" * 40,
            output_base=self.temp_root,
            _dependencies=dependencies,
            _allow_fake_dependencies=True,
        )
        return result, events, training, validation

    def test_01_import_is_inert_and_formal_disclosure_is_locked(self):
        before = self._snapshot(config.B2_OUTPUT_BASE)
        importlib.reload(config)
        importlib.reload(b2)
        self.assertEqual(self._snapshot(config.B2_OUTPUT_BASE), before)
        self.assertTrue(
            all(
                value
                for key, value in config.B2_DISCLOSURE.items()
                if key != "formal_final_test_replacement"
            )
        )
        self.assertFalse(config.B2_DISCLOSURE["formal_final_test_replacement"])
        self.assertTrue(config.B2_DISCLOSURE["post_test_supplementary_tuning"])

    def test_02_formal_experiment_b_snapshot_is_read_only(self):
        before = self._snapshot(config.B2_FORMAL_RUN_ROOT)
        b2.load_fixed_wotl_baseline()
        after = self._snapshot(config.B2_FORMAL_RUN_ROOT)
        self.assertEqual(after, before)

    def test_03_only_three_candidates_are_registered(self):
        self.assertEqual(tuple(config.B2_CANDIDATES), ("B2-P1", "B2-P2", "B2-P3"))
        with self.assertRaises(ValueError):
            config.b2_candidate("B2-P4")
        config.validate_b2_config()

    def test_04_candidate_learning_rates(self):
        self.assertEqual(config.B2_CANDIDATES["B2-P1"].learning_rate, 1e-5)
        self.assertEqual(config.B2_CANDIDATES["B2-P2"].learning_rate, 3e-6)
        self.assertEqual(config.B2_CANDIDATES["B2-P3"].learning_rate, 1e-5)

    def test_05_candidate_trainable_masks(self):
        self.assertEqual(config.B2_CANDIDATES["B2-P1"].trainable_indices, (4, 6))
        self.assertEqual(config.B2_CANDIDATES["B2-P2"].trainable_indices, (4, 6))
        self.assertEqual(config.B2_CANDIDATES["B2-P3"].trainable_indices, (6,))

    def test_06_batch_normalization_is_frozen_for_every_candidate(self):
        for candidate in config.B2_CANDIDATES.values():
            self.assertTrue(set((3, 5)).issubset(candidate.frozen_indices))
            self.assertTrue(candidate.batch_normalization_frozen)

    def test_07_fixed_source_identity_and_sha(self):
        self.assertEqual(config.B2_LOCKED_SOURCE_CANDIDATE_ID, "SRC_lr1e-4")
        self.assertEqual(config.B2_LOCKED_SOURCE_EPOCH, 500)
        self.assertEqual(
            config.B2_LOCKED_SOURCE_SHA256,
            "5cd3db4501d6c6720fbdd8ee3cce692b01cecb82619574f8c6944a8795182d91",
        )
        self.assertEqual(b2.validate_locked_source_checkpoint(), config.B2_LOCKED_SOURCE_CHECKPOINT)

    def test_08_transfer_scope_is_layers_1_to_5_only(self):
        self.assertEqual(config.B2_TRANSFERRED_LAYER_INDICES, (1, 2, 3, 4, 5))
        self.assertNotIn(6, config.B2_TRANSFERRED_LAYER_INDICES)

    def test_09_candidate_builder_transfers_1_to_5_not_output(self):
        source, target, contract = self._build_contract("B2-P1")
        for index in range(1, 6):
            np.testing.assert_array_equal(
                source.layers[index].get_weights()[0], target.layers[index].get_weights()[0]
            )
        np.testing.assert_array_equal(target.layers[6].get_weights()[0], [-60.0])
        self.assertFalse(contract.source_output_head_transferred)

    def test_10_candidate_builder_applies_exact_masks_and_counts(self):
        for candidate_id in config.B2_CANDIDATES:
            _, target, contract = self._build_contract(candidate_id)
            candidate = config.B2_CANDIDATES[candidate_id]
            actual = tuple(index for index in range(1, 7) if target.layers[index].trainable)
            self.assertEqual(actual, candidate.trainable_indices)
            self.assertEqual(contract.trainable_params, candidate.expected_trainable_params)
            self.assertEqual(contract.total_params, 46681)

    def test_11_linear_mse_compile_contract(self):
        _, target, _ = self._build_contract("B2-P3")
        self.assertEqual(target.loss, "mse")
        self.assertEqual(config.B2_LOSS, "mse")
        self.assertEqual(config.LINEAR_EXPERIMENTS["B"].activation, "linear")

    def test_12_fit_contract_is_seed_cpu_batch_shuffle_locked(self):
        self.assertEqual(config.B2_SEED, 1234)
        self.assertEqual(config.B2_DEVICE_POLICY.device, "CPU")
        self.assertEqual(config.B2_DEVICE_POLICY.cuda_visible_devices, "-1")
        self.assertEqual(config.B2_BATCH_SIZE, 128)
        self.assertFalse(config.B2_SHUFFLE)
        self.assertEqual(config.B2_MAXIMUM_EPOCHS, 500)

    def test_12a_corrected_split_and_scaler_contract_is_locked(self):
        self.assertEqual(
            dict(config.B2_EXPECTED_TARGET_ROW_COUNTS),
            {"training": 521, "validation": 131, "test": 2612},
        )
        self.assertEqual(
            dict(config.B2_EXPECTED_TARGET_SEQUENCE_COUNTS),
            {"training": 516, "validation": 126, "test": 2607},
        )
        self.assertEqual(config.B2_SCALER_FIT_SPLIT, "training")
        self.assertTrue(config.B2_VALIDATION_TEST_TRANSFORM_ONLY)

    def test_13_callback_policy_matches_formal(self):
        policy = config.B2_CALLBACK_POLICY
        self.assertEqual(policy.checkpoint.monitor, "val_loss")
        self.assertTrue(policy.checkpoint.save_best_only)
        self.assertEqual(policy.reduce_lr.factor, 0.1)
        self.assertEqual(policy.reduce_lr.patience, 20)
        self.assertEqual(policy.reduce_lr.min_lr, 1e-7)
        self.assertEqual(policy.early_stopping.patience, 50)
        self.assertEqual(policy.early_stopping.min_delta, 0.0)
        self.assertTrue(policy.terminate_on_nan)

    def test_14_fake_fit_uses_training_validation_only(self):
        model, contract = self._fit_contract()
        training = SimpleNamespace(X_seq=np.ones((4, 5, 5)), y_seq=np.ones(4))
        validation = SimpleNamespace(X_seq=np.ones((2, 5, 5)), y_seq=np.ones(2))

        def callback_factory(pattern):
            model.checkpoint_pattern = pattern
            return (object(), object(), object(), object())

        evidence = b2.fit_b2_candidate_validation_only(
            contract,
            training,
            validation,
            checkpoint_directory=self.temp_root / "checkpoint",
            callback_factory=callback_factory,
        )
        _, _, kwargs = model.fit_calls[0]
        self.assertIs(kwargs["validation_data"][0], validation.X_seq)
        self.assertEqual(kwargs["batch_size"], 128)
        self.assertFalse(kwargs["shuffle"])
        self.assertEqual(kwargs["epochs"], 500)
        self.assertNotIn("test_data", kwargs)
        self.assertEqual(evidence.selection_split, "validation")
        self.assertFalse(evidence.test_used_for_epoch_selection)

    def test_15_best_epoch_uses_val_loss_only(self):
        model, contract = self._fit_contract()
        sequences = SimpleNamespace(X_seq=np.ones((3, 5, 5)), y_seq=np.ones(3))

        def callback_factory(pattern):
            model.checkpoint_pattern = pattern
            return (object(), object(), object(), object())

        evidence = b2.fit_b2_candidate_validation_only(
            contract,
            sequences,
            sequences,
            checkpoint_directory=self.temp_root / "checkpoint",
            callback_factory=callback_factory,
        )
        self.assertEqual(evidence.best_epoch, 2)
        self.assertEqual(evidence.best_val_loss, 0.2)
        self.assertTrue(evidence.checkpoint_sha256_verified)

    def test_16_fit_history_rejects_test_metrics(self):
        model, contract = self._fit_contract(
            history={"loss": [0.2], "val_loss": [0.1], "test_loss": [0.01]}
        )
        sequences = SimpleNamespace(X_seq=np.ones((2, 5, 5)), y_seq=np.ones(2))

        def callback_factory(pattern):
            model.checkpoint_pattern = pattern
            return (object(), object(), object(), object())

        with self.assertRaisesRegex(b2.B2ContractError, "Test metrics"):
            b2.fit_b2_candidate_validation_only(
                contract,
                sequences,
                sequences,
                checkpoint_directory=self.temp_root / "checkpoint",
                callback_factory=callback_factory,
            )

    def test_17_fit_api_has_no_test_parameter(self):
        parameters = inspect.signature(b2.fit_b2_candidate_validation_only).parameters
        self.assertNotIn("test", parameters)
        self.assertNotIn("test_data", parameters)

    def test_18_fixed_wotl_baseline_matches_sealed_json_exactly(self):
        baseline = b2.load_fixed_wotl_baseline()
        self.assertEqual(asdict_like(baseline), dict(config.B2_FIXED_WOTL_METRICS))

    def test_19_raw_prediction_is_not_clipped(self):
        evaluated = b2.evaluate_raw_prediction(
            np.array([0.0, 0.5, 1.0]),
            np.array([-0.1, 0.4, 0.8]),
            FakeScaler(),
        )
        np.testing.assert_array_equal(evaluated.prediction_original, [-10000.0, 40000.0, 80000.0])
        self.assertEqual(evaluated.metrics.negative_prediction_count, 1)

    def test_20_original_scale_metric_metadata(self):
        metrics = b2.compute_original_scale_metrics(
            np.array([0.0, 1.0, 2.0]), np.array([-1.0, 1.0, 3.0])
        )
        self.assertEqual(metrics.target_name, "DC_POWER")
        self.assertEqual(metrics.scale, "original")
        self.assertIn("documented unit: kW", metrics.unit)

    def test_21_four_conditions_are_required_for_success(self):
        baseline = self._metric(10, 100, 10, 0.7)
        candidate = self._metric(9, 81, 9, 0.8)
        comparison = b2.compare_with_fixed_wotl(
            "B2-P1", candidate, baseline, validation_best_val_loss=0.2
        )
        self.assertTrue(comparison.success)
        self.assertIs(comparison.classification, b2.B2Classification.POSITIVE)

    def test_22_three_errors_only_is_not_positive(self):
        baseline = self._metric(10, 100, 10, 0.7)
        candidate = self._metric(9, 81, 9, 0.7)
        comparison = b2.compare_with_fixed_wotl(
            "B2-P1", candidate, baseline, validation_best_val_loss=0.2
        )
        self.assertFalse(comparison.success)
        self.assertIs(comparison.classification, b2.B2Classification.PARTIAL_POSITIVE)

    def test_23_mixed_and_negative_classification(self):
        baseline = self._metric(10, 100, 10, 0.7)
        mixed = b2.compare_with_fixed_wotl(
            "B2-P1", self._metric(9, 121, 11, 0.8), baseline, validation_best_val_loss=0.2
        )
        negative = b2.compare_with_fixed_wotl(
            "B2-P2", self._metric(11, 121, 11, 0.6), baseline, validation_best_val_loss=0.3
        )
        self.assertIs(mixed.classification, b2.B2Classification.MIXED)
        self.assertIs(negative.classification, b2.B2Classification.NEGATIVE)

    def test_24_successful_candidate_selection_uses_validation_loss(self):
        baseline = self._metric(10, 100, 10, 0.7)
        comparisons = {
            candidate_id: b2.compare_with_fixed_wotl(
                candidate_id,
                self._metric(9, 81, 9, 0.8),
                baseline,
                validation_best_val_loss=loss,
            )
            for candidate_id, loss in (("B2-P1", 0.3), ("B2-P2", 0.1), ("B2-P3", 0.2))
        }
        selected, achieved = b2.select_supplementary_candidate(comparisons)
        self.assertTrue(achieved)
        self.assertEqual(selected, "B2-P2")

    def test_25_no_success_stops_without_b2_2(self):
        baseline = self._metric(10, 100, 10, 0.7)
        comparisons = {
            candidate_id: b2.compare_with_fixed_wotl(
                candidate_id,
                self._metric(11, 121, 11, 0.6),
                baseline,
                validation_best_val_loss=loss,
            )
            for candidate_id, loss in (("B2-P1", 0.3), ("B2-P2", 0.1), ("B2-P3", 0.2))
        }
        self.assertEqual(b2.select_supplementary_candidate(comparisons), (None, False))
        self.assertFalse(config.B2_AUTOMATIC_NEXT_ROUND)

    def test_26_test_reuse_disclosure_is_explicit(self):
        disclosure = b2.test_reuse_disclosure()
        self.assertTrue(disclosure["original_plant1_test_previously_revealed"])
        self.assertTrue(disclosure["plant1_test_reused"])
        self.assertTrue(disclosure["post_test_supplementary_tuning"])
        self.assertFalse(disclosure["formal_final_test_replacement"])
        self.assertEqual(disclosure["test_disk_load_count"], 1)

    def test_27_test_batch_loads_once_and_shares_same_x_object(self):
        data = self._fake_test_data()
        calls = []
        disclosure = self.temp_root / "test_reuse_disclosure.json"

        def loader():
            self.assertTrue(disclosure.is_file())
            calls.append(1)
            return data

        models = {
            "B2-P1": FakePredictionModel(data.y_test + 0.02),
            "B2-P2": FakePredictionModel(data.y_test + 0.01),
            "B2-P3": FakePredictionModel(data.y_test + 0.03),
        }
        baseline = b2.load_fixed_wotl_baseline()
        result = b2.evaluate_supplementary_test_batch(
            models,
            {"B2-P1": 0.2, "B2-P2": 0.1, "B2-P3": 0.3},
            baseline,
            test_loader=loader,
            disclosure_path=disclosure,
            _allow_unlocked_test_models=True,
        )
        self.assertEqual(calls, [1])
        self.assertEqual(result.test_disk_load_count, 1)
        self.assertTrue(result.shared_x_test_object)
        for model in models.values():
            self.assertIs(model.predict_calls[0][0], data.X_test)

    def test_28_test_batch_rejects_a_fourth_candidate(self):
        data = self._fake_test_data()
        models = {candidate_id: FakePredictionModel(data.y_test) for candidate_id in config.B2_CANDIDATES}
        models["B2-P4"] = FakePredictionModel(data.y_test)
        with self.assertRaises(b2.B2ContractError):
            b2.evaluate_supplementary_test_batch(
                models,
                {candidate_id: 0.1 for candidate_id in config.B2_CANDIDATES},
                b2.load_fixed_wotl_baseline(),
                test_loader=lambda: data,
                disclosure_path=self.temp_root / "disclosure.json",
                _allow_unlocked_test_models=True,
            )

    def test_28a_test_batch_rejects_a_nonformal_wotl_baseline_before_load(self):
        data = self._fake_test_data()
        models = {
            candidate_id: FakePredictionModel(data.y_test)
            for candidate_id in config.B2_CANDIDATES
        }
        calls = []
        tampered = b2.B2Metrics(
            **{
                **asdict_like(b2.load_fixed_wotl_baseline()),
                "mae": config.B2_FIXED_WOTL_METRICS["mae"] - 1.0,
            }
        )
        with self.assertRaisesRegex(b2.B2ContractError, "Runtime WOTL baseline"):
            b2.evaluate_supplementary_test_batch(
                models,
                {candidate_id: 0.1 for candidate_id in config.B2_CANDIDATES},
                tampered,
                test_loader=lambda: calls.append(1),
                disclosure_path=self.temp_root / "disclosure.json",
                _allow_unlocked_test_models=True,
            )
        self.assertEqual(calls, [])
        self.assertFalse((self.temp_root / "disclosure.json").exists())

    def test_29_output_path_is_isolated_from_formal(self):
        paths = self._paths()
        self.assertTrue(paths.run_root.is_relative_to(self.temp_root))
        self.assertFalse(paths.run_root.is_relative_to(config.B2_FORMAL_RUN_ROOT))
        self.assertIn("Linear_Supplementary", str(config.B2_OUTPUT_BASE))
        self.assertNotIn("Linear_Formal", str(config.B2_OUTPUT_BASE))

    def test_30_existing_destination_is_never_overwritten(self):
        paths = self._paths()
        paths.run_root.mkdir(parents=True)
        with self.assertRaises(b2.B2ContractError):
            b2.validate_new_b2_destination(paths)

    def test_31_manifest_records_dirty_untracked_and_disclosure(self):
        manifest = self._manifest()
        self.assertTrue(manifest["tracked_dirty"])
        self.assertEqual(manifest["untracked_paths"], ("r2_config/solar_linear_b2.py",))
        self.assertTrue(manifest["formal_experiment_b_preserved"])
        self.assertFalse(manifest["formal_result_replaced"])
        self.assertEqual(manifest["candidate_ids"], ("B2-P1", "B2-P2", "B2-P3"))
        self.assertEqual(manifest["scaler_fit_split"], "training")
        self.assertTrue(manifest["validation_test_transform_only"])
        self.assertTrue(manifest["target_split_reused_without_resplitting"])

    def test_32_temp_run_initialization_creates_required_isolated_layout(self):
        paths = self._paths()
        baseline = b2.load_fixed_wotl_baseline()
        b2.initialize_b2_run(paths, self._manifest(), baseline)
        self.assertTrue(paths.run_manifest.is_file())
        self.assertTrue(paths.wotl_baseline.is_file())
        self.assertTrue(paths.summary_root.is_dir())
        self.assertEqual(set(paths.candidate_roots), set(config.B2_CANDIDATES))
        for root in paths.candidate_roots.values():
            self.assertTrue((root / "config.json").is_file())

    def test_33_temp_batch_outputs_are_supplementary_only(self):
        paths = self._paths()
        baseline = b2.load_fixed_wotl_baseline()
        b2.initialize_b2_run(paths, self._manifest(), baseline)
        data = self._fake_test_data()
        models = {
            "B2-P1": FakePredictionModel(data.y_test + 0.02),
            "B2-P2": FakePredictionModel(data.y_test + 0.01),
            "B2-P3": FakePredictionModel(data.y_test + 0.03),
        }
        result = b2.evaluate_supplementary_test_batch(
            models,
            {"B2-P1": 0.2, "B2-P2": 0.1, "B2-P3": 0.3},
            baseline,
            test_loader=lambda: data,
            disclosure_path=paths.test_reuse_disclosure,
            _allow_unlocked_test_models=True,
        )
        b2.write_supplementary_batch_results(paths, data, result, baseline)
        self.assertTrue(paths.comparison.is_file())
        self.assertTrue(paths.results_summary.is_file())
        summary = json.loads(paths.results_summary.read_text(encoding="utf-8"))
        self.assertFalse(summary["formal_final_test_replacement"])
        self.assertFalse(summary["automatic_b2_2_started"])
        for root in paths.candidate_roots.values():
            self.assertTrue((root / "test_metrics_original_scale.json").is_file())
            self.assertTrue((root / "predictions.csv").is_file())

    def test_34_candidate_validation_evidence_is_validation_only(self):
        _, contract = self._fit_contract()
        fit = b2.B2FitEvidence(
            candidate_id="B2-P1",
            history=MappingProxyType({"val_loss": (0.2,)}),
            best_epoch=1,
            best_val_loss=0.2,
            checkpoint_path=self.temp_root / "checkpoint.hdf5",
            checkpoint_sha256="a" * 64,
            checkpoint_sha256_verified=True,
        )
        evidence = b2.candidate_validation_evidence(
            contract,
            fit,
            {
                "normalized_mae": 0.1,
                "normalized_mse": 0.02,
                "normalized_rmse": 0.14,
                "normalized_r2": 0.8,
                "original_mae": 100.0,
                "original_mse": 20000.0,
                "original_rmse": 141.4,
                "original_r2": 0.8,
            },
        )
        self.assertEqual(evidence["selection_split"], "validation")
        self.assertFalse(evidence["test_used_for_epoch_selection"])
        self.assertIn("trainable_layer_names", evidence)
        self.assertIn("checkpoint_sha256", evidence)

    def test_35_formal_classification_and_artifacts_remain_unchanged(self):
        before = self._snapshot(config.B2_FORMAL_RUN_ROOT)
        comparison = json.loads(config.B2_FORMAL_COMPARISON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(comparison["classification"], "Negative Transfer")
        paths = self._paths()
        b2.initialize_b2_run(paths, self._manifest(), b2.load_fixed_wotl_baseline())
        self.assertEqual(self._snapshot(config.B2_FORMAL_RUN_ROOT), before)

    def test_36_no_formal_training_or_real_test_is_executed_by_import(self):
        before = self._snapshot(config.B2_OUTPUT_BASE)
        source = inspect.getsource(b2)
        self.assertNotIn("SOLAR_RUN_FINAL_TEST", source)
        self.assertNotIn("SOLAR_RUN_FORMAL", source)
        self.assertEqual(self._snapshot(config.B2_OUTPUT_BASE), before)

    def test_37_checkpoint_lock_records_complete_identity(self):
        lock, _, _ = self._locked_candidate()
        self.assertEqual(lock.candidate_id, "B2-P1")
        self.assertEqual(lock.best_epoch, 2)
        self.assertEqual(lock.selection_split, "validation")
        self.assertFalse(lock.test_used_for_epoch_selection)
        self.assertEqual(lock.execution_git_head, "1" * 40)

    def test_38_reload_uses_exact_best_epoch_path_and_sha_twice(self):
        lock, _, _ = self._locked_candidate()
        loaded = []
        hashed = []

        def digest(path):
            hashed.append(path)
            return b2._sha256(path)

        reloaded = b2.reload_b2_best_checkpoint(
            lock,
            sha256_func=digest,
            model_loader_validator=lambda path, candidate, evidence: loaded.append(path)
            or FakePredictionModel([0.1, 0.2, 0.3]),
        )
        self.assertEqual(loaded, [lock.checkpoint_path])
        self.assertEqual(hashed, [lock.checkpoint_path, lock.checkpoint_path])
        self.assertTrue(reloaded.checkpoint_sha256_verified)

    def test_39_reload_rejects_missing_best_checkpoint(self):
        lock, _, _ = self._locked_candidate()
        lock.checkpoint_path.unlink()
        with self.assertRaisesRegex(b2.B2ContractError, "missing/empty"):
            b2.reload_b2_best_checkpoint(lock, model_loader_validator=lambda *args: object())

    def test_40_reload_rejects_actual_checkpoint_byte_tampering(self):
        lock, _, _ = self._locked_candidate()
        lock.checkpoint_path.write_bytes(b"tampered-after-lock")
        with self.assertRaisesRegex(b2.B2ContractError, "differs from locked evidence"):
            b2.reload_b2_best_checkpoint(lock, model_loader_validator=lambda *args: object())

    def test_40a_reload_rejects_locked_sha_field_tampering(self):
        _, _, locked = self._locked_candidate()
        calls = []
        with self.assertRaisesRegex(b2.B2ContractError, "differs from locked evidence"):
            b2.reload_b2_best_checkpoint(
                replace(locked, checkpoint_sha256="f" * 64),
                model_loader_validator=lambda *args: calls.append(1),
            )
        self.assertEqual(calls, [])

    def test_41_best_checkpoint_not_final_epoch_model_supplies_evidence(self):
        lock, _, _ = self._locked_candidate()
        final_epoch_model = FakePredictionModel([0.9, 0.9, 0.9])
        best_prediction = np.array([0.21, 0.41, 0.61])
        reloaded = b2.reload_b2_best_checkpoint(
            lock,
            model_loader_validator=lambda *args: FakePredictionModel(best_prediction),
        )
        evaluation = b2.evaluate_b2_validation_checkpoint(
            reloaded,
            SimpleNamespace(
                X_seq=np.zeros((3, 5, 5)), y_seq=np.array([0.2, 0.4, 0.6])
            ),
            FakeScaler(),
            expected_count=3,
        )
        self.assertEqual(final_epoch_model.predict_calls, [])
        self.assertAlmostEqual(evaluation.normalized_metrics["mae"], 0.01)
        np.testing.assert_array_equal(reloaded.model.prediction.reshape(-1), best_prediction)

    def test_42_validation_normalized_metrics_are_complete(self):
        _, _, locked = self._locked_candidate()
        metrics = locked.validation_metrics_normalized
        self.assertEqual(
            set(metrics),
            {"mae", "mse", "rmse", "r2", "prediction_count", "negative_prediction_count", "target_name", "scale", "unit"},
        )
        self.assertEqual(metrics["prediction_count"], 3)
        self.assertEqual(metrics["scale"], "normalized")

    def test_43_validation_original_metrics_are_complete(self):
        _, _, locked = self._locked_candidate()
        metrics = locked.validation_metrics_original_scale
        self.assertEqual(metrics["prediction_count"], 3)
        self.assertEqual(metrics["target_name"], "DC_POWER")
        self.assertEqual(metrics["scale"], "original")
        self.assertGreater(metrics["rmse"], 0)

    def test_44_validation_scaler_is_inverse_transform_only(self):
        lock, _, _ = self._locked_candidate()
        scaler = FakeScaler()
        reloaded = b2.reload_b2_best_checkpoint(
            lock,
            model_loader_validator=lambda *args: FakePredictionModel([0.2, 0.4, 0.6]),
        )
        b2.evaluate_b2_validation_checkpoint(
            reloaded,
            SimpleNamespace(X_seq=np.zeros((3, 5, 5)), y_seq=np.array([0.2, 0.4, 0.6])),
            scaler,
            expected_count=3,
        )
        self.assertEqual(scaler.inverse_transform_calls, 2)

    def test_45_final_locked_candidate_is_immutable_and_complete(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaises(Exception):
            locked.best_epoch = 3
        payload = b2.locked_candidate_mapping(locked)
        self.assertIn("validation_metrics_normalized", payload)
        self.assertIn("validation_metrics_original_scale", payload)

    def test_46_locked_candidate_writer_outputs_all_validation_evidence(self):
        paths = self._paths()
        b2.initialize_b2_run(paths, self._manifest(), b2.load_fixed_wotl_baseline())
        _, _, locked = self._locked_candidate()
        b2.write_locked_candidate_validation_results(
            paths,
            locked,
            {"loss": (0.5, 0.3), "val_loss": (0.4, 0.2)},
        )
        root = paths.candidate_roots[locked.candidate_id]
        for name in (
            "training_history.csv",
            "best_checkpoint.json",
            "validation_metrics_normalized.json",
            "validation_metrics_original_scale.json",
            "locked_candidate.json",
        ):
            self.assertTrue((root / name).is_file(), name)

    def test_47_validation_registry_requires_and_records_all_three(self):
        paths = self._paths()
        b2.initialize_b2_run(paths, self._manifest(), b2.load_fixed_wotl_baseline())
        registry = self._locked_registry()
        path = b2.write_b2_validation_registry(paths, registry)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidate_order"], ["B2-P1", "B2-P2", "B2-P3"])
        self.assertEqual(set(payload["candidates"]), set(config.B2_CANDIDATES))
        self.assertFalse(payload["supplementary_test_executed"])

    def test_48_locked_trainable_mask_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        tampered = replace(locked, trainable_layer_indices=(6,))
        with self.assertRaisesRegex(b2.B2ContractError, "trainable indices"):
            b2.reload_b2_best_checkpoint(tampered, model_loader_validator=lambda *args: object())

    def test_48a_locked_frozen_mask_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaisesRegex(b2.B2ContractError, "frozen indices"):
            b2.reload_b2_best_checkpoint(
                replace(locked, frozen_layer_indices=(1, 2, 3, 4, 5)),
                model_loader_validator=lambda *args: object(),
            )

    def test_48b_default_reload_reapplies_exact_mask_bn_and_parameter_count(self):
        from keras import models as keras_models
        from keras import optimizers as keras_optimizers
        from r2_helpers import solar_linear_runtime as runtime

        for candidate_id, candidate in config.B2_CANDIDATES.items():
            lock, _, _ = self._locked_candidate(candidate_id)
            loaded_model = FakeBuildModel(prefix=1.0)
            validation_calls = []
            with patch.object(keras_models, "load_model", return_value=loaded_model), patch.object(
                keras_optimizers, "Adam", return_value=object()
            ), patch.object(
                runtime,
                "validate_linear_model",
                side_effect=lambda model, expected_learning_rate=None: validation_calls.append(
                    expected_learning_rate
                ),
            ), patch.object(
                runtime,
                "parameter_counts",
                return_value=SimpleNamespace(
                    total=config.B2_TOTAL_PARAMS,
                    trainable=candidate.expected_trainable_params,
                    non_trainable=config.B2_TOTAL_PARAMS - candidate.expected_trainable_params,
                ),
            ):
                reloaded = b2.reload_b2_best_checkpoint(lock)
            actual_trainable = tuple(
                index for index in range(1, 7) if reloaded.model.layers[index].trainable
            )
            self.assertEqual(actual_trainable, candidate.trainable_indices)
            self.assertFalse(reloaded.model.layers[3].trainable)
            self.assertFalse(reloaded.model.layers[5].trainable)
            self.assertEqual(validation_calls, [candidate.learning_rate])

    def test_49_locked_learning_rate_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaisesRegex(b2.B2ContractError, "learning rate"):
            b2.reload_b2_best_checkpoint(
                replace(locked, learning_rate=9e-4), model_loader_validator=lambda *args: object()
            )

    def test_50_locked_strategy_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaisesRegex(b2.B2ContractError, "strategy"):
            b2.reload_b2_best_checkpoint(
                replace(locked, strategy="other"), model_loader_validator=lambda *args: object()
            )

    def test_51_locked_selection_split_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaisesRegex(b2.B2ContractError, "selection split"):
            b2.reload_b2_best_checkpoint(
                replace(locked, selection_split="test"), model_loader_validator=lambda *args: object()
            )

    def test_52_locked_test_selection_flag_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaisesRegex(b2.B2ContractError, "used Test"):
            b2.reload_b2_best_checkpoint(
                replace(locked, test_used_for_epoch_selection=True),
                model_loader_validator=lambda *args: object(),
            )

    def test_53_locked_candidate_id_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaises(b2.B2ContractError):
            b2.reload_b2_best_checkpoint(
                replace(locked, candidate_id="B2-P2"), model_loader_validator=lambda *args: object()
            )

    def test_54_locked_checkpoint_path_tamper_fails_before_loader(self):
        _, _, locked = self._locked_candidate()
        with self.assertRaisesRegex(b2.B2ContractError, "path/epoch/run"):
            b2.reload_b2_best_checkpoint(
                replace(locked, checkpoint_path=self.temp_root / "other.hdf5"),
                model_loader_validator=lambda *args: object(),
            )

    def test_55_arbitrary_model_test_entry_is_rejected(self):
        calls = []
        with self.assertRaisesRegex(b2.B2ContractError, "Arbitrary B2 models"):
            b2.evaluate_supplementary_test_batch(
                {candidate_id: object() for candidate_id in config.B2_CANDIDATES},
                {candidate_id: 0.1 for candidate_id in config.B2_CANDIDATES},
                b2.load_fixed_wotl_baseline(),
                test_loader=lambda: calls.append(1),
                disclosure_path=self.temp_root / "disclosure.json",
            )
        self.assertEqual(calls, [])

    def test_56_safe_entry_reloads_all_candidates_before_test_loader(self):
        registry = self._locked_registry()
        events = []
        data = self._fake_test_data()

        def reloader(locked):
            events.append(f"reload:{locked.candidate_id}")
            return b2.B2ReloadedCandidate(
                locked_candidate=locked,
                model=FakePredictionModel(data.y_test),
                checkpoint_sha256_verified=True,
            )

        def loader():
            events.append("test-load")
            return data

        with patch.object(b2, "reload_b2_best_checkpoint", side_effect=reloader):
            b2.evaluate_locked_supplementary_test_batch(
                registry,
                b2.load_fixed_wotl_baseline(),
                test_loader=loader,
                disclosure_path=self.temp_root / "disclosure.json",
            )
        self.assertEqual(events[:3], ["reload:B2-P1", "reload:B2-P2", "reload:B2-P3"])
        self.assertEqual(events[3:], ["test-load"])

    def test_57_safe_entry_loads_test_once_and_shares_x(self):
        registry = self._locked_registry()
        data = self._fake_test_data()
        models = {}
        loads = []

        def reloader(locked):
            model = FakePredictionModel(data.y_test)
            models[locked.candidate_id] = model
            return b2.B2ReloadedCandidate(locked, model, True)

        with patch.object(b2, "reload_b2_best_checkpoint", side_effect=reloader):
            result = b2.evaluate_locked_supplementary_test_batch(
                registry,
                b2.load_fixed_wotl_baseline(),
                test_loader=lambda: loads.append(1) or data,
                disclosure_path=self.temp_root / "disclosure.json",
            )
        self.assertEqual(loads, [1])
        self.assertEqual(result.test_disk_load_count, 1)
        self.assertTrue(all(model.predict_calls[0][0] is data.X_test for model in models.values()))

    def test_58_safe_entry_detects_byte_tamper_before_test_loader(self):
        registry = dict(self._locked_registry())
        registry["B2-P1"].checkpoint_path.write_bytes(b"tampered")
        loads = []
        with self.assertRaisesRegex(b2.B2ContractError, "differs from locked evidence"):
            b2.evaluate_locked_supplementary_test_batch(
                MappingProxyType(registry),
                b2.load_fixed_wotl_baseline(),
                test_loader=lambda: loads.append(1),
                disclosure_path=self.temp_root / "disclosure.json",
            )
        self.assertEqual(loads, [])

    def test_59_runner_signature_has_no_test_argument(self):
        parameters = inspect.signature(b2.run_b2_training_validation).parameters
        self.assertFalse(any("test" in name.lower() for name in parameters))

    def test_60_runner_source_contains_no_test_loader_or_split(self):
        source = inspect.getsource(b2.run_b2_training_validation)
        self.assertNotIn("load_revealed_plant1_test", source)
        self.assertNotIn("evaluate_supplementary_test_batch", source)
        self.assertNotIn('load_split(', source)

    def test_61_runner_uses_nested_sequences_and_fixed_candidate_order(self):
        result, events, training, validation = self._run_fake_orchestrator()
        del training, validation
        lifecycle = [event for event in events if event.startswith("B2-")]
        self.assertEqual(
            lifecycle,
            [
                f"{candidate_id}:{stage}"
                for candidate_id in config.B2_CANDIDATES
                for stage in ("build", "fit", "reload", "validation", "write")
            ],
        )
        self.assertEqual(tuple(result.locked_candidates), tuple(config.B2_CANDIDATES))

    def test_62_runner_writes_complete_registry_and_stops_after_validation(self):
        result, events, _, _ = self._run_fake_orchestrator()
        payload = json.loads(result.validation_registry_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["training_validation_complete"])
        self.assertFalse(payload["supplementary_test_executed"])
        self.assertEqual(result.total_training_epochs_executed, 6)
        self.assertNotIn("test-load", events)

    def test_63_execution_git_gate_rejects_dirty_or_uncommitted_critical_files(self):
        clean = b2.B2ExecutionGitProvenance("1" * 40, "branch", False, (), True)
        b2.validate_b2_execution_git_provenance(clean, expected_git_head="1" * 40)
        with self.assertRaisesRegex(b2.B2ContractError, "dirty"):
            b2.validate_b2_execution_git_provenance(
                replace(clean, tracked_dirty=True), expected_git_head="1" * 40
            )
        with self.assertRaisesRegex(b2.B2ContractError, "not committed"):
            b2.validate_b2_execution_git_provenance(
                replace(clean, critical_files_committed=False), expected_git_head="1" * 40
            )

    def test_64_cpu_seed_gate_sets_visibility_before_framework_import(self):
        source = inspect.getsource(b2.enforce_b2_cpu_seed_determinism)
        self.assertLess(source.index('os.environ["CUDA_VISIBLE_DEVICES"]'), source.index("import keras"))
        self.assertIn("PYTHONHASHSEED", source)
        self.assertIn("enable_op_determinism", source)

    def test_65_candidate_research_parameters_remain_exact(self):
        self.assertEqual(
            [(item.candidate_id, item.learning_rate, item.trainable_indices, item.frozen_indices) for item in config.B2_CANDIDATES.values()],
            [
                ("B2-P1", 1e-5, (4, 6), (1, 2, 3, 5)),
                ("B2-P2", 3e-6, (4, 6), (1, 2, 3, 5)),
                ("B2-P3", 1e-5, (6,), (1, 2, 3, 4, 5)),
            ],
        )

    def test_66_hardening_does_not_create_b2_formal_output(self):
        before = self._snapshot(config.B2_OUTPUT_BASE)
        self._locked_candidate()
        self.assertEqual(self._snapshot(config.B2_OUTPUT_BASE), before)


def asdict_like(value):
    return {
        "mae": value.mae,
        "mse": value.mse,
        "rmse": value.rmse,
        "r2": value.r2,
        "prediction_count": value.prediction_count,
        "negative_prediction_count": value.negative_prediction_count,
        "target_name": value.target_name,
        "scale": value.scale,
        "unit": value.unit,
    }


if __name__ == "__main__":
    unittest.main()
