"""Zero-epoch build and lifecycle tests for the Solar R2 runtime contract."""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from r2_helpers.solar_runtime import (
    EXPECTED_LAYER_TOPOLOGY,
    NON_TRANSFERRED_WEIGHT_LAYER_INDICES,
    R2_BATCH_SIZE,
    R2_CUSTOM_OBJECTS,
    R2_INPUT_SHAPE,
    R2_METHOD_NAMES,
    TRANSFERABLE_LAYER_INDICES,
    NamedSequenceSplit,
    ReloadedCheckpointModel,
    RuntimeContractError,
    build_model,
    build_r2_model,
    checkpoint_paths,
    configure_reproducibility,
    evaluate_test,
    fit_train_validation,
    make_r2_callbacks,
    reload_best_checkpoint,
    rmse,
    validate_transferred_weights,
)


class FitStub:
    def __init__(self):
        self.calls = []
        self.real_epochs_executed = 0

    def fit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"stub": True}


class SolarR2RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.paths = checkpoint_paths(cls.root)

        configure_reproducibility(1234)
        cls.source = build_r2_model(output_dir=cls.root)
        cls._set_distinct_source_weights(cls.source)

        configure_reproducibility(1234)
        cls.target_initial = build_r2_model(output_dir=cls.root)
        configure_reproducibility(1234)
        cls.tl_freeze = build_r2_model(
            output_dir=cls.root,
            pretrained_model=cls.source,
            freeze=True,
        )
        configure_reproducibility(1234)
        cls.tl_full = build_r2_model(
            output_dir=cls.root,
            pretrained_model=cls.source,
            freeze=False,
        )

        cls.paths.source_pretrain.parent.mkdir(parents=True)
        cls.source.save(str(cls.paths.source_pretrain))
        cls.reloaded = reload_best_checkpoint(cls.paths.source_pretrain)

        cls.X = np.arange(4 * 5 * 5, dtype=np.float32).reshape(4, 5, 5) / 100.0
        cls.y = np.arange(4, dtype=np.float32) / 10.0
        cls.training = NamedSequenceSplit("training", cls.X, cls.y)
        cls.validation = NamedSequenceSplit("validation", cls.X + 1.0, cls.y + 1.0)
        cls.test = NamedSequenceSplit("test", cls.X + 2.0, cls.y + 2.0)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @staticmethod
    def _set_distinct_source_weights(model):
        for layer_index in range(1, len(model.layers)):
            weights = model.layers[layer_index].get_weights()
            if not weights:
                continue
            values = []
            for weight_index, weight in enumerate(weights):
                value = 0.01 * (layer_index + 1) * (weight_index + 1)
                if type(model.layers[layer_index]).__name__ == "BatchNormalization" and weight_index == 3:
                    value = 1.0 + value
                values.append(np.full_like(weight, value))
            model.layers[layer_index].set_weights(values)

    def test_01_utils_model_import_compatibility(self):
        self.assertTrue(callable(build_model))
        self.assertTrue(callable(rmse))

    def test_02_build_input_output_shape(self):
        self.assertEqual(tuple(self.source.input_shape[1:]), R2_INPUT_SHAPE)
        self.assertEqual(tuple(self.source.output_shape[1:]), (1,))

    def test_03_exact_topology(self):
        actual = tuple(type(layer).__name__ for layer in self.source.layers)
        self.assertEqual(actual, EXPECTED_LAYER_TOPOLOGY)

    def test_04_units(self):
        self.assertEqual(self.source.layers[1].layer.units, 10)
        self.assertEqual(self.source.layers[2].units, 60)
        self.assertEqual(self.source.layers[4].units, 60)
        self.assertEqual(self.source.layers[6].units, 1)

    def test_05_sigmoid_output(self):
        self.assertEqual(self.source.layers[6].activation.__name__, "sigmoid")

    def test_06_optimizer_loss_learning_rates(self):
        self.assertEqual(type(self.source.optimizer).__name__, "Adam")
        self.assertIn(self.source.loss, ("mse", "mean_squared_error"))
        self.assertAlmostEqual(float(self.source.optimizer.learning_rate.numpy()), 1e-4, places=11)
        self.assertAlmostEqual(float(self.tl_freeze.optimizer.learning_rate.numpy()), 1e-5, places=12)
        self.assertAlmostEqual(float(self.tl_full.optimizer.learning_rate.numpy()), 1e-5, places=12)

    def test_07_same_seed_deterministic_initialization(self):
        configure_reproducibility(1234)
        first = build_r2_model(output_dir=self.root)
        configure_reproducibility(1234)
        second = build_r2_model(output_dir=self.root)
        for first_weight, second_weight in zip(first.get_weights(), second.get_weights()):
            np.testing.assert_array_equal(first_weight, second_weight)

    def test_08_transferable_indices(self):
        self.assertEqual(TRANSFERABLE_LAYER_INDICES, (2, 3, 4, 5))
        self.assertEqual(NON_TRANSFERRED_WEIGHT_LAYER_INDICES, (1, 6))

    def test_09_freeze_copied_weights(self):
        for index in TRANSFERABLE_LAYER_INDICES:
            for source_weight, target_weight in zip(
                self.source.layers[index].get_weights(),
                self.tl_freeze.layers[index].get_weights(),
            ):
                np.testing.assert_array_equal(source_weight, target_weight)

    def test_10_full_finetune_copied_weights(self):
        for index in TRANSFERABLE_LAYER_INDICES:
            for source_weight, target_weight in zip(
                self.source.layers[index].get_weights(),
                self.tl_full.layers[index].get_weights(),
            ):
                np.testing.assert_array_equal(source_weight, target_weight)

    def test_11_freeze_trainable_flags(self):
        validate_transferred_weights(
            self.source, self.tl_freeze, self.target_initial, freeze=True
        )
        for index in TRANSFERABLE_LAYER_INDICES:
            self.assertFalse(self.tl_freeze.layers[index].trainable)

    def test_12_full_finetune_trainable_flags(self):
        validate_transferred_weights(
            self.source, self.tl_full, self.target_initial, freeze=False
        )
        self.assertTrue(all(layer.trainable for layer in self.tl_full.layers))

    def test_13_non_transferred_layers_retain_target_initialization(self):
        for target in (self.tl_freeze, self.tl_full):
            for index in NON_TRANSFERRED_WEIGHT_LAYER_INDICES:
                for initial_weight, target_weight, source_weight in zip(
                    self.target_initial.layers[index].get_weights(),
                    target.layers[index].get_weights(),
                    self.source.layers[index].get_weights(),
                ):
                    np.testing.assert_array_equal(initial_weight, target_weight)
                    self.assertFalse(np.array_equal(source_weight, target_weight))

    def test_14_models_do_not_share_weight_objects(self):
        for target in (self.tl_freeze, self.tl_full):
            for source_variable, target_variable in zip(self.source.weights, target.weights):
                self.assertIsNot(source_variable, target_variable)

    def test_15_callback_classes_and_order(self):
        callbacks = make_r2_callbacks(self.paths.without_tl)
        self.assertEqual(
            tuple(type(callback) for callback in callbacks),
            (ReduceLROnPlateau, ModelCheckpoint, EarlyStopping),
        )

    def test_16_callback_parameters(self):
        reduce_lr, checkpoint, early = make_r2_callbacks(self.paths.tl_freeze)
        self.assertEqual(reduce_lr.monitor, "val_loss")
        self.assertEqual(reduce_lr.factor, 0.5)
        self.assertEqual(reduce_lr.patience, 4)
        self.assertEqual(reduce_lr.min_lr, 1e-7)
        self.assertEqual(checkpoint.monitor, "val_loss")
        self.assertTrue(checkpoint.save_best_only)
        self.assertFalse(checkpoint.save_weights_only)
        self.assertEqual(early.monitor, "val_loss")
        self.assertEqual(early.patience, 10)
        self.assertTrue(early.restore_best_weights)

    def test_17_checkpoint_paths_are_unique_and_method_specific(self):
        values = self.paths.as_dict()
        self.assertEqual(len(set(values.values())), 4)
        self.assertEqual(tuple(name for name in R2_METHOD_NAMES), ("without_tl", "tl_freeze", "tl_full_finetune"))
        self.assertEqual(self.paths.source_pretrain.relative_to(self.root).as_posix(), "source/pretrain/best_model.hdf5")
        for method in R2_METHOD_NAMES:
            self.assertEqual(values[method].relative_to(self.root).as_posix(), f"target/{method}/best_model.hdf5")

    def test_18_missing_checkpoint_reload_fails(self):
        with self.assertRaises(RuntimeContractError):
            reload_best_checkpoint(self.root / "missing" / "best_model.hdf5")

    def test_19_manual_save_load_round_trip(self):
        self.assertIsInstance(self.reloaded, ReloadedCheckpointModel)
        self.assertEqual(self.reloaded.checkpoint_path, self.paths.source_pretrain)
        for saved_weight, loaded_weight in zip(
            self.source.get_weights(), self.reloaded.model.get_weights()
        ):
            np.testing.assert_array_equal(saved_weight, loaded_weight)

    def test_20_reload_prediction_consistency(self):
        before = self.source.predict(self.X, verbose=0)
        after = self.reloaded.model.predict(self.X, verbose=0)
        np.testing.assert_allclose(before, after, rtol=0.0, atol=0.0)

    def test_21_custom_objects_contains_rmse(self):
        self.assertEqual(set(R2_CUSTOM_OBJECTS), {"rmse"})
        self.assertIs(R2_CUSTOM_OBJECTS["rmse"], rmse)

    def test_22_fit_wrapper_uses_shuffle_true(self):
        stub = FitStub()
        fit_train_validation(
            stub,
            training=self.training,
            validation=self.validation,
            callbacks=(),
            epochs=7,
            verbose=0,
        )
        self.assertTrue(stub.calls[0][1]["shuffle"])
        self.assertEqual(stub.calls[0][1]["batch_size"], R2_BATCH_SIZE)
        self.assertNotIn("validation_split", stub.calls[0][1])

    def test_23_validation_tuple_correctly_passed(self):
        stub = FitStub()
        fit_train_validation(
            stub,
            training=self.training,
            validation=self.validation,
            callbacks=(),
            epochs=7,
            verbose=0,
        )
        validation_data = stub.calls[0][1]["validation_data"]
        self.assertIs(validation_data[0], self.validation.X_seq)
        self.assertIs(validation_data[1], self.validation.y_seq)

    def test_24_fit_rejects_test_split(self):
        self.assertNotIn("test", inspect.signature(fit_train_validation).parameters)
        with self.assertRaises(RuntimeContractError):
            fit_train_validation(
                FitStub(),
                training=self.test,
                validation=self.validation,
                callbacks=(),
                epochs=1,
            )

    def test_25_evaluation_rejects_training_split(self):
        with self.assertRaises(RuntimeContractError):
            evaluate_test(self.reloaded, test=self.training)

    def test_26_evaluation_rejects_validation_split(self):
        with self.assertRaises(RuntimeContractError):
            evaluate_test(self.reloaded, test=self.validation)

    def test_27_evaluation_rejects_non_reloaded_model(self):
        with self.assertRaises(RuntimeContractError):
            evaluate_test(self.source, test=self.test)

    def test_28_evaluation_accepts_reloaded_checkpoint_wrapper(self):
        result = evaluate_test(self.reloaded, test=self.test)
        self.assertEqual(result.checkpoint_path, self.paths.source_pretrain)
        self.assertEqual(len(result.predictions), len(self.test.y_seq))
        np.testing.assert_array_equal(result.y_true, self.test.y_seq)

    def test_29_pythonhashseed_status_is_checked_and_recorded(self):
        before = os.environ.get("PYTHONHASHSEED")
        status = configure_reproducibility(1234)
        self.assertEqual(status.pythonhashseed, before)
        self.assertEqual(status.pythonhashseed_matches_seed, before == "1234")
        self.assertEqual(status.strict_hash_reproducibility, before == "1234")
        self.assertTrue(status.deterministic_ops_enabled)

    def test_30_zero_real_training_epochs_executed(self):
        stub = FitStub()
        fit_train_validation(
            stub,
            training=self.training,
            validation=self.validation,
            callbacks=(),
            epochs=99,
            verbose=0,
        )
        self.assertEqual(stub.real_epochs_executed, 0)
        self.assertEqual(len(stub.calls), 1)


if __name__ == "__main__":
    unittest.main()
