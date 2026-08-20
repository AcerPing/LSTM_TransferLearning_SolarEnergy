from __future__ import annotations

import contextlib
import inspect
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2_helpers import solar_linear_runtime as runtime
from utils.model import build_model


class SolarLinearRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.source = runtime.build_linear_source_model(
                output_dir=cls.root, learning_rate=1e-4
            )
            cls.without_tl = runtime.build_linear_without_tl_model(
                output_dir=cls.root, learning_rate=1e-4
            )

            # Make Source adapters visibly different.  Candidate layers 1 and
            # 6 must still retain fresh target initialization.
            for index in (1, 6):
                cls.source.layers[index].set_weights(
                    [np.full_like(weight, 0.321) for weight in cls.source.layers[index].get_weights()]
                )
            cls.candidate = runtime.build_linear_partial_ft_candidate(
                cls.source,
                output_dir=cls.root,
                learning_rate=1e-5,
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_fresh_source_and_without_tl_are_linear(self):
        for model in (self.source, self.without_tl):
            runtime.validate_linear_model(model)
            self.assertEqual(model.layers[6].activation.__name__, "linear")
            self.assertEqual(model.count_params(), 46681)

    def test_02_exact_transfer_scope(self):
        for index in (2, 3, 4, 5):
            self.assertEqual(
                type(self.source.layers[index]).__name__,
                type(self.candidate.model.layers[index]).__name__,
            )
            for source_weight, target_weight in zip(
                self.source.layers[index].get_weights(),
                self.candidate.model.layers[index].get_weights(),
            ):
                np.testing.assert_array_equal(source_weight, target_weight)

    def test_03_output_and_input_adapter_are_not_transferred(self):
        for index in (1, 6):
            target_weights = self.candidate.model.layers[index].get_weights()
            for initial_weight, target_weight in zip(
                self.candidate.initial_non_transferred_weights[index], target_weights
            ):
                np.testing.assert_array_equal(initial_weight, target_weight)
            self.assertTrue(
                any(
                    not np.array_equal(source_weight, target_weight)
                    for source_weight, target_weight in zip(
                        self.source.layers[index].get_weights(), target_weights
                    )
                )
            )

    def test_04_trainable_and_frozen_masks(self):
        model = self.candidate.model
        actual_trainable = tuple(index for index in range(1, 7) if model.layers[index].trainable)
        actual_frozen = tuple(index for index in range(1, 7) if not model.layers[index].trainable)
        self.assertEqual(actual_trainable, (1, 4, 6))
        self.assertEqual(actual_frozen, (2, 3, 5))

    def test_05_batch_normalization_is_frozen(self):
        for index in (3, 5):
            self.assertEqual(type(self.candidate.model.layers[index]).__name__, "BatchNormalization")
            self.assertFalse(self.candidate.model.layers[index].trainable)

    def test_06_parameter_counts(self):
        self.assertEqual(
            self.candidate.parameter_counts,
            runtime.ParameterCounts(total=46681, trainable=29161, non_trainable=17520),
        )

    def test_07_class_and_weight_shape_keys(self):
        runtime.validate_linear_model(self.candidate.model, expected_learning_rate=1e-5)
        self.assertEqual(
            tuple(type(layer).__name__ for layer in self.candidate.model.layers),
            runtime.EXPECTED_LAYER_CLASSES,
        )
        self.assertEqual(
            tuple(tuple(weight.shape) for weight in self.candidate.model.layers[6].weights),
            ((60, 1), (1,)),
        )

    def test_08_reject_sigmoid_source(self):
        with contextlib.redirect_stdout(io.StringIO()):
            sigmoid = build_model(
                input_shape=(5, 5),
                gpu=False,
                write_result_out_dir=str(self.root),
                noise=None,
                verbose=False,
                savefig=False,
            )
            with self.assertRaises(runtime.LinearRuntimeContractError):
                runtime.build_linear_partial_ft_candidate(
                    sigmoid, output_dir=self.root, learning_rate=1e-5
                )

    def test_09_explicit_lr_is_required(self):
        with self.assertRaises(runtime.LinearRuntimeContractError):
            runtime.build_linear_source_model(output_dir=self.root, learning_rate=None)

    def test_10_runtime_exposes_no_training_or_checkpoint_api(self):
        source = inspect.getsource(runtime)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("train_on_batch", source)
        self.assertNotIn("ModelCheckpoint", source)
        self.assertFalse(any(name.startswith("evaluate") for name in vars(runtime)))


if __name__ == "__main__":
    unittest.main()
