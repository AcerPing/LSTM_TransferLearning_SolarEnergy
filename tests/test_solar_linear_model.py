from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Importing the Linear runtime installs the same Keras compatibility shim used
# by the existing R2 runtime before the unchanged Legacy import is evaluated.
from r2_helpers import solar_linear_runtime as linear_runtime
from utils.model import build_model


EXPECTED_CLASSES = (
    "InputLayer",
    "TimeDistributed",
    "LSTM",
    "BatchNormalization",
    "LSTM",
    "BatchNormalization",
    "Dense",
)


def build_quiet(root: Path, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_model(
            input_shape=(5, 5),
            gpu=False,
            write_result_out_dir=str(root),
            noise=None,
            verbose=False,
            savefig=False,
            **kwargs,
        )


class SolarLinearModelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.legacy = build_quiet(cls.root)
        cls.explicit_sigmoid = build_quiet(cls.root, output_activation="sigmoid")
        cls.linear = build_quiet(
            cls.root,
            output_activation="linear",
            learning_rate=2.5e-5,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_legacy_default_is_sigmoid(self):
        self.assertEqual(self.legacy.layers[6].activation.__name__, "sigmoid")
        self.assertEqual(self.legacy.count_params(), 46681)

    def test_02_explicit_sigmoid_is_compatible(self):
        self.assertEqual(self.explicit_sigmoid.layers[6].activation.__name__, "sigmoid")
        self.assertEqual(self.explicit_sigmoid.count_params(), 46681)

    def test_03_linear_build(self):
        self.assertEqual(self.linear.layers[6].activation.__name__, "linear")
        self.assertEqual(self.linear.count_params(), 46681)
        linear_runtime.validate_linear_model(self.linear, expected_learning_rate=2.5e-5)

    def test_04_exact_topology_and_weight_shapes(self):
        for model in (self.legacy, self.explicit_sigmoid, self.linear):
            self.assertEqual(tuple(type(layer).__name__ for layer in model.layers), EXPECTED_CLASSES)
            self.assertEqual(len(model.layers), 7)
        actual = {
            index: tuple(tuple(weight.shape) for weight in layer.weights)
            for index, layer in enumerate(self.linear.layers)
        }
        self.assertEqual(actual, dict(linear_runtime.EXPECTED_WEIGHT_SHAPES))

    def test_05_legacy_fresh_default_lr(self):
        self.assertAlmostEqual(float(self.legacy.optimizer.learning_rate.numpy()), 1e-4, places=11)

    def test_06_explicit_lr_wins(self):
        self.assertTrue(
            np.isclose(
                float(self.linear.optimizer.learning_rate.numpy()),
                2.5e-5,
                rtol=0.0,
                atol=1e-10,
            )
        )

    def test_07_legacy_tl_lr_and_copy_scope(self):
        target_initial = build_quiet(self.root)
        target = build_quiet(self.root, pre_model=self.legacy)
        self.assertEqual(target.layers[6].activation.__name__, "sigmoid")
        self.assertAlmostEqual(float(target.optimizer.learning_rate.numpy()), 1e-5, places=12)
        for index in (2, 3, 4, 5):
            for source_weight, target_weight in zip(
                self.legacy.layers[index].get_weights(), target.layers[index].get_weights()
            ):
                np.testing.assert_array_equal(source_weight, target_weight)
        for index in (1, 6):
            for initial_weight, target_weight in zip(
                target_initial.layers[index].get_weights(), target.layers[index].get_weights()
            ):
                np.testing.assert_array_equal(initial_weight, target_weight)

    def test_08_existing_r2_imports_remain_compatible(self):
        from r2_config import solar_r2, solar_r25
        from r2_helpers import solar_partial_ft, solar_runtime

        self.assertEqual(solar_r2.WINDOW, 5)
        self.assertTrue(callable(solar_runtime.build_r2_model))
        self.assertTrue(callable(solar_partial_ft.build_partial_candidate))
        self.assertIsNotNone(solar_r25.PARTIAL_STRATEGY_SPECS)

    def test_09_invalid_activation_and_lr_fail(self):
        with self.assertRaises(ValueError):
            build_quiet(self.root, output_activation="relu")
        with self.assertRaises(ValueError):
            build_quiet(self.root, learning_rate=0)


if __name__ == "__main__":
    unittest.main()
