"""Build-only contracts for the R2.5 Phase-C training smoke runtime."""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from r2_config.solar_r25 import (  # noqa: E402
    BATCH_NORMALIZATION_INDICES,
    PARTIAL_STRATEGY_ORDER,
    PRIMARY_STRATEGY_ID,
)
from r2_helpers.solar_partial_smoke import (  # noqa: E402
    PARTIAL_SMOKE_EPOCHS,
    SMOKE_TARGET_CHECKSUM_FILES,
    PartialSmokeError,
    create_smoke_root,
    generate_smoke_run_id,
    prepare_candidate_checkpoint_directory,
    prepare_target_training_validation,
    run_partial_smoke,
    validation_smoke_metrics,
    verify_bn_unchanged,
    verify_layer_changes,
)


class FakeLayer:
    def __init__(self, name, weights, trainable):
        self.name = name
        self._weights = [np.asarray(value).copy() for value in weights]
        self.trainable = trainable

    def get_weights(self):
        return [value.copy() for value in self._weights]


class SolarR25PartialSmokeContractTests(unittest.TestCase):
    def test_01_exactly_two_epochs(self):
        self.assertEqual(PARTIAL_SMOKE_EPOCHS, 2)

    def test_02_candidate_b_is_the_only_approved_strategy(self):
        self.assertEqual(PARTIAL_STRATEGY_ORDER, (PRIMARY_STRATEGY_ID,))

    def test_03_run_id_is_deterministic_utc_format(self):
        value = generate_smoke_run_id(datetime(2026, 8, 16, 1, 2, 3, tzinfo=timezone.utc))
        self.assertEqual(value, "20260816T010203Z_seed1234")

    def test_04_smoke_root_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = create_smoke_root("20260816T010203Z_seed1234", output_base=base)
            self.assertTrue(root.is_dir())
            with self.assertRaises(PartialSmokeError):
                create_smoke_root("20260816T010203Z_seed1234", output_base=base)

    def test_05_checkpoint_parent_explicitly_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pattern = prepare_candidate_checkpoint_directory(root, PRIMARY_STRATEGY_ID)
            self.assertTrue(pattern.parent.is_dir())
            self.assertFalse(pattern.exists())
            with self.assertRaises(PartialSmokeError):
                prepare_candidate_checkpoint_directory(root, PRIMARY_STRATEGY_ID)

    def test_06_layer_change_contract_for_candidate_b(self):
        layers = [FakeLayer("input", (), False)]
        for index in range(1, 7):
            layers.append(FakeLayer(f"layer_{index}", ([float(index)],), index in (1, 4, 6)))
        model = SimpleNamespace(layers=layers)
        before = {index: tuple(layers[index].get_weights()) for index in range(1, 7)}
        for index in (1, 4, 6):
            layers[index]._weights[0] += 1.0
        result = verify_layer_changes(model, before, PRIMARY_STRATEGY_ID)
        self.assertTrue(all(row["verified"] for row in result.values()))

    def test_07_unapproved_strategy_is_rejected(self):
        layers = [FakeLayer("input", (), False)]
        for index in range(1, 7):
            layers.append(FakeLayer(f"layer_{index}", ([float(index)],), index in (1, 4, 6)))
        model = SimpleNamespace(layers=layers)
        before = {index: tuple(layers[index].get_weights()) for index in range(1, 7)}
        with self.assertRaises(KeyError):
            verify_layer_changes(model, before, "partial_all_non_bn")

    def test_08_frozen_change_is_rejected(self):
        layers = [FakeLayer("input", (), False)]
        for index in range(1, 7):
            layers.append(FakeLayer(f"layer_{index}", ([float(index)],), index in (1, 4, 6)))
        model = SimpleNamespace(layers=layers)
        before = {index: tuple(layers[index].get_weights()) for index in range(1, 7)}
        for index in (1, 2, 4, 6):
            layers[index]._weights[0] += 1.0
        with self.assertRaises(PartialSmokeError):
            verify_layer_changes(model, before, PRIMARY_STRATEGY_ID)

    def test_09_validation_metrics_are_smoke_only(self):
        target_scaler = SimpleNamespace(
            inverse_transform=lambda values: np.asarray(values) * 10.0
        )
        validation = SimpleNamespace(
            named=SimpleNamespace(y_seq=np.array([0.1, 0.2])),
            split_data=SimpleNamespace(y_original=np.array([1.0, 2.0])),
            sequences=SimpleNamespace(target_row_index=np.array([0, 1])),
        )
        role = SimpleNamespace(
            validation=validation,
            scalers=SimpleNamespace(target_scaler=target_scaler),
        )
        metrics = validation_smoke_metrics(role, np.array([0.1, 0.2]))
        self.assertTrue(metrics["SMOKE_ONLY"])
        self.assertEqual(metrics["split"], "validation")
        self.assertEqual(metrics["original"]["RMSE"], 0.0)

    def test_10_runtime_does_not_reference_test_loader_or_evaluator(self):
        source = inspect.getsource(run_partial_smoke)
        self.assertNotIn("load_formal_test", source)
        self.assertNotIn("evaluate_test", source)
        self.assertNotIn("transfer_interpretation", source)

    def test_11_target_loader_never_opens_test_csv(self):
        with patch.object(pd, "read_csv", wraps=pd.read_csv) as reader:
            data = prepare_target_training_validation()
        opened = [Path(call.args[0]).name for call in reader.call_args_list]
        self.assertEqual(
            set(opened),
            {
                "normalized_scale_training.csv",
                "original_scale_training.csv",
                "normalized_scale_validation.csv",
                "original_scale_validation.csv",
            },
        )
        self.assertTrue(all("test" not in name for name in opened))
        self.assertTrue(all("test" not in name for name in data.checksums))
        self.assertTrue(all("test" not in name for name in SMOKE_TARGET_CHECKSUM_FILES))


if __name__ == "__main__":
    unittest.main()
