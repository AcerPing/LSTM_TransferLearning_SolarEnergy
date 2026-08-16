"""Zero-epoch contracts for R2.5 Phase-D formal validation."""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from r2_config.solar_r25 import PRIMARY_STRATEGY_ID, R25_MAX_EPOCHS
from r2_helpers.solar_partial_formal import (
    FORMAL_CHECKPOINT_TEMPLATE,
    PartialFormalError,
    create_partial_formal_root,
    formal_checkpoint_pattern,
    generate_formal_run_id,
    make_formal_partial_callbacks,
    prepare_formal_checkpoint_directory,
    run_partial_formal_validation,
)
from r2_helpers.solar_partial_smoke import prepare_target_training_validation


class SolarR25PartialFormalContracts(unittest.TestCase):
    def test_01_candidate_b_only(self):
        self.assertEqual(PRIMARY_STRATEGY_ID, "partial_target_adapters_last_lstm")

    def test_02_max_epochs_fixed(self):
        self.assertEqual(R25_MAX_EPOCHS, 500)

    def test_03_run_id_utc(self):
        value = generate_formal_run_id(datetime(2026, 8, 16, 2, 3, 4, tzinfo=timezone.utc))
        self.assertEqual(value, "20260816T020304Z_seed1234")

    def test_04_formal_root_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            create_partial_formal_root("20260816T020304Z_seed1234", output_base=base)
            with self.assertRaises(PartialFormalError):
                create_partial_formal_root("20260816T020304Z_seed1234", output_base=base)

    def test_05_formal_separate_from_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_partial_formal_root(
                "20260816T020304Z_seed1234", output_base=Path(temporary)
            )
            self.assertNotIn("_smoke", root.parts)

    def test_06_checkpoint_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            pattern = formal_checkpoint_pattern(Path(temporary))
            self.assertEqual(pattern.name, FORMAL_CHECKPOINT_TEMPLATE)
            self.assertIn(PRIMARY_STRATEGY_ID, pattern.parts)

    def test_07_checkpoint_parent_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            pattern = prepare_formal_checkpoint_directory(Path(temporary))
            self.assertTrue(pattern.parent.is_dir())
            with self.assertRaises(PartialFormalError):
                prepare_formal_checkpoint_directory(Path(temporary))

    def test_08_callbacks_exact(self):
        callbacks = make_formal_partial_callbacks(
            Path(tempfile.gettempdir()) / FORMAL_CHECKPOINT_TEMPLATE
        )
        self.assertEqual([type(item).__name__ for item in callbacks], [
            "ReduceLROnPlateau", "ModelCheckpoint", "EarlyStopping"
        ])
        self.assertEqual(callbacks[0].monitor, "val_loss")
        self.assertEqual(callbacks[0].factor, 0.5)
        self.assertEqual(callbacks[0].patience, 4)
        self.assertEqual(callbacks[0].min_lr, 1e-7)
        self.assertEqual(callbacks[1].monitor, "val_loss")
        self.assertTrue(callbacks[1].save_best_only)
        self.assertFalse(callbacks[1].save_weights_only)
        self.assertEqual(callbacks[2].monitor, "val_loss")
        self.assertEqual(callbacks[2].patience, 10)
        self.assertTrue(callbacks[2].restore_best_weights)

    def test_09_target_loader_roles(self):
        data = prepare_target_training_validation()
        self.assertEqual(len(data.training.named.X_seq), 516)
        self.assertEqual(len(data.validation.named.X_seq), 126)
        self.assertTrue(all("test" not in name for name in data.checksums))

    def test_10_runner_has_no_test_api(self):
        source = inspect.getsource(run_partial_formal_validation)
        self.assertNotIn("load_formal_test", source)
        self.assertNotIn("evaluate_test", source)
        self.assertNotIn("transfer_interpretation", source)

    def test_11_zero_epoch_suite_does_not_call_runner(self):
        self.assertTrue(callable(run_partial_formal_validation))


if __name__ == "__main__":
    unittest.main()
