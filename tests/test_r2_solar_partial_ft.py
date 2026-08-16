"""Zero-epoch tests for the Solar R2.5 partial fine-tuning contract."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Keep this zero-epoch contract suite on the approved device before Keras imports.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model

from r2_config.solar_r25 import (
    BASELINE_RUN_ROOT,
    BASELINE_SOURCE_CHECKPOINT_SHA256,
    BATCH_NORMALIZATION_INDICES,
    EXPECTED_LAYER_CLASSES,
    EXPECTED_PARAMETER_COUNTS,
    PARTIAL_STRATEGY_ORDER,
    PRIMARY_STRATEGY_ID,
    R25_BATCH_SIZE,
    R25_LEARNING_RATE,
    R25_LOSS,
    R25_MAX_EPOCHS,
    R25_OUTPUT_BASE,
    R25_SEED,
    TARGET_ADAPTER_INDICES,
    TRANSFERRED_LAYER_INDICES,
)
from r2_helpers.solar_partial_ft import (
    STRATEGY_REGISTRY,
    PartialFTContractError,
    ValidationScore,
    apply_partial_strategy,
    authorize_selected_test,
    batch_normalization_state,
    build_partial_candidate,
    candidate_checkpoint_pattern,
    compile_partial_model,
    inverse_transform_original,
    layer_contract_manifest,
    load_selection_lock,
    lock_selection,
    make_partial_callbacks,
    parameter_counts,
    select_validation_winner,
    sha256_file,
    validate_baseline_source_checkpoint,
    validate_exact_topology,
    validate_partial_transfer,
    validate_r25_output_root,
    zero_epoch_contract_summary,
)
from r2_helpers.solar_runtime import (
    NON_TRANSFERRED_WEIGHT_LAYER_INDICES,
    R2_INPUT_SHAPE,
    TRANSFERABLE_LAYER_INDICES,
    ReloadedCheckpointModel,
    build_r2_model,
    configure_reproducibility,
    reload_best_checkpoint,
    validate_transferred_weights,
)


class SolarR25PartialFTZeroEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        cls.baseline = validate_baseline_source_checkpoint()
        cls.baseline_hash_before = sha256_file(cls.baseline.checkpoint_path)
        cls.source_wrapper = reload_best_checkpoint(cls.baseline.checkpoint_path)
        cls.source = cls.source_wrapper.model

        configure_reproducibility(R25_SEED)
        cls.target_initial = build_r2_model(output_dir=cls.root)
        cls.candidates = {
            strategy_id: build_partial_candidate(
                cls.source,
                strategy_id=strategy_id,
                output_dir=cls.root,
            )
            for strategy_id in PARTIAL_STRATEGY_ORDER
        }

        configure_reproducibility(R25_SEED)
        cls.freeze = build_r2_model(
            output_dir=cls.root,
            pretrained_model=cls.source,
            freeze=True,
        )
        configure_reproducibility(R25_SEED)
        cls.full = build_r2_model(
            output_dir=cls.root,
            pretrained_model=cls.source,
            freeze=False,
        )

    @classmethod
    def tearDownClass(cls):
        after = sha256_file(cls.baseline.checkpoint_path)
        if after != cls.baseline_hash_before:
            raise AssertionError("Existing R2 baseline checkpoint changed")
        cls.temporary.cleanup()
        K.clear_session()

    def _score(
        self,
        strategy_id,
        *,
        rmse=10.0,
        mae=8.0,
        r2=0.5,
        split_name="validation",
        checkpoint_path=None,
    ):
        return ValidationScore(
            strategy_id=strategy_id,
            split_name=split_name,
            scale="original",
            rmse=rmse,
            mae=mae,
            r2=r2,
            best_epoch=7,
            checkpoint_path=checkpoint_path or self.root / f"{strategy_id}.hdf5",
        )

    def test_01_baseline_run_status_pass(self):
        manifest = json.loads(self.baseline.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(self.baseline.run_root, BASELINE_RUN_ROOT.resolve())

    def test_02_source_checkpoint_path_and_sha256(self):
        self.assertEqual(self.baseline.best_epoch, 472)
        self.assertEqual(self.baseline.sha256, BASELINE_SOURCE_CHECKPOINT_SHA256)
        self.assertEqual(sha256_file(self.baseline.checkpoint_path), self.baseline.sha256)

    def test_03_exact_seven_layer_topology(self):
        validate_exact_topology(self.source)
        self.assertEqual(len(self.source.layers), 7)
        self.assertEqual(
            tuple(type(layer).__name__ for layer in self.source.layers),
            EXPECTED_LAYER_CLASSES,
        )

    def test_04_expected_layer_classes_are_index_asserted(self):
        manifest = layer_contract_manifest(self.source)
        self.assertEqual(tuple(row["index"] for row in manifest), tuple(range(7)))
        self.assertEqual(tuple(row["class_name"] for row in manifest), EXPECTED_LAYER_CLASSES)
        self.assertTrue(all(row["runtime_name"] for row in manifest))

    def test_05_sigmoid_output_unchanged(self):
        self.assertEqual(self.source.layers[6].activation.__name__, "sigmoid")
        for candidate in self.candidates.values():
            self.assertEqual(candidate.model.layers[6].activation.__name__, "sigmoid")

    def test_06_transfer_scope_remains_two_through_five(self):
        self.assertEqual(TRANSFERRED_LAYER_INDICES, (2, 3, 4, 5))
        self.assertEqual(TRANSFERABLE_LAYER_INDICES, TRANSFERRED_LAYER_INDICES)
        self.assertEqual(TARGET_ADAPTER_INDICES, (1, 6))
        self.assertEqual(NON_TRANSFERRED_WEIGHT_LAYER_INDICES, TARGET_ADAPTER_INDICES)

    def test_07_strategy_b_indices_exact(self):
        strategy = STRATEGY_REGISTRY[PRIMARY_STRATEGY_ID]
        self.assertEqual(strategy.trainable_indices, (1, 4, 6))
        self.assertEqual(strategy.frozen_indices, (2, 3, 5))

    def test_08_registry_contains_candidate_b_only(self):
        self.assertEqual(tuple(STRATEGY_REGISTRY), (PRIMARY_STRATEGY_ID,))

    def test_09_batch_normalization_frozen_for_candidate_b(self):
        for candidate in self.candidates.values():
            before = batch_normalization_state(candidate.model)
            for index in BATCH_NORMALIZATION_INDICES:
                self.assertFalse(candidate.model.layers[index].trainable)
                self.assertEqual(len(before[index]), 4)
                self.assertTrue(
                    all(
                        any(
                            weight is non_trainable
                            for non_trainable in candidate.model.non_trainable_weights
                        )
                        for weight in candidate.model.layers[index].weights
                    )
                )

    def test_10_runtime_parameter_counts_exact(self):
        for strategy_id, candidate in self.candidates.items():
            counts = parameter_counts(candidate.model)
            self.assertEqual(counts.total_params, EXPECTED_PARAMETER_COUNTS["total"])
            self.assertEqual(counts.trainable_params, EXPECTED_PARAMETER_COUNTS[strategy_id])
            self.assertEqual(
                counts.non_trainable_params,
                counts.total_params - counts.trainable_params,
            )
            self.assertAlmostEqual(
                counts.trainable_percentage,
                100.0 * counts.trainable_params / counts.total_params,
            )

    def test_11_source_transferred_weights_equal_before_training(self):
        for candidate in self.candidates.values():
            validate_partial_transfer(
                self.source,
                candidate.model,
                target_initial_model=self.target_initial,
            )

    def test_12_target_adapters_are_not_source_transferred(self):
        configure_reproducibility(R25_SEED)
        altered_source = build_r2_model(output_dir=self.root)
        for index in TARGET_ADAPTER_INDICES:
            altered_source.layers[index].set_weights(
                [np.full_like(weight, 7.25) for weight in altered_source.layers[index].get_weights()]
            )
        candidate = build_partial_candidate(
            altered_source,
            strategy_id=PRIMARY_STRATEGY_ID,
            output_dir=self.root,
        )
        validate_partial_transfer(
            altered_source,
            candidate.model,
            target_initial_model=self.target_initial,
        )
        for index in TARGET_ADAPTER_INDICES:
            self.assertTrue(
                any(
                    not np.array_equal(source_weight, target_weight)
                    for source_weight, target_weight in zip(
                        altered_source.layers[index].get_weights(),
                        candidate.model.layers[index].get_weights(),
                    )
                )
            )

    def test_13_fresh_optimizer_and_recompile_contract(self):
        first = self.candidates[PRIMARY_STRATEGY_ID]
        second = build_partial_candidate(
            self.source,
            strategy_id=PRIMARY_STRATEGY_ID,
            output_dir=self.root,
        )
        self.assertIsNot(first.model.optimizer, second.model.optimizer)
        self.assertNotEqual(first.optimizer_identity, second.optimizer_identity)
        model = build_r2_model(output_dir=self.root, pretrained_model=self.source)
        original_optimizer = model.optimizer
        apply_partial_strategy(model, PRIMARY_STRATEGY_ID)
        identity = compile_partial_model(model)
        self.assertIsNot(model.optimizer, original_optimizer)
        self.assertEqual(identity, id(model.optimizer))

    def test_14_learning_rate_unchanged(self):
        self.assertEqual(R25_LEARNING_RATE, 1e-5)
        for candidate in self.candidates.values():
            self.assertAlmostEqual(
                float(candidate.model.optimizer.learning_rate.numpy()),
                R25_LEARNING_RATE,
                places=12,
            )

    def test_15_mse_loss_unchanged(self):
        self.assertEqual(R25_LOSS, "mse")
        for candidate in self.candidates.values():
            self.assertIn(candidate.model.loss, ("mse", "mean_squared_error"))

    def test_16_batch_size_unchanged(self):
        self.assertEqual(R25_BATCH_SIZE, 128)

    def test_17_seed_unchanged(self):
        self.assertEqual(R25_SEED, 1234)
        self.assertTrue(all(candidate.seed == 1234 for candidate in self.candidates.values()))

    def test_18_existing_freeze_behavior_regression(self):
        validate_transferred_weights(
            self.source, self.freeze, self.target_initial, freeze=True
        )
        self.assertEqual(parameter_counts(self.freeze).trainable_params, 121)

    def test_19_existing_full_finetune_behavior_regression(self):
        validate_transferred_weights(
            self.source, self.full, self.target_initial, freeze=False
        )
        self.assertEqual(parameter_counts(self.full).trainable_params, 46441)

    def test_20_test_rejected_before_selection_lock(self):
        missing = self.root / "missing" / "selection.json"
        with self.assertRaises(PartialFTContractError):
            authorize_selected_test(
                missing,
                strategy_id=PRIMARY_STRATEGY_ID,
                checkpoint_model=self.source_wrapper,
            )

    def test_21_validation_selector_rejects_test_metrics(self):
        scores = [self._score(PRIMARY_STRATEGY_ID, split_name="test")]
        with self.assertRaises(PartialFTContractError):
            select_validation_winner(scores)

    def test_22_validation_selection_accepts_candidate_b_only(self):
        primary = self._score(PRIMARY_STRATEGY_ID, rmse=10, mae=8, r2=0.5)
        decision = select_validation_winner((primary,))
        self.assertEqual(decision.winner, primary)
        with self.assertRaises(PartialFTContractError):
            select_validation_winner(
                (self._score("not_approved_strategy", rmse=9),)
            )

    def test_23_selection_json_immutable_once_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            decision = select_validation_winner(
                (self._score(PRIMARY_STRATEGY_ID),)
            )
            lock_selection(
                path,
                decision,
                source_checkpoint_sha256=BASELINE_SOURCE_CHECKPOINT_SHA256,
            )
            before = path.read_bytes()
            with self.assertRaises(PartialFTContractError):
                lock_selection(
                    path,
                    decision,
                    source_checkpoint_sha256=BASELINE_SOURCE_CHECKPOINT_SHA256,
                )
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(load_selection_lock(path)["locked"])
            self.assertNotIn("overwrite", inspect.signature(lock_selection).parameters)

    def test_24_only_selected_strategy_can_enter_final_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            decision = select_validation_winner(
                (
                    self._score(
                        PRIMARY_STRATEGY_ID,
                        checkpoint_path=self.baseline.checkpoint_path,
                    ),
                )
            )
            lock_selection(path, decision, source_checkpoint_sha256=self.baseline.sha256)
            with self.assertRaises(PartialFTContractError):
                authorize_selected_test(
                    path,
                    strategy_id="not_approved_strategy",
                    checkpoint_model=self.source_wrapper,
                )
            accepted = authorize_selected_test(
                path,
                strategy_id=PRIMARY_STRATEGY_ID,
                checkpoint_model=self.source_wrapper,
            )
            self.assertEqual(accepted.strategy_id, PRIMARY_STRATEGY_ID)

    def test_25_final_test_requires_reloaded_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            decision = select_validation_winner(
                (
                    self._score(
                        PRIMARY_STRATEGY_ID,
                        checkpoint_path=self.baseline.checkpoint_path,
                    ),
                )
            )
            lock_selection(path, decision, source_checkpoint_sha256=self.baseline.sha256)
            with self.assertRaises(PartialFTContractError):
                authorize_selected_test(
                    path,
                    strategy_id=PRIMARY_STRATEGY_ID,
                    checkpoint_model=self.source,
                )

    def test_26_unique_checkpoint_namespace(self):
        patterns = [
            candidate_checkpoint_pattern(self.root, strategy_id)
            for strategy_id in PARTIAL_STRATEGY_ORDER
        ]
        self.assertEqual(len(patterns), 1)
        self.assertEqual(len(set(patterns)), 1)
        for strategy_id, pattern in zip(PARTIAL_STRATEGY_ORDER, patterns):
            self.assertIn(strategy_id, pattern.parts)
            self.assertEqual(pattern.name, "checkpoint_epoch_{epoch:04d}.hdf5")
        with self.assertRaises(PartialFTContractError):
            candidate_checkpoint_pattern(self.root, "not_approved_strategy")

    def test_27_no_overwrite_existing_r2_reports(self):
        self.assertFalse(R25_OUTPUT_BASE.resolve().is_relative_to(BASELINE_RUN_ROOT.resolve()))
        with self.assertRaises(PartialFTContractError):
            validate_r25_output_root(BASELINE_RUN_ROOT / "forbidden")
        approved = R25_OUTPUT_BASE / "20990101T000000Z_seed1234"
        if not approved.exists():
            self.assertEqual(validate_r25_output_root(approved), approved.resolve())

    def test_28_original_scale_inverse_transform_contract(self):
        scaler = SimpleNamespace(
            inverse_transform=lambda values: np.asarray(values) * 100.0 + 7.0
        )
        normalized = np.array([0.0, 0.5, 1.0])
        expected = np.array([7.0, 57.0, 107.0])
        np.testing.assert_allclose(
            inverse_transform_original(
                normalized,
                scaler,
                expected_original=expected,
            ),
            expected,
        )

    def test_29_zero_epoch_contract_summary(self):
        with patch.object(Model, "fit", side_effect=AssertionError("fit must not run")):
            summary = zero_epoch_contract_summary(self.root)
        self.assertEqual(summary["training_epochs_executed"], 0)
        self.assertEqual(tuple(summary["strategies"]), PARTIAL_STRATEGY_ORDER)

    def test_30_zero_training_epochs_executed(self):
        source = inspect.getsource(zero_epoch_contract_summary)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("fit_train_validation", source)
        self.assertNotIn("Target Test", source)

    def test_31_callback_and_max_epoch_protocol(self):
        self.assertEqual(R25_MAX_EPOCHS, 500)
        callbacks = make_partial_callbacks(self.root, PRIMARY_STRATEGY_ID)
        self.assertEqual(
            tuple(type(callback) for callback in callbacks),
            (ReduceLROnPlateau, ModelCheckpoint, EarlyStopping),
        )
        reduce_lr, checkpoint, early = callbacks
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


if __name__ == "__main__":
    unittest.main()
