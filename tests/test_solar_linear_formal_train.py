from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from r2_config.solar_linear import LINEAR_EXPERIMENTS, LINEAR_FORMAL_BASE
from r2_config.solar_linear_formal import (
    FORMAL_CALLBACK_POLICY,
    FORMAL_PROTOCOL_VERSION,
    FormalPathContract,
    candidate_registry,
)
from r2_helpers import solar_linear_formal as protocol
from r2_helpers import solar_linear_formal_train as train


class FakeModel:
    def __init__(self):
        self.fit_calls = []
        self.optimizer = SimpleNamespace(learning_rate=np.float32(1e-4), lr=np.float32(1e-4))

    def fit(self, X, y, **kwargs):
        self.fit_calls.append((X, y, kwargs))
        recorder = kwargs["callbacks"][-1]
        recorder.set_model(self)
        recorder.on_epoch_begin(0)
        recorder.on_epoch_begin(1)
        return SimpleNamespace(
            history={"loss": [0.4, 0.3], "val_loss": [0.5, 0.2]}
        )


class IdentityScale:
    def inverse_transform(self, values):
        return np.asarray(values, dtype=float) * 100.0


class PredictModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float).reshape(-1, 1)
        self.calls = []

    def predict(self, X, **kwargs):
        self.calls.append((X, kwargs))
        return self.values.copy()


class FakeLayer:
    def __init__(self, values):
        self.values = tuple(np.asarray(value).copy() for value in values)

    def get_weights(self):
        return tuple(value.copy() for value in self.values)


class SolarLinearFormalTrainTests(unittest.TestCase):
    RUN_ID = "20991231T235959Z_seed1234"
    GIT_HEAD = "a" * 40

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        experiment_root = base / "Experiment_B"
        run_root = experiment_root / self.RUN_ID
        self.paths = FormalPathContract(
            experiment_id="B",
            run_id=self.RUN_ID,
            experiment_root=experiment_root,
            run_root=run_root,
            protocol_manifest=run_root / "protocol_manifest.json",
            source_candidates=run_root / "source_candidates",
            target_wotl_candidates=run_root / "target_wotl_candidates",
            target_partial_ft_candidates=run_root / "target_partial_ft_candidates",
            selection=run_root / "selection",
            final=run_root / "final",
        )
        self.environment = train.FormalEnvironment(
            python="3.10.11",
            tensorflow="2.10.0",
            keras="2.10.0",
            numpy="1.23.5",
            device="CPU",
            cuda_visible_devices="-1",
            pythonhashseed="1234",
            seed=1234,
            deterministic_ops=True,
        )
        self.git = train.FormalGitProvenance(
            head=self.GIT_HEAD,
            branch="test",
            tracked_dirty=False,
            known_untracked_paths=("notebook/修改紀錄/evidence.md",),
            unknown_untracked_paths=(),
        )
        self.plan = train.prepare_formal_training_run(
            "B",
            self.RUN_ID,
            expected_git_head=self.GIT_HEAD,
            git_provenance=self.git,
            environment=self.environment,
            path_factory=lambda experiment_id, run_id: self.paths,
            destination_validator=lambda paths: None,
        )

    def _profile(self, role="target"):
        counts = {
            "source": (2088, 523, 2083, 518),
            "target": (521, 131, 516, 126),
        }[role]

        def split(rows, sequences):
            return SimpleNamespace(
                split_data=SimpleNamespace(row_count=rows),
                sequences=SimpleNamespace(
                    X_seq=np.zeros((sequences, 5, 5), dtype=float),
                    y_seq=np.linspace(0.1, 0.9, sequences),
                    count=sequences,
                ),
            )

        return SimpleNamespace(
            training=split(counts[0], counts[2]),
            validation=split(counts[1], counts[3]),
            scalers=SimpleNamespace(target_scaler=IdentityScale()),
        )

    def _score(self, lifecycle, learning_rate, loss, sha_char):
        identifier = dict(candidate_registry()[lifecycle])
        candidate_id = next(
            key for key, value in identifier.items() if value == learning_rate
        )
        directory = {
            "source": "source_candidates",
            "wotl": "target_wotl_candidates",
            "partial_ft": "target_partial_ft_candidates",
        }[lifecycle]
        checkpoint_path = (
            LINEAR_EXPERIMENTS["B"].output_root
            / self.RUN_ID
            / directory
            / candidate_id
            / "checkpoint_epoch_0002.hdf5"
        )
        return protocol.ValidationCandidateScore(
            protocol_version=FORMAL_PROTOCOL_VERSION,
            experiment_id="B",
            run_id=self.RUN_ID,
            git_head=self.GIT_HEAD,
            target_protocol_fingerprint=protocol.target_protocol_fingerprint("B"),
            lifecycle=lifecycle,
            candidate_id=candidate_id,
            learning_rate=learning_rate,
            best_epoch=2,
            validation_loss=loss,
            validation_original_mae=10.0,
            validation_original_rmse=12.0,
            validation_original_r2=0.8,
            trainable_params=29161 if lifecycle == "partial_ft" else 46681,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=sha_char * 64,
            checkpoint_sha256_verified=True,
        )

    def _selection(self):
        source = (
            self._score("source", 1e-4, 0.2, "1"),
            self._score("source", 3e-5, 0.3, "2"),
        )
        wotl = (
            self._score("wotl", 1e-4, 0.25, "3"),
            self._score("wotl", 3e-5, 0.35, "4"),
        )
        partial = (
            self._score("partial_ft", 1e-5, 0.22, "5"),
            self._score("partial_ft", 3e-5, 0.32, "6"),
        )
        evidence = protocol.SourceArchitectureEvidence(
            activation="linear",
            git_head=self.GIT_HEAD,
            layer_classes=protocol.FORMAL_LINEAR_LAYER_CLASSES,
            input_shape=(5, 5),
            output_shape=(1,),
            total_params=46681,
            formal_eligible=True,
        )
        return protocol.build_selection_record(
            experiment_id="B",
            selection_timestamp="2099-12-31T23:59:59Z",
            source_scores=source,
            source_architecture_evidence=evidence,
            expected_git_head=self.GIT_HEAD,
            wotl_scores=wotl,
            partial_ft_scores=partial,
        )

    def _base_manifest(self):
        identity = train.protocol_git_identity(self.plan)
        manifest, _ = protocol.build_protocol_manifest("B", self.paths, identity)
        return train.formal_manifest_for_training(self.plan, manifest)

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

    def test_01_a2_b_mapping(self):
        self.assertEqual((LINEAR_EXPERIMENTS["A2"].source_plant, LINEAR_EXPERIMENTS["A2"].target_plant), ("Plant1", "Plant2"))
        self.assertEqual((self.plan.spec.source_plant, self.plan.spec.target_plant), ("Plant2", "Plant1"))

    def test_02_candidate_registry_is_phase_iii_registry(self):
        self.assertEqual(
            tuple((item.candidate_id, item.learning_rate) for item in self.plan.candidates["source"]),
            candidate_registry()["source"],
        )
        self.assertEqual(tuple(len(self.plan.candidates[key]) for key in ("source", "wotl", "partial_ft")), (2, 2, 2))

    def test_03_formal_path_contract_is_reused(self):
        self.assertIs(self.plan.paths, self.paths)
        for lifecycle, entries in self.plan.candidates.items():
            for entry in entries:
                self.assertTrue(entry.candidate_root.is_relative_to(self.paths.run_root))
                self.assertIn(lifecycle, ("source", "wotl", "partial_ft"))

    def test_04_forbidden_split_filenames_are_rejected(self):
        for scale in ("normalized", "original"):
            with self.subTest(scale=scale), self.assertRaises(train.FormalTrainingError):
                train.guard_data_path(Path(self.temporary.name) / f"{scale}_scale_{'test'}.csv")

    def test_05_access_spy_blocks_before_open(self):
        path = Path(self.temporary.name) / f"normalized_scale_{'test'}.csv"
        with train.record_data_access([]):
            with self.assertRaises(train.FormalTrainingError):
                path.open("rb")

    def test_06_train_validation_loader_dispatch(self):
        profile = self._profile("source")
        with patch.object(train.solar_data, "load_training_validation_profile", return_value=profile) as loader:
            loaded = train.load_role_training_validation(self.plan.spec, "source", [])
        self.assertIs(loaded, profile)
        kwargs = loader.call_args.kwargs
        self.assertEqual(kwargs["expected_profile"], "source")
        self.assertEqual(kwargs["expected_profile_dir_name"], "source_profile")

    def test_07_data_count_contract(self):
        train.validate_profile_counts(self._profile("source"), "source")
        train.validate_profile_counts(self._profile("target"), "target")
        broken = self._profile("target")
        broken.training.split_data.row_count = 999
        with self.assertRaises(train.FormalTrainingError):
            train.validate_profile_counts(broken, "target")

    def test_08_model_builder_dispatch(self):
        source_plan = self.plan.candidates["source"][0]
        wotl_plan = self.plan.candidates["wotl"][0]
        partial_plan = self.plan.candidates["partial_ft"][0]
        locked = object()
        with patch.object(train, "build_linear_source_model", return_value="source") as source_builder, patch.object(
            train, "build_linear_without_tl_model", return_value="wotl"
        ) as wotl_builder, patch.object(
            train,
            "build_linear_partial_ft_candidate",
            return_value=SimpleNamespace(model="partial"),
        ) as partial_builder:
            self.assertEqual(train.build_candidate_model(source_plan), "source")
            self.assertEqual(train.build_candidate_model(wotl_plan), "wotl")
            self.assertEqual(train.build_candidate_model(partial_plan, locked_source_model=locked), "partial")
        source_builder.assert_called_once()
        wotl_builder.assert_called_once()
        self.assertIs(partial_builder.call_args.args[0], locked)

    def test_09_callback_factory_is_reused(self):
        factory = Mock(return_value=(object(), object(), object(), object()))
        model = FakeModel()
        train._fit_formal_candidate(model, self._profile(), Path("checkpoint_{epoch}.hdf5"), callback_factory=factory)
        factory.assert_called_once()
        self.assertEqual(len(model.fit_calls[0][2]["callbacks"]), 5)

    def test_10_formal_fit_parameters(self):
        model = FakeModel()
        history, learning_rates = train._fit_formal_candidate(
            model,
            self._profile(),
            Path("checkpoint_{epoch}.hdf5"),
            callback_factory=lambda path: (object(), object(), object(), object()),
        )
        kwargs = model.fit_calls[0][2]
        self.assertEqual(kwargs["epochs"], FORMAL_CALLBACK_POLICY.maximum_epochs)
        self.assertEqual(kwargs["batch_size"], 128)
        self.assertFalse(kwargs["shuffle"])
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertEqual(len(learning_rates), 2)

    def test_11_fake_fit_invoked_exactly_once(self):
        model = FakeModel()
        train._fit_formal_candidate(
            model,
            self._profile(),
            Path("checkpoint_{epoch}.hdf5"),
            callback_factory=lambda path: (object(), object(), object(), object()),
        )
        self.assertEqual(len(model.fit_calls), 1)

    def test_12_best_epoch_checkpoint_selection(self):
        candidate = Path(self.temporary.name)
        checkpoint = candidate / "checkpoint_epoch_0002.hdf5"
        checkpoint.write_bytes(b"formal-checkpoint")
        selected = train.select_best_checkpoint({"val_loss": [0.5, 0.2, 0.3]}, candidate)
        self.assertEqual(selected.best_epoch, 2)
        self.assertEqual(selected.path, checkpoint)

    def test_13_checkpoint_sha_logic(self):
        candidate = Path(self.temporary.name)
        content = b"sha-evidence"
        (candidate / "checkpoint_epoch_0001.hdf5").write_bytes(content)
        selected = train.select_best_checkpoint({"val_loss": [0.1]}, candidate)
        self.assertEqual(selected.sha256, hashlib.sha256(content).hexdigest())
        self.assertTrue(selected.sha256_verified)

    def test_14_validation_inverse_transform_and_metrics(self):
        validation = self._profile().validation
        prediction = validation.sequences.y_seq + 0.01
        model = PredictModel(prediction)
        metrics = train.evaluate_validation_candidate(model, validation, IdentityScale())
        self.assertEqual(metrics.prediction_count, 126)
        self.assertAlmostEqual(metrics.original_mae, 1.0)
        self.assertAlmostEqual(metrics.original_mse, 1.0)
        self.assertAlmostEqual(metrics.original_rmse, 1.0)
        self.assertEqual(model.calls[0][1]["batch_size"], 128)

    def test_15_validation_candidate_score_fields(self):
        source_plan = self.plan.candidates["wotl"][0]
        candidate_root = (
            LINEAR_EXPERIMENTS["B"].output_root
            / self.RUN_ID
            / "target_wotl_candidates"
            / source_plan.candidate_id
        )
        plan = train.FormalCandidatePlan(
            lifecycle=source_plan.lifecycle,
            candidate_id=source_plan.candidate_id,
            learning_rate=source_plan.learning_rate,
            candidate_root=candidate_root,
            checkpoint_pattern=candidate_root / "checkpoint_epoch_{epoch:04d}.hdf5",
        )
        checkpoint = train.BestCheckpoint(1, 0.1, candidate_root / "checkpoint_epoch_0001.hdf5", "b" * 64, True)
        metrics = train.ValidationMetrics(0.1, 0.1, 0.1, 0.8, 10, 120, 11, 0.7, 126)
        score = train.build_validation_score(
            formal_plan=self.plan,
            candidate_plan=plan,
            checkpoint=checkpoint,
            metrics=metrics,
            trainable_params=46681,
        )
        self.assertTrue(score.validation_only)
        self.assertFalse(score.test_accessed)
        self.assertFalse(score.test_metrics_used)

    def test_16_both_partial_candidates_accept_same_locked_source(self):
        locked = object()
        with patch.object(
            train,
            "build_linear_partial_ft_candidate",
            side_effect=lambda source, **kwargs: SimpleNamespace(model=(source, kwargs["learning_rate"])),
        ) as builder:
            built = [
                train.build_candidate_model(plan, locked_source_model=locked)
                for plan in self.plan.candidates["partial_ft"]
            ]
        self.assertTrue(all(item[0] is locked for item in built))
        self.assertTrue(all(call.args[0] is locked for call in builder.call_args_list))

    def test_17_partial_ft_state_audit(self):
        before = {index: (np.array([float(index)]),) for index in range(1, 7)}
        after = {index: tuple(value.copy() for value in values) for index, values in before.items()}
        for index in (1, 4, 6):
            after[index][0][0] += 1.0
        audit = train.audit_partial_ft_state(before, after)
        self.assertTrue(all(audit.__dict__.values()))
        after[3][0][0] += 1.0
        with self.assertRaises(train.FormalTrainingError):
            train.audit_partial_ft_state(before, after)

    def test_18_selection_json_round_trip(self):
        path = Path(self.temporary.name) / "selection.json"
        loaded = train.write_and_reload_selection(self._selection(), path)
        self.assertEqual(loaded["run_id"], self.RUN_ID)
        protocol.validate_selection_record(loaded)

    def test_19_manifest_initial_and_selection_transition(self):
        manifest = self._base_manifest()
        self.assertFalse(manifest["dry_run"])
        self.assertTrue(manifest["formal_eligible"])
        self.assertFalse(manifest["selection_locked"])
        self.assertFalse(manifest["git"]["dirty"])
        self.assertTrue(manifest["known_untracked_present"])
        locked = train.lock_manifest_after_selection(manifest, self._selection())
        self.assertTrue(locked["selection_locked"])
        self.assertTrue(locked["checkpoint_locked"])
        self.assertFalse(locked["test_authorized"])
        self.assertEqual(locked["protocol_stage"], protocol.ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value)

    def test_20_runner_exposes_no_final_authorization(self):
        source = inspect.getsource(train)
        self.assertNotIn("authorize_final_test", source)
        self.assertNotIn("human_authorized=True", source)

    def test_21_source_scan_has_no_forbidden_split_literals(self):
        source = inspect.getsource(train)
        self.assertNotIn(f"normalized_scale_{'test'}.csv", source)
        self.assertNotIn(f"original_scale_{'test'}.csv", source)

    def test_22_prepare_only_executes_zero_epochs(self):
        with patch.object(train, "prepare_formal_training_run", return_value=self.plan):
            result = train.run_formal_train_validation(
                "B",
                self.RUN_ID,
                expected_git_head=self.GIT_HEAD,
                execute_training=False,
            )
        self.assertIs(result, self.plan)
        self.assertEqual(result.training_epochs_executed, 0)

    def test_23_environment_gate(self):
        train.validate_formal_environment(self.environment)
        broken = train.FormalEnvironment(**{**self.environment.__dict__, "numpy": "wrong"})
        with self.assertRaises(train.FormalTrainingError):
            train.validate_formal_environment(broken)

    def test_24_git_provenance_gate(self):
        train.validate_formal_git_provenance(self.git, expected_git_head=self.GIT_HEAD)
        broken = train.FormalGitProvenance(
            head=self.GIT_HEAD,
            branch="test",
            tracked_dirty=False,
            known_untracked_paths=(),
            unknown_untracked_paths=("runner.py",),
        )
        with self.assertRaises(train.FormalTrainingError):
            train.validate_formal_git_provenance(broken, expected_git_head=self.GIT_HEAD)

    def test_25_no_real_formal_root_creation(self):
        before = self._filesystem_snapshot(LINEAR_FORMAL_BASE)
        self.assertFalse(self.paths.run_root.exists())
        self.assertEqual(self._filesystem_snapshot(LINEAR_FORMAL_BASE), before)

    def test_26_reduced_lr_checkpoint_reload_is_valid(self):
        checkpoint_path = Path(self.temporary.name) / "checkpoint_epoch_0001.hdf5"
        checkpoint_path.write_bytes(b"reduced-lr-checkpoint")
        checkpoint = train.BestCheckpoint(
            best_epoch=1,
            best_val_loss=0.1,
            path=checkpoint_path,
            sha256=hashlib.sha256(b"reduced-lr-checkpoint").hexdigest(),
            sha256_verified=True,
        )
        reloaded = SimpleNamespace(optimizer=SimpleNamespace(learning_rate=1e-7))
        with patch.object(train, "load_model", return_value=reloaded) as loader, patch.object(
            train, "validate_linear_model"
        ) as validator:
            self.assertIs(train.reload_best_checkpoint(checkpoint), reloaded)
        loader.assert_called_once()
        validator.assert_called_once_with(reloaded)

        changed = train.BestCheckpoint(
            best_epoch=1,
            best_val_loss=0.1,
            path=checkpoint_path,
            sha256="0" * 64,
            sha256_verified=True,
        )
        with self.assertRaises(train.FormalTrainingError):
            train.reload_best_checkpoint(changed)

    def test_27_known_untracked_does_not_mark_git_dirty(self):
        identity = train.protocol_git_identity(self.plan)
        self.assertFalse(identity.dirty)
        manifest = self._base_manifest()
        self.assertFalse(manifest["git"]["dirty"])
        self.assertTrue(manifest["known_untracked_present"])

        tracked_dirty = train.FormalGitProvenance(
            head=self.GIT_HEAD,
            branch="test",
            tracked_dirty=True,
            known_untracked_paths=(),
            unknown_untracked_paths=(),
        )
        with self.assertRaises(train.FormalTrainingError):
            train.validate_formal_git_provenance(
                tracked_dirty, expected_git_head=self.GIT_HEAD
            )

    def test_28_access_guard_records_reads_not_output_writes(self):
        output = Path(self.temporary.name) / "artifact.json"
        source = Path(self.temporary.name) / "source_input.csv"
        source.write_text("value\n1\n", encoding="utf-8")
        accessed = []
        with train.record_data_access(accessed):
            with output.open("x", encoding="utf-8") as stream:
                stream.write("{}")
            with source.open("r", encoding="utf-8") as stream:
                stream.read()
        self.assertTrue(output.is_file())
        self.assertNotIn(output.resolve(), accessed)
        self.assertIn(source.resolve(), accessed)

    def test_29_fake_execute_training_full_orchestration(self):
        events = []
        candidate_access = []
        partial_sources = []
        source_profile = self._profile("source")
        target_profile = self._profile("target")
        source_input = Path(self.temporary.name) / "source_profile" / "training.csv"
        target_input = Path(self.temporary.name) / "target_profile" / "training.csv"
        source_input.parent.mkdir()
        target_input.parent.mkdir()
        source_input.write_text("source", encoding="utf-8")
        target_input.write_text("target", encoding="utf-8")

        dry_manifest, _ = protocol.build_protocol_manifest(
            "B",
            self.paths,
            train.protocol_git_identity(self.plan),
        )
        scores = {
            plan.candidate_id: self._score(
                plan.lifecycle,
                plan.learning_rate,
                {
                    "SRC_lr1e-4": 0.20,
                    "SRC_lr3e-5": 0.30,
                    "WOTL_lr1e-4": 0.25,
                    "WOTL_lr3e-5": 0.35,
                    "PFT_lr1e-5": 0.22,
                    "PFT_lr3e-5": 0.32,
                }[plan.candidate_id],
                {
                    "SRC_lr1e-4": "1",
                    "SRC_lr3e-5": "2",
                    "WOTL_lr1e-4": "3",
                    "WOTL_lr3e-5": "4",
                    "PFT_lr1e-5": "5",
                    "PFT_lr3e-5": "6",
                }[plan.candidate_id],
            )
            for plans in self.plan.candidates.values()
            for plan in plans
        }
        epoch_counts = iter((2, 3, 4, 5, 6, 7))

        def load_role(spec, role, accessed):
            del spec
            events.append(f"load:{role}")
            accessed.append(source_input.resolve() if role == "source" else target_input.resolve())
            return source_profile if role == "source" else target_profile

        def build_model(candidate_plan, *, locked_source_model=None):
            if candidate_plan.lifecycle == "partial_ft":
                partial_sources.append(locked_source_model)
            return f"model:{candidate_plan.candidate_id}"

        def execute_candidate(**kwargs):
            candidate_plan = kwargs["candidate_plan"]
            events.append(f"execute:{candidate_plan.lifecycle}:{candidate_plan.candidate_id}")
            candidate_access.append(
                (candidate_plan.lifecycle, tuple(kwargs["candidate_accessed_files"]))
            )
            score = scores[candidate_plan.candidate_id]
            checkpoint = train.BestCheckpoint(
                score.best_epoch,
                score.validation_loss,
                score.checkpoint_path,
                score.checkpoint_sha256,
                True,
            )
            return train.CandidateRunResult(
                plan=candidate_plan,
                completed_epochs=next(epoch_counts),
                score=score,
                metrics=train.ValidationMetrics(
                    score.validation_loss,
                    0.1,
                    0.1,
                    0.8,
                    score.validation_original_mae,
                    score.validation_original_rmse**2,
                    score.validation_original_rmse,
                    score.validation_original_r2,
                    1,
                ),
                checkpoint=checkpoint,
                reloaded_model=(
                    "selected-source-model"
                    if candidate_plan.candidate_id == "SRC_lr1e-4"
                    else f"reloaded:{candidate_plan.candidate_id}"
                ),
                state_audit=None,
            )

        architecture = protocol.SourceArchitectureEvidence(
            activation="linear",
            git_head=self.GIT_HEAD,
            layer_classes=protocol.FORMAL_LINEAR_LAYER_CLASSES,
            input_shape=(5, 5),
            output_shape=(1,),
            total_params=46681,
            formal_eligible=True,
        )
        real_provenance_builder = protocol.build_source_checkpoint_provenance_from_selection

        def provenance_builder(**kwargs):
            events.append("source:provenance")
            return real_provenance_builder(**kwargs)

        with patch.dict(
            train.os.environ,
            {"SOLAR_RUN_FORMAL": "1", "PYTHONHASHSEED": "1234"},
        ), patch.object(
            train, "prepare_formal_training_run", return_value=self.plan
        ), patch.object(
            train, "set_reproducibility"
        ), patch.object(
            train, "build_protocol_manifest", return_value=(dry_manifest, ())
        ), patch.object(
            train, "load_role_training_validation", side_effect=load_role
        ), patch.object(
            train, "build_candidate_model", side_effect=build_model
        ), patch.object(
            train, "_execute_candidate", side_effect=execute_candidate
        ), patch.object(
            train, "_source_architecture", return_value=architecture
        ), patch.object(
            train,
            "build_source_checkpoint_provenance_from_selection",
            side_effect=provenance_builder,
        ):
            result = train.run_formal_train_validation(
                "B",
                self.RUN_ID,
                expected_git_head=self.GIT_HEAD,
                execute_training=True,
            )

        self.assertEqual(len(result.source_results), 2)
        self.assertEqual(len(result.wotl_results), 2)
        self.assertEqual(len(result.partial_ft_results), 2)
        self.assertLess(events.index("load:source"), events.index("execute:source:SRC_lr1e-4"))
        self.assertLess(events.index("execute:source:SRC_lr3e-5"), events.index("source:provenance"))
        self.assertLess(events.index("source:provenance"), events.index("load:target"))
        self.assertEqual(partial_sources, ["selected-source-model", "selected-source-model"])
        self.assertTrue(result.selection.comparison_pair_locked)
        self.assertFalse(result.selection.test_accessed)
        self.assertFalse(result.selection.test_authorized)
        self.assertEqual(result.total_training_epochs_executed, 27)
        locked_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            locked_manifest["protocol_stage"],
            protocol.ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value,
        )
        for lifecycle, paths in candidate_access:
            if lifecycle == "source":
                self.assertEqual(paths, (source_input.resolve(),))
            else:
                self.assertEqual(paths, (target_input.resolve(),))
        self.assertFalse(
            any(path.name in train.FORBIDDEN_SPLIT_FILENAMES for path in result.accessed_files)
        )
        self.assertTrue(result.plan.paths.run_root.is_relative_to(Path(self.temporary.name)))

    def test_30_filesystem_safety_with_absent_or_existing_simulated_root(self):
        base = Path(self.temporary.name)
        absent = base / "absent_formal"
        self.assertFalse(absent.exists())
        self.assertEqual(self._filesystem_snapshot(absent), None)

        existing = base / "existing_formal"
        existing.mkdir()
        evidence = existing / "existing_result.json"
        evidence.write_text('{"sealed": true}', encoding="utf-8")
        before = self._filesystem_snapshot(existing)
        train._candidate_plans(self.paths)
        self.assertEqual(self._filesystem_snapshot(existing), before)


if __name__ == "__main__":
    unittest.main()
