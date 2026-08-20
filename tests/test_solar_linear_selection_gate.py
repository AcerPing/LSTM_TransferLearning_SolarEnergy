from __future__ import annotations

import json
import unittest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from r2_config.solar_linear import LINEAR_EXPERIMENTS
from r2_config.solar_linear_formal import (
    FORMAL_LINEAR_LAYER_CLASSES,
    FORMAL_PROTOCOL_VERSION,
)
from r2_helpers import solar_linear_formal as formal


class SolarLinearSelectionGateTests(unittest.TestCase):
    RUN_ID = "20991231T235959Z_seed1234"
    GIT_HEAD = "1" * 40

    def _score(
        self,
        lifecycle,
        learning_rate,
        validation_loss,
        *,
        experiment_id="A2",
        run_id=RUN_ID,
        rmse=100.0,
        mae=80.0,
        r2=0.5,
        sha_char="a",
        best_epoch=40,
    ):
        identifiers = {
            ("source", 1e-4): "SRC_lr1e-4",
            ("source", 3e-5): "SRC_lr3e-5",
            ("wotl", 1e-4): "WOTL_lr1e-4",
            ("wotl", 3e-5): "WOTL_lr3e-5",
            ("partial_ft", 1e-5): "PFT_lr1e-5",
            ("partial_ft", 3e-5): "PFT_lr3e-5",
        }
        directories = {
            "source": "source_candidates",
            "wotl": "target_wotl_candidates",
            "partial_ft": "target_partial_ft_candidates",
        }
        identifier = identifiers[(lifecycle, learning_rate)]
        return formal.ValidationCandidateScore(
            protocol_version=FORMAL_PROTOCOL_VERSION,
            experiment_id=experiment_id,
            run_id=run_id,
            git_head=self.GIT_HEAD,
            target_protocol_fingerprint=formal.target_protocol_fingerprint(experiment_id),
            lifecycle=lifecycle,
            candidate_id=identifier,
            learning_rate=learning_rate,
            best_epoch=best_epoch,
            validation_loss=validation_loss,
            validation_original_mae=mae,
            validation_original_rmse=rmse,
            validation_original_r2=r2,
            trainable_params=29161 if lifecycle == "partial_ft" else 46681,
            checkpoint_path=(
                LINEAR_EXPERIMENTS[experiment_id].output_root
                / run_id
                / directories[lifecycle]
                / identifier
                / f"checkpoint_epoch_{best_epoch:04d}.hdf5"
            ),
            checkpoint_sha256=sha_char * 64,
            checkpoint_sha256_verified=True,
        )

    def _source_scores(self, experiment_id="A2"):
        return (
            self._score("source", 1e-4, 0.20, experiment_id=experiment_id, sha_char="a"),
            self._score("source", 3e-5, 0.25, experiment_id=experiment_id, sha_char="b"),
        )

    def _wotl_scores(self, experiment_id="A2"):
        return (
            self._score("wotl", 1e-4, 0.21, experiment_id=experiment_id, sha_char="c"),
            self._score("wotl", 3e-5, 0.24, experiment_id=experiment_id, sha_char="d"),
        )

    def _partial_scores(self, experiment_id="A2"):
        return (
            self._score("partial_ft", 1e-5, 0.22, experiment_id=experiment_id, sha_char="e"),
            self._score("partial_ft", 3e-5, 0.23, experiment_id=experiment_id, sha_char="f"),
        )

    def _architecture(self):
        return formal.SourceArchitectureEvidence(
            activation="linear",
            git_head=self.GIT_HEAD,
            layer_classes=FORMAL_LINEAR_LAYER_CLASSES,
            input_shape=(5, 5),
            output_shape=(1,),
            total_params=46681,
            formal_eligible=True,
        )

    def _selection(self, experiment_id="A2", **overrides):
        values = {
            "source_scores": self._source_scores(experiment_id),
            "wotl_scores": self._wotl_scores(experiment_id),
            "partial_ft_scores": self._partial_scores(experiment_id),
        }
        values.update(overrides)
        return formal.build_selection_record(
            experiment_id=experiment_id,
            selection_timestamp="2099-12-31T23:59:59Z",
            source_scores=values["source_scores"],
            source_architecture_evidence=self._architecture(),
            expected_git_head=self.GIT_HEAD,
            wotl_scores=values["wotl_scores"],
            partial_ft_scores=values["partial_ft_scores"],
        )

    def test_01_complete_source_registry_builds_selected_provenance(self):
        provenance = formal.build_source_checkpoint_provenance_from_selection(
            experiment_id="A2",
            run_id=self.RUN_ID,
            source_scores=self._source_scores(),
            expected_git_head=self.GIT_HEAD,
            architecture_evidence=self._architecture(),
        )
        self.assertEqual(provenance.source_candidate_id, "SRC_lr1e-4")
        self.assertEqual(provenance.source_checkpoint_sha256, "a" * 64)
        self.assertTrue(provenance.validation_selected)

    def test_02_missing_source_lr1e4_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(source_scores=(self._source_scores()[1],))

    def test_03_missing_source_lr3e5_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(source_scores=(self._source_scores()[0],))

    def test_04_duplicate_source_candidate_fails(self):
        score = self._source_scores()[0]
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(source_scores=(score, score))

    def test_05_source_test_taint_fails(self):
        tainted = replace(self._source_scores()[0], test_accessed=True)
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(source_scores=(tainted, self._source_scores()[1]))

    def test_06_source_smoke_checkpoint_fails(self):
        smoke = replace(
            self._source_scores()[0],
            checkpoint_path=Path("reports/Solar Energy Result/_smoke_linear/source/checkpoint_epoch_0040.hdf5"),
        )
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(source_scores=(smoke, self._source_scores()[1]))

    def test_07_source_wrong_run_fails(self):
        wrong = replace(self._source_scores()[1], run_id="20991230T235959Z_seed1234")
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(source_scores=(self._source_scores()[0], wrong))

    def test_08_best_wotl_and_partial_are_independently_locked(self):
        selection = self._selection()
        self.assertEqual(selection.wotl_selected_candidate, "WOTL_lr1e-4")
        self.assertEqual(selection.partial_ft_selected_candidate, "PFT_lr1e-5")
        self.assertTrue(selection.comparison_pair_locked)

    def test_09_no_cross_method_winner_fields(self):
        selection = self._selection()
        self.assertFalse(hasattr(selection, "selected_method"))
        self.assertFalse(hasattr(selection, "selected_checkpoint_sha256"))
        mapping = formal.selection_record_mapping(selection)
        self.assertNotIn("selected_method", mapping)
        self.assertNotIn("selected_checkpoint_sha256", mapping)

    def test_10_pair_checkpoint_shas_are_both_locked(self):
        selection = self._selection()
        self.assertEqual(selection.wotl_selected_checkpoint_sha256, "c" * 64)
        self.assertEqual(selection.partial_ft_selected_checkpoint_sha256, "e" * 64)
        self.assertTrue(selection.wotl_selected_checkpoint_sha256_verified)
        self.assertTrue(selection.partial_ft_selected_checkpoint_sha256_verified)

    def test_11_wotl_or_partial_candidate_missing_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(wotl_scores=(self._wotl_scores()[0],))
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(partial_ft_scores=(self._partial_scores()[0],))

    def test_12_wotl_partial_different_run_fails(self):
        different_run = replace(
            self._partial_scores()[1], run_id="20991230T235959Z_seed1234"
        )
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(partial_ft_scores=(self._partial_scores()[0], different_run))

    def test_13_lifecycle_tie_uses_lower_lr(self):
        scores = (
            self._score("wotl", 1e-4, 0.2, rmse=100),
            self._score("wotl", 3e-5, 0.2, rmse=100),
        )
        self.assertEqual(
            formal.select_validation_candidate("wotl", scores).candidate_id,
            "WOTL_lr3e-5",
        )

    def test_14_selection_schema_contains_three_comparisons(self):
        selection = self._selection()
        formal.validate_selection_record(selection)
        mapping = formal.selection_record_mapping(selection)
        self.assertTrue(set(formal.REQUIRED_SELECTION_FIELDS).issubset(mapping))
        self.assertEqual(len(mapping["source_validation_comparison"]), 2)
        self.assertEqual(len(mapping["wotl_validation_comparison"]), 2)
        self.assertEqual(len(mapping["partial_ft_validation_comparison"]), 2)
        self.assertFalse(mapping["post_test_tuning_allowed"])

    def test_15_unverified_wotl_or_partial_sha_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(self._selection(), wotl_selected_checkpoint_sha256_verified=False)
            )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(self._selection(), partial_ft_selected_checkpoint_sha256_verified=False)
            )

    def _authorization_manifest(self, experiment_id="A2"):
        return {
            "protocol_version": FORMAL_PROTOCOL_VERSION,
            "target_protocol_fingerprint": formal.target_protocol_fingerprint(experiment_id),
            "git": {"head": self.GIT_HEAD, "branch": "test", "dirty": False},
            "run_id": self.RUN_ID,
            "dry_run": False,
            "protocol_stage": (
                formal.ProtocolStage.SELECTION_LOCKED_AWAITING_TEST_AUTHORIZATION.value
            ),
            "formal_eligible": True,
            "config_locked": True,
            "selection_locked": True,
            "checkpoint_locked": True,
            "test_authorized": False,
            "post_test_tuning_allowed": False,
            "test_accessed": False,
            "test_metrics_used_for_selection": False,
            "experiment_id": experiment_id,
            "historical_target_test_previously_revealed": experiment_id == "A2",
        }

    def test_16_authorization_defaults_denied(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                self._authorization_manifest(), self._selection(), human_authorized=False
            )

    def test_17_valid_authorization_contains_comparison_pair(self):
        authorization = formal.authorize_final_test(
            self._authorization_manifest(), self._selection(), human_authorized=True
        )
        formal.validate_authorization_proof(authorization)
        self.assertEqual(authorization.wotl_selected_candidate, "WOTL_lr1e-4")
        self.assertEqual(authorization.partial_ft_selected_candidate, "PFT_lr1e-5")
        self.assertEqual(authorization.wotl_checkpoint_sha256, "c" * 64)
        self.assertEqual(authorization.partial_ft_checkpoint_sha256, "e" * 64)
        self.assertTrue(authorization.comparison_pair_locked)
        self.assertTrue(authorization.test_authorized)
        self.assertEqual(authorization.protocol_stage, formal.ProtocolStage.TEST_AUTHORIZED)

    def test_18_invalid_pair_sha_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                self._authorization_manifest(),
                replace(self._selection(), wotl_selected_checkpoint_sha256="bad"),
                human_authorized=True,
            )
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                self._authorization_manifest(),
                replace(self._selection(), partial_ft_selected_checkpoint_sha256="bad"),
                human_authorized=True,
            )

    def test_19_unverified_pair_sha_fails_authorization(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                self._authorization_manifest(),
                replace(self._selection(), wotl_selected_checkpoint_sha256_verified=False),
                human_authorized=True,
            )
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                self._authorization_manifest(),
                replace(self._selection(), partial_ft_selected_checkpoint_sha256_verified=False),
                human_authorized=True,
            )

    def test_20_unlocked_pair_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                self._authorization_manifest(),
                replace(self._selection(), comparison_pair_locked=False),
                human_authorized=True,
            )

    def test_21_a2_b_disclosure_is_preserved(self):
        a2 = formal.authorize_final_test(
            self._authorization_manifest("A2"), self._selection("A2"), human_authorized=True
        )
        b = formal.authorize_final_test(
            self._authorization_manifest("B"), self._selection("B"), human_authorized=True
        )
        self.assertEqual((a2.experiment_id, b.experiment_id), ("A2", "B"))
        bad_b = self._authorization_manifest("B")
        bad_b["historical_target_test_previously_revealed"] = True
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(bad_b, self._selection("B"), human_authorized=True)

    def test_22_protocol_git_and_target_identity_are_locked(self):
        selection = self._selection()
        self.assertEqual(selection.protocol_version, FORMAL_PROTOCOL_VERSION)
        self.assertEqual(selection.git_head, self.GIT_HEAD)
        self.assertEqual(
            selection.target_protocol_fingerprint,
            formal.target_protocol_fingerprint("A2"),
        )
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(
                wotl_scores=(
                    replace(self._wotl_scores()[0], git_head="2" * 40),
                    self._wotl_scores()[1],
                )
            )
        with self.assertRaises(formal.FormalProtocolError):
            self._selection(
                partial_ft_scores=(
                    replace(
                        self._partial_scores()[0],
                        target_protocol_fingerprint="0" * 64,
                    ),
                    self._partial_scores()[1],
                )
            )

    def test_23_authorization_rejects_protocol_identity_mismatch(self):
        bad_protocol = self._authorization_manifest()
        bad_protocol["protocol_version"] = "wrong"
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                bad_protocol, self._selection(), human_authorized=True
            )
        bad_git = self._authorization_manifest()
        bad_git["git"] = {"head": "2" * 40, "branch": "test", "dirty": False}
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(bad_git, self._selection(), human_authorized=True)

    def test_24_normal_selection_revalidation_passes(self):
        selection = self._selection()
        formal.validate_selection_record(selection)
        formal.validate_selection_record(formal.selection_record_mapping(selection))

    def test_25_tampered_source_selected_candidate_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(self._selection(), source_selected_candidate="SRC_lr3e-5")
            )

    def test_26_tampered_wotl_selected_candidate_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(self._selection(), wotl_selected_candidate="WOTL_lr3e-5")
            )

    def test_27_tampered_partial_selected_candidate_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(
                    self._selection(),
                    partial_ft_selected_candidate="PFT_lr3e-5",
                )
            )

    def test_28_tampered_source_selected_sha_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(self._selection(), source_checkpoint_sha256="b" * 64)
            )

    def test_29_tampered_wotl_selected_sha_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(
                    self._selection(),
                    wotl_selected_checkpoint_sha256="d" * 64,
                )
            )

    def test_30_tampered_partial_selected_sha_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(
                    self._selection(),
                    partial_ft_selected_checkpoint_sha256="f" * 64,
                )
            )

    def test_31_tampered_source_comparison_winner_fails(self):
        selection = self._selection()
        comparison = (
            replace(selection.source_validation_comparison[0], validation_loss=0.50),
            selection.source_validation_comparison[1],
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(selection, source_validation_comparison=comparison)
            )

    def test_32_tampered_wotl_comparison_winner_fails(self):
        selection = self._selection()
        comparison = (
            replace(selection.wotl_validation_comparison[0], validation_loss=0.50),
            selection.wotl_validation_comparison[1],
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(selection, wotl_validation_comparison=comparison)
            )

    def test_33_tampered_partial_comparison_winner_fails(self):
        selection = self._selection()
        comparison = (
            replace(
                selection.partial_ft_validation_comparison[0],
                validation_loss=0.50,
            ),
            selection.partial_ft_validation_comparison[1],
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(selection, partial_ft_validation_comparison=comparison)
            )

    def test_34_json_round_trip_selection_mapping_passes(self):
        def thaw(value):
            if isinstance(value, Mapping):
                return {key: thaw(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [thaw(item) for item in value]
            if isinstance(value, Path):
                return str(value)
            return value

        payload = json.loads(
            json.dumps(thaw(formal.selection_record_mapping(self._selection())))
        )
        self.assertEqual(payload["run_id"], self.RUN_ID)
        formal.validate_selection_record(payload)

    def test_35_comparison_mapping_requires_exact_fields(self):
        payload = dict(formal.selection_record_mapping(self._selection()))
        missing = [dict(item) for item in payload["wotl_validation_comparison"]]
        missing[0].pop("checkpoint_sha256")
        payload["wotl_validation_comparison"] = missing
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(payload)

        payload = dict(formal.selection_record_mapping(self._selection()))
        unknown = [dict(item) for item in payload["wotl_validation_comparison"]]
        unknown[0]["unregistered_evidence"] = True
        payload["wotl_validation_comparison"] = unknown
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(payload)

    def test_36_dry_run_manifest_cannot_authorize(self):
        manifest = self._authorization_manifest()
        manifest["dry_run"] = True
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                manifest, self._selection(), human_authorized=True
            )

    def test_37_missing_dry_run_field_cannot_authorize(self):
        manifest = self._authorization_manifest()
        manifest.pop("dry_run")
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                manifest, self._selection(), human_authorized=True
            )

    def test_38_missing_protocol_stage_cannot_authorize(self):
        manifest = self._authorization_manifest()
        manifest.pop("protocol_stage")
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                manifest, self._selection(), human_authorized=True
            )

    def test_39_wrong_protocol_stages_cannot_authorize(self):
        invalid_stages = (
            formal.ProtocolStage.CONFIG_LOCKED_AWAITING_TRAINING,
            formal.ProtocolStage.TEST_AUTHORIZED.value,
            formal.ProtocolStage.TEST_COMPLETED.value,
        )
        for invalid_stage in invalid_stages:
            with self.subTest(stage=invalid_stage):
                manifest = self._authorization_manifest()
                manifest["protocol_stage"] = invalid_stage
                with self.assertRaises(formal.FormalProtocolError):
                    formal.authorize_final_test(
                        manifest, self._selection(), human_authorized=True
                    )

    def test_40_formal_selection_stage_authorizes_pair(self):
        manifest = self._authorization_manifest()
        authorization = formal.authorize_final_test(
            manifest, self._selection(), human_authorized=True
        )
        self.assertTrue(authorization.test_authorized)
        self.assertTrue(authorization.comparison_pair_locked)

    def test_41_selection_locks_source_run_id(self):
        selection = self._selection()
        self.assertEqual(selection.run_id, self.RUN_ID)
        formal.validate_selection_record(selection)

    def test_42_tampered_selection_valid_run_id_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(
                    self._selection(),
                    run_id="20991230T235959Z_seed1234",
                )
            )

    def test_43_tampered_selection_bad_run_id_fails(self):
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(self._selection(), run_id="bad-run")
            )

    def test_44_source_comparison_other_run_fails(self):
        selection = self._selection()
        comparison = (
            selection.source_validation_comparison[0],
            replace(
                selection.source_validation_comparison[1],
                run_id="20991230T235959Z_seed1234",
            ),
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(selection, source_validation_comparison=comparison)
            )

    def test_45_wotl_comparison_other_run_fails(self):
        selection = self._selection()
        comparison = (
            selection.wotl_validation_comparison[0],
            replace(
                selection.wotl_validation_comparison[1],
                run_id="20991230T235959Z_seed1234",
            ),
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(selection, wotl_validation_comparison=comparison)
            )

    def test_46_partial_comparison_other_run_fails(self):
        selection = self._selection()
        comparison = (
            selection.partial_ft_validation_comparison[0],
            replace(
                selection.partial_ft_validation_comparison[1],
                run_id="20991230T235959Z_seed1234",
            ),
        )
        with self.assertRaises(formal.FormalProtocolError):
            formal.validate_selection_record(
                replace(selection, partial_ft_validation_comparison=comparison)
            )

    def test_47_manifest_run_mismatch_fails_authorization(self):
        manifest = self._authorization_manifest()
        manifest["run_id"] = "20991230T235959Z_seed1234"
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                manifest, self._selection(), human_authorized=True
            )

    def test_48_missing_manifest_run_fails_authorization(self):
        manifest = self._authorization_manifest()
        manifest.pop("run_id")
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                manifest, self._selection(), human_authorized=True
            )

    def test_49_authorization_proof_locks_run_id(self):
        authorization = formal.authorize_final_test(
            self._authorization_manifest(),
            self._selection(),
            human_authorized=True,
        )
        self.assertEqual(authorization.run_id, self.RUN_ID)
        formal.validate_authorization_proof(authorization)

    def test_50_dry_run_manifest_uses_path_contract_run_id(self):
        dry_run = formal.prepare_formal_dry_run(
            "A2",
            self.RUN_ID,
            git_identity=formal.GitIdentity(
                head=self.GIT_HEAD,
                branch="test",
                dirty=False,
            ),
        )
        self.assertEqual(dry_run.manifest["run_id"], dry_run.paths.run_id)
        self.assertTrue(dry_run.manifest["dry_run"])
        self.assertFalse(dry_run.manifest["formal_eligible"])
        with self.assertRaises(formal.FormalProtocolError):
            formal.authorize_final_test(
                dry_run.manifest,
                self._selection(),
                human_authorized=True,
            )


if __name__ == "__main__":
    unittest.main()
