"""Corrected R1 transfer-learning integration gates.

This module loads the sealed Source checkpoint with ``compile=False``, validates
the mapped Target contract, and builds in-memory Freeze/Unfreeze candidates for
the approved Experiment A/B direction.  Phase C1 adds one explicitly bounded in-memory
Freeze smoke epoch.  Neither path predicts, evaluates, executes callbacks,
writes checkpoints, or creates a formal-run directory.

Phase C2 adds the formal Freeze runner contract and its zero-epoch dry-run.
Only the formal execution function contains training/evaluation calls; C2 does
not expose that function through the CLI.

Phase C4 adds one explicitly bounded in-memory Unfreeze smoke epoch.  It does
not create callbacks, predictions, checkpoints, or a formal Unfreeze run.

Phase C5 adds the formal Unfreeze runner contract and its isolated zero-epoch
dry-run.  Formal Unfreeze execution is deliberately not exposed through CLI.

Phase D4 reuses the same in-memory contract and exact weight-transfer audit for
Experiment B.  It does not authorize Experiment-B smoke or formal execution.

Phase D6 generalizes the formal Freeze runner contract and zero-epoch dry-run to
Experiment B.  The Experiment-B formal execution CLI remains deliberately
blocked until a later phase grants explicit authorization.

Phase D8 generalizes the one-epoch in-memory Unfreeze smoke to Experiment B;
formal Experiment-B Unfreeze dry-run and execution remain blocked.

Phase D9 generalizes the formal Unfreeze runner contract and zero-epoch dry-run
to Experiment B.  The Experiment-B formal execution CLI remains deliberately
blocked until a later phase grants explicit authorization.
"""

import csv
import hashlib
import inspect
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from r2_config.corrected_r1 import (
    CORRECTED_R1_EXPERIMENTS,
    CORRECTED_R1_EXPECTED_ROWS,
    CORRECTED_R1_EXPECTED_SEQUENCES,
    CORRECTED_R1_FEATURE_ORDER,
    CORRECTED_R1_HORIZON,
    CORRECTED_R1_ROOT,
    CORRECTED_R1_TARGET,
    CORRECTED_R1_TIMESTAMP,
    CORRECTED_R1_WINDOW,
    R2_EXPERIMENT_A_TL_FREEZE_FORMAL_OUTPUTS,
    R2_EXPERIMENT_A_TL_FREEZE_FORMAL_RUN_ID,
    R2_EXPERIMENT_A_TL_FREEZE_OUTPUT_UNITS,
    R2_EXPERIMENT_A_TL_CONTRACT,
    R2_EXPERIMENT_A_TL_UNFREEZE_FORMAL_OUTPUTS,
    R2_EXPERIMENT_A_TL_UNFREEZE_FORMAL_RUN_ID,
    R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR,
    R2_EXPERIMENT_B_WITHOUT_TL_ANCHOR,
    R2_SOURCE_FORMAL_EVALUATION_OUTPUTS,
    R2_SOURCE_FORMAL_OUTPUT_ROOT,
    R2_SOURCE_FORMAL_OUTPUTS,
    corrected_r1_profile_path,
    resolve_r2_transfer_learning_contract,
)
from utils.data_io import read_corrected_r1_profile, validate_corrected_r1_profile


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FORMAL_TL_CORE_EXECUTION_FILES = (
    "main.py",
    "utils/data_io.py",
    "utils/model.py",
    "r2_config/corrected_r1.py",
    "r2_helpers/r2_transfer_learning.py",
)
_FORMAL_TARGET_EVIDENCE_FILES = (
    "split_manifest.json",
    "scaler_manifest.json",
    "feature_scaler.joblib",
    "target_scaler.joblib",
)
_FORMAL_TL_FREEZE_OUTPUTS = R2_EXPERIMENT_A_TL_FREEZE_FORMAL_OUTPUTS
_FORMAL_TL_UNFREEZE_OUTPUTS = R2_EXPERIMENT_A_TL_UNFREEZE_FORMAL_OUTPUTS


def _transfer_learning_direction(experiment):
    try:
        return {
            "A": "Plant1 -> Plant2",
            "B": "Plant2 -> Plant1",
        }[experiment]
    except KeyError as exc:
        raise ValueError(
            f"Transfer-learning experiment must be A or B, got {experiment!r}"
        ) from exc


def _load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must contain an object: {path}")
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_state(root):
    root = Path(root)
    return {
        str(path.relative_to(root)).replace("\\", "/"): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def load_target_training_contract(*, corrected_r1_root=None, experiment="A"):
    """Return Target Training/Validation arrays without exposing Target Test."""

    contract = resolve_r2_transfer_learning_contract(experiment)
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    target_mapping = CORRECTED_R1_EXPERIMENTS[experiment]["target"]
    target_directory = corrected_r1_profile_path(root, experiment, "target")
    profile = read_corrected_r1_profile(target_directory)
    report = validate_corrected_r1_profile(
        profile,
        expected_plant=target_mapping["manifest_plant"],
        expected_profile=target_mapping["manifest_profile"],
        expected_rows=CORRECTED_R1_EXPECTED_ROWS["target"],
        expected_sequences=CORRECTED_R1_EXPECTED_SEQUENCES["target"],
        feature_order=CORRECTED_R1_FEATURE_ORDER,
        target_column=CORRECTED_R1_TARGET,
        window=CORRECTED_R1_WINDOW,
        horizon=CORRECTED_R1_HORIZON,
    )
    training = report["sequence_data"]["training"]
    validation = report["sequence_data"]["validation"]
    test = report["sequence_data"]["test"]
    return {
        "usage": "formal-target-training-validation-only",
        "profile_directory": report["profile_dir"].resolve(),
        "X_train": training["X"].copy(),
        "y_train": training["y"].copy(),
        "X_validation": validation["X"].copy(),
        "y_validation": validation["y"].copy(),
        "rows": dict(report["rows"]),
        "sequences": dict(report["sequences"]),
        "feature_order": tuple(report["feature_order"]),
        "target": report["target"],
        "alignment_ok": report["alignment_ok"],
        "alignment": "X[0:5] -> y[5]",
        "validation_duplicate_count": report["validation_duplicate_count"],
        "validation_padding_count": report["validation_padding_count"],
        "test_first_target_timestamp": test["target_timestamps"][0],
        "test_last_target_timestamp": test["target_timestamps"][-1],
        "test_sequence_count": len(test["y"]),
        "target_test_used_for_training": False,
        "target_test_used_for_validation": False,
        "target_test_used_for_callbacks": False,
        "target_test_used_for_selection": False,
        "target_test_metrics_computed": False,
    }


def load_target_evaluation_contract(*, corrected_r1_root=None, experiment="A"):
    """Load Target Test assets for post-selection evaluation only.

    This contract is intentionally separate from ``load_target_training_contract``.
    It validates normalized/original alignment and the Target target-scaler
    round-trip without predicting or calculating metrics.
    """

    resolve_r2_transfer_learning_contract(experiment)
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    target_mapping = CORRECTED_R1_EXPERIMENTS[experiment]["target"]
    target_directory = corrected_r1_profile_path(root, experiment, "target")
    normalized_profile = read_corrected_r1_profile(target_directory)
    normalized_report = validate_corrected_r1_profile(
        normalized_profile,
        expected_plant=target_mapping["manifest_plant"],
        expected_profile=target_mapping["manifest_profile"],
        expected_rows=CORRECTED_R1_EXPECTED_ROWS["target"],
        expected_sequences=CORRECTED_R1_EXPECTED_SEQUENCES["target"],
        feature_order=CORRECTED_R1_FEATURE_ORDER,
        target_column=CORRECTED_R1_TARGET,
        window=CORRECTED_R1_WINDOW,
        horizon=CORRECTED_R1_HORIZON,
    )
    original_profile = read_corrected_r1_profile(
        target_directory,
        splits=("test",),
        scale="original",
    )
    normalized_test = normalized_profile["splits"]["test"]
    original_test = original_profile["splits"]["test"]
    expected_columns = (
        CORRECTED_R1_TIMESTAMP,
        *CORRECTED_R1_FEATURE_ORDER,
        CORRECTED_R1_TARGET,
    )
    if tuple(original_test.columns) != expected_columns:
        raise ValueError("Corrected R1 original-scale Target Test columns mismatch")
    if not normalized_test[CORRECTED_R1_TIMESTAMP].equals(
        original_test[CORRECTED_R1_TIMESTAMP]
    ):
        raise ValueError("Target Test timestamps differ across scales")
    if original_test.isna().any().any():
        raise ValueError("Corrected R1 original-scale Target Test contains NaN")

    test_sequences = normalized_report["sequence_data"]["test"]
    target_indices = test_sequences["target_indices"]
    y_test_normalized = test_sequences["y"].copy()
    y_test_original = original_test.loc[:, CORRECTED_R1_TARGET].to_numpy(
        dtype=np.float64,
        copy=True,
    )[target_indices]
    scaler_file = normalized_profile["scaler_manifest"].get("target_scaler_file")
    if scaler_file != "target_scaler.joblib":
        raise ValueError(f"Unexpected Target scaler filename: {scaler_file!r}")
    target_scaler_path = (target_directory / scaler_file).resolve()
    if (
        target_scaler_path.parent != target_directory.resolve()
        or not target_scaler_path.is_file()
    ):
        raise FileNotFoundError(f"Target scaler not found: {target_scaler_path}")
    target_scaler = joblib.load(target_scaler_path)
    if int(target_scaler.n_features_in_) != 1:
        raise ValueError("Target scaler must have one input feature")
    fit_rows = int(np.asarray(target_scaler.n_samples_seen_).reshape(-1)[0])
    if fit_rows != CORRECTED_R1_EXPECTED_ROWS["target"]["training"]:
        raise ValueError("Target scaler was not fit on Target Training only")
    round_trip = target_scaler.inverse_transform(
        y_test_normalized.reshape(-1, 1)
    ).reshape(-1)
    round_trip_ok = bool(
        np.allclose(round_trip, y_test_original, rtol=1e-10, atol=1e-6)
    )
    if not round_trip_ok:
        raise ValueError("Target scaler/original-scale round-trip mismatch")
    if len(y_test_original) != CORRECTED_R1_EXPECTED_SEQUENCES["target"]["test"]:
        raise ValueError("Target evaluation sequence count mismatch")

    return {
        "usage": "post-best-validation-checkpoint-evaluation-only",
        "profile_directory": target_directory.resolve(),
        "target_scaler_path": target_scaler_path,
        "target_scaler": target_scaler,
        "target_scaler_fit_rows": fit_rows,
        "X_test": test_sequences["X"].copy(),
        "target_indices": target_indices.copy(),
        "target_timestamps": test_sequences["target_timestamps"].copy(),
        "y_test_normalized": y_test_normalized,
        "y_test_original": y_test_original,
        "y_test_round_trip": round_trip,
        "sequence_count": len(y_test_original),
        "round_trip_ok": round_trip_ok,
        "metrics_computed": False,
        "prediction_executed": False,
    }


def _source_checkpoint_gate(contract=None):
    contract = (
        R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    )
    run_directory = (
        _REPOSITORY_ROOT
        / R2_SOURCE_FORMAL_OUTPUT_ROOT
        / contract["source_run_id"]
    ).resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Sealed Source run directory missing: {run_directory}")
    manifest_path = run_directory / "run_manifest.json"
    manifest = _load_json(manifest_path)
    expected_artifacts = (
        *R2_SOURCE_FORMAL_OUTPUTS,
        *R2_SOURCE_FORMAL_EVALUATION_OUTPUTS,
    )
    missing = [
        relative_name
        for relative_name in expected_artifacts
        if not (run_directory / relative_name).is_file()
        or (run_directory / relative_name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f"Sealed Source artifacts missing/empty: {missing}")
    if manifest.get("run_id") != contract["source_run_id"]:
        raise ValueError("Sealed Source run ID mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("Sealed Source run is not complete")
    if manifest.get("experiment") != contract["experiment"]:
        raise ValueError("Sealed Source experiment mismatch")
    if manifest.get("source") != contract["source"]:
        raise ValueError("Sealed Source profile mismatch")
    checkpoint_path = run_directory / contract["source_checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Sealed Source checkpoint missing: {checkpoint_path}")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if checkpoint_sha256 != contract["source_checkpoint_sha256"]:
        raise ValueError("Sealed Source checkpoint SHA256 mismatch")

    with (run_directory / "epoch_log.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        epoch_rows = list(csv.DictReader(handle))
    if not epoch_rows:
        raise ValueError("Sealed Source epoch log is empty")
    source_final_lr = float(epoch_rows[-1]["lr"])
    if not np.isclose(source_final_lr, 1e-7, rtol=0.0, atol=1e-15):
        raise ValueError(f"Unexpected sealed Source final LR: {source_final_lr}")
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "run_directory": run_directory,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256_before": checkpoint_sha256,
        "artifact_count": len(expected_artifacts),
        "artifact_state_before": _file_state(run_directory),
        "final_learning_rate": source_final_lr,
        "final_seal_audit": True,
    }


def _scaler_isolation(corrected_r1_root, target_contract, *, experiment="A"):
    contract = resolve_r2_transfer_learning_contract(experiment)
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    source_directory = corrected_r1_profile_path(root, experiment, "source").resolve()
    target_directory = target_contract["profile_directory"]
    target_mapping = CORRECTED_R1_EXPERIMENTS[experiment]["target"]
    source_manifest = _load_json(source_directory / "scaler_manifest.json")
    target_manifest = _load_json(target_directory / "scaler_manifest.json")
    source_feature_path = source_directory / source_manifest["feature_scaler_file"]
    source_target_path = source_directory / source_manifest["target_scaler_file"]
    target_feature_path = target_directory / target_manifest["feature_scaler_file"]
    target_target_path = target_directory / target_manifest["target_scaler_file"]

    source_feature_scaler = joblib.load(source_feature_path)
    source_target_scaler = joblib.load(source_target_path)
    target_feature_scaler = joblib.load(target_feature_path)
    target_target_scaler = joblib.load(target_target_path)
    target_feature_seen = np.asarray(
        target_feature_scaler.n_samples_seen_
    ).reshape(-1)
    target_feature_rows = int(target_feature_seen[0])
    target_target_rows = int(
        np.asarray(target_target_scaler.n_samples_seen_).reshape(-1)[0]
    )
    identity_ok = (
        target_manifest.get("plant") == target_mapping["manifest_plant"]
        and target_manifest.get("profile") == "target"
        and target_manifest.get("feature_scaler_fit_split") == "training"
        and target_manifest.get("target_scaler_fit_split") == "training"
        and np.all(target_feature_seen == CORRECTED_R1_EXPECTED_ROWS["target"]["training"])
        and target_target_rows == CORRECTED_R1_EXPECTED_ROWS["target"]["training"]
        and target_manifest.get("clipping_applied") is False
        and getattr(target_feature_scaler, "clip", False) is False
        and getattr(target_target_scaler, "clip", False) is False
        and source_feature_path.resolve() != target_feature_path.resolve()
        and source_target_path.resolve() != target_target_path.resolve()
        and source_feature_scaler is not target_feature_scaler
        and source_target_scaler is not target_target_scaler
    )
    if not identity_ok:
        raise ValueError("Target scaler identity/isolation audit failed")

    normalized_test = read_corrected_r1_profile(
        target_directory,
        splits=("test",),
        scale="normalized",
    )["splits"]["test"]
    original_test = read_corrected_r1_profile(
        target_directory,
        splits=("test",),
        scale="original",
    )["splits"]["test"]
    if not normalized_test[CORRECTED_R1_TIMESTAMP].equals(
        original_test[CORRECTED_R1_TIMESTAMP]
    ):
        raise ValueError("Target Test timestamps differ across scales")
    first_target_index = CORRECTED_R1_WINDOW + CORRECTED_R1_HORIZON - 1
    target_indices = np.arange(first_target_index, len(normalized_test), dtype=np.int64)
    y_test_normalized = normalized_test.loc[:, CORRECTED_R1_TARGET].to_numpy(
        dtype=np.float64,
        copy=True,
    )[target_indices]
    y_test_original = original_test.loc[:, CORRECTED_R1_TARGET].to_numpy(
        dtype=np.float64,
        copy=True,
    )[target_indices]
    y_test_round_trip = target_target_scaler.inverse_transform(
        y_test_normalized.reshape(-1, 1)
    ).reshape(-1)
    round_trip_max_abs_difference = float(
        np.max(np.abs(y_test_round_trip - y_test_original))
    )
    round_trip_ok = bool(
        len(target_indices) == CORRECTED_R1_EXPECTED_SEQUENCES["target"]["test"]
        and np.allclose(
            y_test_round_trip,
            y_test_original,
            rtol=1e-10,
            atol=1e-6,
        )
    )
    if not round_trip_ok:
        raise ValueError("Target target-scaler round-trip audit failed")
    return {
        "source_scaler_used_on_target": False,
        "target_scaler_identity": True,
        "target_feature_scaler_fit_rows": target_feature_rows,
        "target_target_scaler_fit_rows": target_target_rows,
        "target_clipping_applied": target_manifest["clipping_applied"],
        "validation_transform_only": True,
        "test_transform_only": True,
        "target_target_scaler_round_trip": round_trip_ok,
        "target_target_scaler_round_trip_rows": len(target_indices),
        "target_target_scaler_round_trip_max_abs_difference": (
            round_trip_max_abs_difference
        ),
        "source_paths": {
            "feature": source_feature_path,
            "target": source_target_path,
        },
        "target_paths": {
            "feature": target_feature_path,
            "target": target_target_path,
        },
        "target_sha256": {
            "feature_scaler.joblib": _sha256_file(target_feature_path),
            "target_scaler.joblib": _sha256_file(target_target_path),
            "split_manifest.json": _sha256_file(target_directory / "split_manifest.json"),
            "scaler_manifest.json": _sha256_file(target_directory / "scaler_manifest.json"),
        },
    }


def _without_tl_anchor(target_contract, scaler_audit, *, experiment="A"):
    if experiment == "B":
        anchor = R2_EXPERIMENT_B_WITHOUT_TL_ANCHOR
        canonical_directory = (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / anchor["canonical_run_id"]
        )
        return {
            "canonical_run_id": anchor["canonical_run_id"],
            "canonical_directory_exists": canonical_directory.exists(),
            "evidence_run_id": None,
            "evidence_manifest": None,
            "evidence_status": anchor["status"],
            "target_profile": "Plant1/target_profile",
            "window": CORRECTED_R1_WINDOW,
            "horizon": CORRECTED_R1_HORIZON,
            "test_sequences": target_contract["test_sequence_count"],
            "test_first_target_timestamp": str(
                target_contract["test_first_target_timestamp"]
            ),
            "test_last_target_timestamp": str(
                target_contract["test_last_target_timestamp"]
            ),
            "output_activation": "sigmoid",
            "seed": 1234,
            "scaler_compatible": None,
            "compatible": None,
            "deferred": True,
            "note": anchor["note"],
        }

    anchor = R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR
    manifest_path = (_REPOSITORY_ROOT / anchor["evidence_manifest"]).resolve()
    manifest = _load_json(manifest_path)
    identity = manifest.get("identity", {})
    target_test = manifest.get("data_contract", {}).get("target", {}).get("test", {})
    model_contract = manifest.get("model_contract", {})
    environment = manifest.get("environment", {})
    recorded_hashes = (
        manifest.get("data_contract", {})
        .get("file_checksums", {})
        .get("target", {})
    )
    scaler_compatible = all(
        recorded_hashes.get(file_name) == digest
        for file_name, digest in scaler_audit["target_sha256"].items()
    )
    first_timestamp_equal = np.datetime64(target_test["first_target_timestamp"]) == np.datetime64(
        target_contract["test_first_target_timestamp"]
    )
    last_timestamp_equal = np.datetime64(target_test["last_target_timestamp"]) == np.datetime64(
        target_contract["test_last_target_timestamp"]
    )
    compatible = (
        manifest.get("status") == "PASS"
        and identity.get("experiment") == "A"
        and identity.get("target") == "Plant2/target_profile"
        and manifest["data_contract"].get("window") == 5
        and manifest["data_contract"].get("horizon") == 1
        and target_test.get("sequences") == 2607
        and first_timestamp_equal
        and last_timestamp_equal
        and model_contract.get("output_activation") == "sigmoid"
        and environment.get("seed") == 1234
        and scaler_compatible
    )
    if not compatible:
        raise ValueError("Sealed Plant2 Without-TL comparison anchor is incompatible")
    canonical_directory = (
        _REPOSITORY_ROOT / R2_SOURCE_FORMAL_OUTPUT_ROOT / anchor["canonical_run_id"]
    )
    return {
        "canonical_run_id": anchor["canonical_run_id"],
        "canonical_directory_exists": canonical_directory.exists(),
        "evidence_run_id": identity.get("run_id"),
        "evidence_manifest": manifest_path,
        "evidence_status": manifest.get("status"),
        "target_profile": identity.get("target"),
        "window": manifest["data_contract"]["window"],
        "horizon": manifest["data_contract"]["horizon"],
        "test_sequences": target_test["sequences"],
        "test_first_target_timestamp": target_test["first_target_timestamp"],
        "test_last_target_timestamp": target_test["last_target_timestamp"],
        "output_activation": model_contract["output_activation"],
        "seed": environment["seed"],
        "scaler_compatible": scaler_compatible,
        "compatible": compatible,
        "deferred": False,
        "note": (
            "Canonical alias directory is absent; compatibility is proven by the "
            "sealed completed evidence manifest."
        ),
    }


def _weights_equal(left, right):
    left_weights = left.get_weights()
    right_weights = right.get_weights()
    return len(left_weights) == len(right_weights) and all(
        np.array_equal(a, b) for a, b in zip(left_weights, right_weights)
    )


def _weight_shapes(layer):
    return tuple(tuple(int(value) for value in weight.shape) for weight in layer.get_weights())


def _parameter_counts(model, K):
    trainable = int(sum(K.count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(
        sum(K.count_params(weight) for weight in model.non_trainable_weights)
    )
    return trainable, non_trainable


def _candidate_summary(model, K):
    trainable, non_trainable = _parameter_counts(model, K)
    loss = model.loss if isinstance(model.loss, str) else model.loss.__name__
    optimizer_variables = model.optimizer.variables()
    return {
        "input_shape": tuple(model.input_shape[1:]),
        "total_params": model.count_params(),
        "output_activation": model.layers[-1].activation.__name__,
        "optimizer": model.optimizer.__class__.__name__,
        "initial_learning_rate": float(K.get_value(model.optimizer.learning_rate)),
        "optimizer_iteration": int(K.get_value(model.optimizer.iterations)),
        "optimizer_variable_count": len(optimizer_variables),
        "loss": loss,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "trainable_layer_names": tuple(
            layer.name for layer in model.layers if layer.trainable and layer.weights
        ),
        "variable_devices": tuple(sorted({weight.device for weight in model.weights})),
    }


def _validate_candidate(summary, mode, contract=None):
    contract = (
        R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    )
    expected = contract[mode]
    fixed = {
        "input_shape": contract["input_shape"],
        "total_params": contract["total_params"],
        "output_activation": contract["output_activation"],
        "optimizer": contract["optimizer"],
        "loss": contract["loss"],
        "trainable_params": expected["trainable_params"],
        "non_trainable_params": expected["non_trainable_params"],
    }
    for key, value in fixed.items():
        if summary[key] != value:
            raise ValueError(
                f"TL {mode} {key} mismatch: expected {value!r}, got {summary[key]!r}"
            )
    if not np.isclose(
        summary["initial_learning_rate"],
        contract["learning_rate"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"TL {mode} initial learning rate mismatch")
    if summary["optimizer_iteration"] != 0 or summary["optimizer_variable_count"] != 1:
        raise ValueError(f"TL {mode} optimizer is not freshly initialized")
    if not summary["variable_devices"] or any(
        "CPU:0" not in device.upper() for device in summary["variable_devices"]
    ):
        raise ValueError(f"TL {mode} model variables are not on CPU")


def _historical_contract(contract=None):
    contract = (
        R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    )
    if contract["experiment"] == "A":
        freeze_params_path = (
            _REPOSITORY_ROOT
            / "reports"
            / "Solar Energy Result"
            / "實驗 A：Plant1 → Plant2"
            / "transfer-learning (Freeze)"
            / "Plant2第二號發電機組"
            / "Plant1第一號發電機組"
            / "params.json"
        )
        unfreeze_params_path = (
            _REPOSITORY_ROOT
            / "reports"
            / "Solar Energy Result"
            / "實驗 A：Plant1 → Plant2"
            / "歷程記錄"
            / "transfer-learning (Unfreeze)_Batch=128&LR=3e-5_20250624"
            / "Plant2第二號發電機組"
            / "Plant1第一號發電機組"
            / "params.json"
        )
    else:
        freeze_params_path = (
            _REPOSITORY_ROOT
            / "reports"
            / "Solar Energy Result"
            / "實驗 B：Plant2 → Plant1"
            / "transfer-learning (Freeze)"
            / "Plant1第一號發電機組"
            / "Plant2第二號發電機組"
            / "params.json"
        )
        unfreeze_params_path = (
            _REPOSITORY_ROOT
            / "reports"
            / "Solar Energy Result"
            / "實驗 B：Plant2 → Plant1"
            / "歷程記錄"
            / "transfer-learning (Unfreeze)_bsize=128&lr=1e-5_20250627"
            / "Plant1第一號發電機組"
            / "Plant2第二號發電機組"
            / "params.json"
        )
    freeze_params = _load_json(freeze_params_path)
    unfreeze_params = _load_json(unfreeze_params_path)
    if not (
        freeze_params.get("nb_epochs") == contract["maximum_epochs"]
        and unfreeze_params.get("nb_epochs") == contract["maximum_epochs"]
        and freeze_params.get("seed") == contract["seed"]
        and unfreeze_params.get("seed") == contract["seed"]
        and freeze_params.get("freeze") is True
        and unfreeze_params.get("freeze") is False
    ):
        raise ValueError(
            f"Experiment-{contract['experiment']} historical TL params evidence mismatch"
        )
    return {
        "maximum_epochs": contract["maximum_epochs"],
        "seed": contract["seed"],
        "callbacks": contract["callbacks"],
        "training_shuffle": contract["training_shuffle"],
        "validation_shuffle": contract["validation_shuffle"],
        "freeze_batch_size": contract["freeze"]["batch_size"],
        "unfreeze_batch_size": contract["unfreeze"]["batch_size"],
        "learning_rate": contract["learning_rate"],
        "evidence": (
            str(freeze_params_path),
            str(unfreeze_params_path),
            "Legacy ReccurentTrainingGenerator randomizes Training order.",
            "Sealed formal evidence locks Training shuffle=True and ordered Validation.",
            "Legacy Validation generator randomization/padding is superseded by the "
            "approved Corrected R1 ordered, unpadded Validation contract.",
        ),
        "requires_human_confirmation": False,
    }


def resolve_transfer_learning_freeze_formal_run_directory(
    *,
    run_id=R2_EXPERIMENT_A_TL_FREEZE_FORMAL_RUN_ID,
    output_root=None,
):
    """Resolve, but never create, the unique formal Freeze run directory."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Formal TL Freeze run_id must be one path component")
    root = (
        _REPOSITORY_ROOT / R2_SOURCE_FORMAL_OUTPUT_ROOT
        if output_root is None
        else Path(output_root)
    )
    if not root.is_absolute():
        root = _REPOSITORY_ROOT / root
    return (root / run_id).resolve()


def assert_transfer_learning_freeze_formal_run_available(
    run_directory,
    *,
    allow_overwrite=False,
):
    """Enforce the formal Freeze non-overwrite policy without creating paths."""

    if allow_overwrite:
        raise ValueError("Formal TL Freeze requires allow_overwrite=False")
    run_directory = Path(run_directory)
    if run_directory.exists():
        raise FileExistsError(
            "Formal TL Freeze run already exists; overwrite is forbidden: "
            f"{run_directory}"
        )
    return run_directory


def _audit_transfer_learning_freeze_overwrite_guard():
    """Exercise the FileExistsError guard only against a temporary directory."""

    with TemporaryDirectory(prefix="r2_tl_freeze_overwrite_guard_") as temporary:
        try:
            assert_transfer_learning_freeze_formal_run_available(
                temporary,
                allow_overwrite=False,
            )
        except FileExistsError:
            return True
    raise AssertionError("Formal TL Freeze overwrite guard did not reject an existing path")


def resolve_transfer_learning_unfreeze_formal_run_directory(
    *,
    run_id=R2_EXPERIMENT_A_TL_UNFREEZE_FORMAL_RUN_ID,
    output_root=None,
):
    """Resolve, but never create, the unique formal Unfreeze run directory."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Formal TL Unfreeze run_id must be one path component")
    root = (
        _REPOSITORY_ROOT / R2_SOURCE_FORMAL_OUTPUT_ROOT
        if output_root is None
        else Path(output_root)
    )
    if not root.is_absolute():
        root = _REPOSITORY_ROOT / root
    return (root / run_id).resolve()


def assert_transfer_learning_unfreeze_formal_run_available(
    run_directory,
    *,
    allow_overwrite=False,
):
    """Enforce the formal Unfreeze non-overwrite policy without creating paths."""

    if allow_overwrite:
        raise ValueError("Formal TL Unfreeze requires allow_overwrite=False")
    run_directory = Path(run_directory)
    if run_directory.exists():
        raise FileExistsError(
            "Formal TL Unfreeze run already exists; overwrite is forbidden: "
            f"{run_directory}"
        )
    return run_directory


def _audit_transfer_learning_unfreeze_overwrite_guard():
    """Exercise the Unfreeze FileExistsError guard in a temporary directory."""

    with TemporaryDirectory(prefix="r2_tl_unfreeze_overwrite_guard_") as temporary:
        try:
            assert_transfer_learning_unfreeze_formal_run_available(
                temporary,
                allow_overwrite=False,
            )
        except FileExistsError:
            return True
    raise AssertionError(
        "Formal TL Unfreeze overwrite guard did not reject an existing path"
    )


def _git_evidence():
    safe_directory = str(_REPOSITORY_ROOT).replace("\\", "/")

    def git(*arguments):
        try:
            completed = subprocess.run(
                ("git", "-c", f"safe.directory={safe_directory}", *arguments),
                cwd=_REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot capture formal TL Git evidence: {exc}") from exc
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"Unexpected Git commit identifier: {commit!r}")
    porcelain = git("status", "--porcelain", "--untracked-files=normal")
    status_lines = tuple(line for line in porcelain.splitlines() if line)
    return {
        "commit": commit,
        "dirty": bool(status_lines),
        "dirty_status": "dirty" if status_lines else "clean",
        "status_porcelain": status_lines,
    }


def _formal_tl_core_code_evidence():
    evidence = {}
    for relative_name in _FORMAL_TL_CORE_EXECUTION_FILES:
        source_path = (_REPOSITORY_ROOT / relative_name).resolve()
        if not source_path.is_file() or _REPOSITORY_ROOT not in source_path.parents:
            raise FileNotFoundError(f"Formal TL core file missing: {source_path}")
        evidence[relative_name] = {
            "source_path": str(source_path),
            "sha256": _sha256_file(source_path),
        }
    return evidence


def _formal_target_data_evidence(corrected_r1_root, experiment):
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    profile_directory = corrected_r1_profile_path(root, experiment, "target").resolve()
    if not profile_directory.is_dir():
        raise FileNotFoundError(f"Target profile missing: {profile_directory}")
    evidence = {}
    for file_name in _FORMAL_TARGET_EVIDENCE_FILES:
        source_path = (profile_directory / file_name).resolve()
        if source_path.parent != profile_directory or not source_path.is_file():
            raise FileNotFoundError(f"Target evidence missing: {source_path}")
        evidence[file_name] = {
            "source_path": str(source_path),
            "run_artifact": file_name,
            "sha256": _sha256_file(source_path),
        }
    return {
        "target_profile_directory": str(profile_directory),
        "artifacts": evidence,
        "scalers_refit": False,
        "snapshot_method": "exclusive-byte-copy",
    }


def _copy_file_exclusive(source_path, destination_path):
    with Path(source_path).open("rb") as source, Path(destination_path).open(
        "xb"
    ) as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def _snapshot_target_evidence(run_directory, data_evidence):
    snapshot = {
        **data_evidence,
        "artifacts": {
            name: dict(record) for name, record in data_evidence["artifacts"].items()
        },
    }
    for file_name, record in snapshot["artifacts"].items():
        destination = Path(run_directory) / file_name
        _copy_file_exclusive(record["source_path"], destination)
        snapshot_sha256 = _sha256_file(destination)
        if snapshot_sha256 != record["sha256"]:
            raise ValueError(f"Target evidence snapshot mismatch: {file_name}")
        record["snapshot_path"] = str(destination)
        record["snapshot_sha256"] = snapshot_sha256
    return snapshot


def _environment_evidence(keras, tf, *, contract=None):
    import sklearn

    contract = R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "device": contract["device"],
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _write_json(path, payload, *, replace=False):
    mode = "w" if replace else "x"
    with Path(path).open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _write_text_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _mark_transfer_learning_formal_failed(manifest_path, manifest, error):
    failure_manifest = dict(manifest)
    failure_manifest.pop("completed_at_utc", None)
    failure_manifest.pop("best_checkpoint", None)
    failure_manifest.update(
        {
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    _write_json(
        manifest_path,
        failure_manifest,
        replace=Path(manifest_path).exists(),
    )
    return failure_manifest


def make_transfer_learning_freeze_formal_callbacks(
    run_directory,
    *,
    contract=None,
):
    """Construct the locked callbacks; constructors do not create artifacts."""

    from keras.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    contract = R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    run_directory = Path(run_directory)
    callbacks = contract["callbacks"]
    reduce = callbacks["ReduceLROnPlateau"]
    checkpoint = callbacks["ModelCheckpoint"]
    early = callbacks["EarlyStopping"]
    return (
        ReduceLROnPlateau(
            monitor=reduce["monitor"],
            factor=reduce["factor"],
            patience=reduce["patience"],
            min_lr=reduce["min_lr"],
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(run_directory / "checkpoints" / "best_model.hdf5"),
            monitor=checkpoint["monitor"],
            save_best_only=checkpoint["save_best_only"],
            verbose=1,
        ),
        EarlyStopping(
            monitor=early["monitor"],
            patience=early["patience"],
            restore_best_weights=early["restore_best_weights"],
            verbose=1,
        ),
        CSVLogger(str(run_directory / "epoch_log.csv")),
    )


def make_transfer_learning_unfreeze_formal_callbacks(
    run_directory,
    *,
    contract=None,
):
    """Construct the identical approved R2 callbacks for formal Unfreeze."""

    return make_transfer_learning_freeze_formal_callbacks(
        run_directory,
        contract=contract,
    )


def _callback_snapshot(callbacks):
    reduce_lr, checkpoint, early_stopping, csv_logger = callbacks
    return {
        "ReduceLROnPlateau": {
            "monitor": reduce_lr.monitor,
            "factor": float(reduce_lr.factor),
            "patience": int(reduce_lr.patience),
            "min_lr": float(reduce_lr.min_lr),
        },
        "ModelCheckpoint": {
            "monitor": checkpoint.monitor,
            "save_best_only": bool(checkpoint.save_best_only),
            "filepath": str(checkpoint.filepath),
        },
        "EarlyStopping": {
            "monitor": early_stopping.monitor,
            "patience": int(early_stopping.patience),
            "restore_best_weights": bool(early_stopping.restore_best_weights),
        },
        "CSVLogger": {"filename": str(csv_logger.filename)},
    }


def _validate_callback_snapshot(snapshot, run_directory, *, contract=None):
    contract = R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    expected = contract["callbacks"]
    for name in ("ReduceLROnPlateau", "ModelCheckpoint", "EarlyStopping"):
        for key, value in expected[name].items():
            if snapshot[name][key] != value:
                raise ValueError(f"Formal TL callback mismatch: {name}.{key}")
    if snapshot["ModelCheckpoint"]["filepath"] != str(
        Path(run_directory) / "checkpoints" / "best_model.hdf5"
    ):
        raise ValueError("Formal TL checkpoint path mismatch")
    if snapshot["CSVLogger"]["filename"] != str(
        Path(run_directory) / "epoch_log.csv"
    ):
        raise ValueError("Formal TL CSVLogger path mismatch")


def _configure_tl_cpu_runtime(seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    configured_hash_seed = os.environ.get("PYTHONHASHSEED")
    if configured_hash_seed not in (None, str(seed)):
        raise ValueError(f"PYTHONHASHSEED must be {seed}, got {configured_hash_seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)

    import keras
    import keras.backend as K
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow initialized before formal TL CPU policy") from exc
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return keras, K, tf


def _validate_formal_target_training_data(training):
    if training["X_train"].shape != (516, 5, 5) or training["y_train"].shape != (516,):
        raise ValueError("Formal TL Training sequence shape mismatch")
    if (
        training["X_validation"].shape != (126, 5, 5)
        or training["y_validation"].shape != (126,)
    ):
        raise ValueError("Formal TL Validation sequence shape mismatch")
    if training["test_sequence_count"] != 2607:
        raise ValueError("Formal TL Target Test sequence count mismatch")
    if "X_test" in training or "y_test" in training:
        raise ValueError("Target Test arrays are exposed to formal TL fit")
    if any(
        training[key]
        for key in (
            "target_test_used_for_training",
            "target_test_used_for_validation",
            "target_test_used_for_callbacks",
            "target_test_used_for_selection",
            "target_test_metrics_computed",
        )
    ):
        raise ValueError("Target Test is present in formal TL selection lifecycle")
    if training["validation_duplicate_count"] or training["validation_padding_count"]:
        raise ValueError("Formal TL Validation contains duplicate/padded samples")
    if not all(
        np.isfinite(value).all()
        for value in (
            training["X_train"],
            training["y_train"],
            training["X_validation"],
            training["y_validation"],
        )
    ):
        raise ValueError("Formal TL fit arrays contain non-finite values")


def _build_and_validate_freeze_candidate(
    source_model,
    K,
    tf,
    *,
    contract=None,
    architecture_directory=None,
    savefig=False,
):
    from utils.model import build_model

    contract = R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    source_contract = {
        "input_shape": tuple(source_model.input_shape[1:]),
        "total_params": source_model.count_params(),
        "output_activation": source_model.layers[-1].activation.__name__,
    }
    for key in ("input_shape", "total_params", "output_activation"):
        if source_contract[key] != contract[key]:
            raise ValueError(
                f"Sealed Source {key} mismatch: expected {contract[key]!r}, "
                f"got {source_contract[key]!r}"
            )
    with tf.device(contract["device"]):
        fresh_model = build_model(
            contract["input_shape"],
            False,
            None,
            verbose=False,
            savefig=False,
            output_activation=contract["output_activation"],
            learning_rate=contract["learning_rate"],
        )
        freeze_model = build_model(
            contract["input_shape"],
            False,
            architecture_directory,
            pre_model=source_model,
            freeze=True,
            verbose=False,
            savefig=savefig,
            output_activation=contract["output_activation"],
            learning_rate=contract["learning_rate"],
        )
    if getattr(source_model, "optimizer", None) is not None:
        raise ValueError("compile=False Source load restored an optimizer")
    summary = _candidate_summary(freeze_model, K)
    _validate_candidate(summary, "freeze", contract)
    transferred_state_ok = True
    transferred_layer_integrity = {}
    for index in contract["transfer_layer_indices"]:
        source_layer = source_model.layers[index]
        target_layer = freeze_model.layers[index]
        tensor_exact = tuple(
            np.array_equal(source_weight, target_weight)
            for source_weight, target_weight in zip(
                source_layer.get_weights(), target_layer.get_weights()
            )
        )
        layer_ok = (
            source_layer.__class__ is target_layer.__class__
            and _weight_shapes(source_layer) == _weight_shapes(target_layer)
            and bool(tensor_exact)
            and all(tensor_exact)
            and target_layer.trainable is False
        )
        transferred_state_ok = transferred_state_ok and layer_ok
        transferred_layer_integrity[str(index)] = {
            "source_layer": source_layer.name,
            "target_layer": target_layer.name,
            "tensor_exact": tensor_exact,
            "frozen": target_layer.trainable is False,
            "exact_and_frozen": layer_ok,
        }
    fresh_layer_integrity = {
        str(index): {
            "fresh_layer": fresh_model.layers[index].name,
            "target_layer": freeze_model.layers[index].name,
            "tensor_exact": _weights_equal(
                fresh_model.layers[index], freeze_model.layers[index]
            ),
            "trainable": freeze_model.layers[index].trainable is True,
        }
        for index in (1, 6)
    }
    fresh_initialization_ok = all(
        record["tensor_exact"] and record["trainable"]
        for record in fresh_layer_integrity.values()
    )
    if not transferred_state_ok:
        raise ValueError("Formal TL transferred/frozen layer gate failed")
    if not fresh_initialization_ok:
        raise ValueError("Formal TL non-transferred fresh initialization failed")
    frozen_weights = {
        index: tuple(
            np.array(weight, copy=True)
            for weight in freeze_model.layers[index].get_weights()
        )
        for index in contract["transfer_layer_indices"]
    }
    return {
        "fresh_model": fresh_model,
        "model": freeze_model,
        "source_model_contract": source_contract,
        "model_snapshot": summary,
        "frozen_transferred_layer_state": transferred_state_ok,
        "transferred_layer_integrity": transferred_layer_integrity,
        "non_transferred_fresh_initialization": fresh_initialization_ok,
        "fresh_layer_integrity": fresh_layer_integrity,
        "frozen_weights_before_fit": frozen_weights,
        "source_optimizer_transferred": False,
    }


def _build_and_validate_unfreeze_candidate(
    source_model,
    K,
    tf,
    *,
    contract=None,
    architecture_directory=None,
    savefig=False,
):
    """Build Unfreeze directly from Source and enforce its initial-weight gate."""

    from utils.model import build_model

    contract = R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    source_contract = {
        "input_shape": tuple(source_model.input_shape[1:]),
        "total_params": source_model.count_params(),
        "output_activation": source_model.layers[-1].activation.__name__,
    }
    for key in ("input_shape", "total_params", "output_activation"):
        if source_contract[key] != contract[key]:
            raise ValueError(
                f"Sealed Source {key} mismatch: expected {contract[key]!r}, "
                f"got {source_contract[key]!r}"
            )
    with tf.device(contract["device"]):
        fresh_model = build_model(
            contract["input_shape"],
            False,
            None,
            verbose=False,
            savefig=False,
            output_activation=contract["output_activation"],
            learning_rate=contract["learning_rate"],
        )
        unfreeze_model = build_model(
            contract["input_shape"],
            False,
            architecture_directory,
            pre_model=source_model,
            freeze=False,
            verbose=False,
            savefig=savefig,
            output_activation=contract["output_activation"],
            learning_rate=contract["learning_rate"],
        )
    if getattr(source_model, "optimizer", None) is not None:
        raise ValueError("compile=False Source load restored an optimizer")
    summary = _candidate_summary(unfreeze_model, K)
    _validate_candidate(summary, "unfreeze", contract)
    transfer_indices = contract["transfer_layer_indices"]
    transferred_layer_integrity = {}
    transferred_initial_weights_exact = True
    for index in transfer_indices:
        source_layer = source_model.layers[index]
        target_layer = unfreeze_model.layers[index]
        tensor_exact = tuple(
            np.array_equal(source_weight, target_weight)
            for source_weight, target_weight in zip(
                source_layer.get_weights(), target_layer.get_weights()
            )
        )
        layer_ok = (
            source_layer.__class__ is target_layer.__class__
            and _weight_shapes(source_layer) == _weight_shapes(target_layer)
            and bool(tensor_exact)
            and all(tensor_exact)
        )
        transferred_initial_weights_exact = (
            transferred_initial_weights_exact and layer_ok
        )
        transferred_layer_integrity[str(index)] = {
            "source_layer": source_layer.name,
            "target_layer": target_layer.name,
            "tensor_exact": tensor_exact,
            "trainable": target_layer.trainable is True,
            "exact": layer_ok,
        }
    transferred_layers_trainable = all(
        unfreeze_model.layers[index].trainable for index in transfer_indices
    )
    fresh_layer_integrity = {
        str(index): {
            "fresh_layer": fresh_model.layers[index].name,
            "target_layer": unfreeze_model.layers[index].name,
            "tensor_exact": _weights_equal(
                fresh_model.layers[index], unfreeze_model.layers[index]
            ),
            "trainable": unfreeze_model.layers[index].trainable is True,
        }
        for index in (1, 6)
    }
    input_projection_transferred = not fresh_layer_integrity["1"]["tensor_exact"]
    output_transferred = not fresh_layer_integrity["6"]["tensor_exact"]
    if not transferred_initial_weights_exact:
        raise ValueError("Formal Unfreeze transferred initial Source weights mismatch")
    if not transferred_layers_trainable:
        raise ValueError("Formal Unfreeze transferred layers are not trainable")
    if input_projection_transferred or output_transferred:
        raise ValueError("Formal Unfreeze Target-fresh initialization was overwritten")
    return {
        "fresh_model": fresh_model,
        "model": unfreeze_model,
        "source_model_contract": source_contract,
        "model_snapshot": summary,
        "transferred_initial_weights_exact": transferred_initial_weights_exact,
        "transferred_layer_integrity": transferred_layer_integrity,
        "transferred_layers_trainable": transferred_layers_trainable,
        "input_projection_transferred": input_projection_transferred,
        "output_transferred": output_transferred,
        "fresh_layer_integrity": fresh_layer_integrity,
        "source_optimizer_transferred": False,
        "freeze_optimizer_transferred": False,
    }


def _formal_static_safety_audit():
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    dry = functions["run_transfer_learning_freeze_formal_dry_run"]
    formal = functions["run_transfer_learning_freeze_formal"]
    failure = functions["_mark_transfer_learning_formal_failed"]

    def attribute_calls(node, name):
        return sum(
            1
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == name
        )

    formal_named_call_nodes = [
        child
        for child in ast.walk(formal)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    ]
    formal_named_calls = {child.func.id for child in formal_named_call_nodes}
    failure_literals = {
        child.value
        for child in ast.walk(failure)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    failure_handler_ready = (
        "_mark_transfer_learning_formal_failed" in formal_named_calls
        and any(isinstance(child, ast.Raise) for child in ast.walk(formal))
        and {
            "status",
            "failed",
            "failed_at_utc",
            "error_type",
            "error_message",
        }.issubset(failure_literals)
    )
    destructive_calls = sum(
        attribute_calls(formal, name)
        for name in ("unlink", "rmdir", "rmtree", "remove")
    )
    run_guard_ready = (
        "assert_transfer_learning_freeze_formal_run_available"
        in formal_named_calls
    )
    exclusive_run_directory_creation = any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "mkdir"
        and any(
            keyword.arg == "exist_ok"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in child.keywords
        )
        for child in ast.walk(formal)
    )
    main_text = (_REPOSITORY_ROOT / "main.py").read_text(encoding="utf-8")
    formal_action = 'if args["corrected_r1_action"] == "tl-freeze-formal-run":'
    formal_call = "tl_freeze_formal = run_transfer_learning_freeze_formal("
    experiment_b_guard = (
        'if args["experiment"] != "A":' in main_text
        and "Formal TL Freeze run is Experiment A only" in main_text
    )
    formal_execution_cli_exposed = formal_action in main_text and formal_call in main_text

    fit_lines = [
        child.lineno
        for child in ast.walk(formal)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "fit"
    ]
    predict_lines = [
        child.lineno
        for child in ast.walk(formal)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "predict"
    ]
    checkpoint_load_lines = [
        child.lineno
        for child in formal_named_call_nodes
        if child.func.id == "load_model"
        and child.args
        and isinstance(child.args[0], ast.Name)
        and child.args[0].id == "checkpoint_path"
    ]
    evaluation_load_lines = [
        child.lineno
        for child in formal_named_call_nodes
        if child.func.id == "load_target_evaluation_contract"
    ]
    checkpoint_guard_lines = [
        child.lineno
        for child in ast.walk(formal)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "is_file"
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "checkpoint_path"
    ]
    test_predict_after_best_checkpoint_fixed = (
        len(fit_lines) == 1
        and len(predict_lines) == 1
        and len(checkpoint_load_lines) == 1
        and len(evaluation_load_lines) == 1
        and len(checkpoint_guard_lines) == 1
        and fit_lines[0] < checkpoint_guard_lines[0]
        < checkpoint_load_lines[0] < evaluation_load_lines[0] < predict_lines[0]
    )
    return {
        "dry_run_fit_calls": attribute_calls(dry, "fit"),
        "dry_run_fit_generator_calls": attribute_calls(dry, "fit_generator"),
        "dry_run_train_on_batch_calls": attribute_calls(dry, "train_on_batch"),
        "dry_run_predict_calls": attribute_calls(dry, "predict"),
        "dry_run_predict_generator_calls": attribute_calls(
            dry, "predict_generator"
        ),
        "dry_run_evaluate_calls": attribute_calls(dry, "evaluate"),
        "formal_fit_calls": attribute_calls(formal, "fit"),
        "formal_fit_generator_calls": attribute_calls(formal, "fit_generator"),
        "formal_train_on_batch_calls": attribute_calls(formal, "train_on_batch"),
        "formal_predict_calls": attribute_calls(formal, "predict"),
        "formal_predict_generator_calls": attribute_calls(
            formal, "predict_generator"
        ),
        "formal_evaluate_calls": attribute_calls(formal, "evaluate"),
        "test_predict_after_best_checkpoint_fixed": (
            test_predict_after_best_checkpoint_fixed
        ),
        "failure_handler_ready": failure_handler_ready,
        "formal_destructive_calls": destructive_calls,
        "run_guard_ready": run_guard_ready,
        "exclusive_run_directory_creation": exclusive_run_directory_creation,
        "formal_execution_cli_exposed": formal_execution_cli_exposed,
        "experiment_b_formal_execution_exposed": (
            formal_execution_cli_exposed and not experiment_b_guard
        ),
    }


def _formal_unfreeze_static_safety_audit():
    """Audit shared C5/D9 dry/formal lifecycle and failure-safety structure."""

    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    dry = functions["run_transfer_learning_unfreeze_formal_dry_run"]
    formal = functions["run_transfer_learning_unfreeze_formal"]
    failure = functions["_mark_transfer_learning_formal_failed"]

    def attribute_calls(node, name):
        return sum(
            1
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == name
        )

    formal_named_call_nodes = [
        child
        for child in ast.walk(formal)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    ]
    formal_named_calls = {child.func.id for child in formal_named_call_nodes}
    failure_literals = {
        child.value
        for child in ast.walk(failure)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    failure_handler_ready = (
        "_mark_transfer_learning_formal_failed" in formal_named_calls
        and any(isinstance(child, ast.Raise) for child in ast.walk(formal))
        and {
            "status",
            "failed",
            "failed_at_utc",
            "error_type",
            "error_message",
        }.issubset(failure_literals)
    )
    destructive_calls = sum(
        attribute_calls(formal, name)
        for name in ("unlink", "rmdir", "rmtree", "remove")
    )
    run_guard_ready = (
        "assert_transfer_learning_unfreeze_formal_run_available"
        in formal_named_calls
    )
    exclusive_run_directory_creation = any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "mkdir"
        and any(
            keyword.arg == "exist_ok"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in child.keywords
        )
        for child in ast.walk(formal)
    )
    main_text = (_REPOSITORY_ROOT / "main.py").read_text(encoding="utf-8")
    formal_action = 'if args["corrected_r1_action"] == "tl-unfreeze-formal-run":'
    formal_call = "tl_unfreeze_formal = run_transfer_learning_unfreeze_formal("
    experiment_b_guard = (
        'if args["experiment"] != "A":' in main_text
        and "Formal TL Unfreeze run is Experiment A only" in main_text
    )
    formal_execution_cli_exposed = formal_action in main_text and formal_call in main_text

    fit_lines = [
        child.lineno
        for child in ast.walk(formal)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "fit"
    ]
    predict_lines = [
        child.lineno
        for child in ast.walk(formal)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "predict"
    ]
    checkpoint_load_lines = [
        child.lineno
        for child in formal_named_call_nodes
        if child.func.id == "load_model"
        and child.args
        and isinstance(child.args[0], ast.Name)
        and child.args[0].id == "checkpoint_path"
    ]
    evaluation_load_lines = [
        child.lineno
        for child in formal_named_call_nodes
        if child.func.id == "load_target_evaluation_contract"
    ]
    checkpoint_guard_lines = [
        child.lineno
        for child in ast.walk(formal)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "is_file"
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "checkpoint_path"
    ]
    test_predict_after_best_checkpoint_fixed = (
        len(fit_lines) == 1
        and len(predict_lines) == 1
        and len(checkpoint_load_lines) == 1
        and len(evaluation_load_lines) == 1
        and len(checkpoint_guard_lines) == 1
        and fit_lines[0] < checkpoint_guard_lines[0]
        < checkpoint_load_lines[0] < evaluation_load_lines[0] < predict_lines[0]
    )
    return {
        "dry_run_fit_calls": attribute_calls(dry, "fit"),
        "dry_run_fit_generator_calls": attribute_calls(dry, "fit_generator"),
        "dry_run_train_on_batch_calls": attribute_calls(dry, "train_on_batch"),
        "dry_run_predict_calls": attribute_calls(dry, "predict"),
        "dry_run_predict_generator_calls": attribute_calls(
            dry, "predict_generator"
        ),
        "dry_run_evaluate_calls": attribute_calls(dry, "evaluate"),
        "formal_fit_calls": attribute_calls(formal, "fit"),
        "formal_fit_generator_calls": attribute_calls(formal, "fit_generator"),
        "formal_train_on_batch_calls": attribute_calls(formal, "train_on_batch"),
        "formal_predict_calls": attribute_calls(formal, "predict"),
        "formal_predict_generator_calls": attribute_calls(
            formal, "predict_generator"
        ),
        "formal_evaluate_calls": attribute_calls(formal, "evaluate"),
        "test_predict_after_best_checkpoint_fixed": (
            test_predict_after_best_checkpoint_fixed
        ),
        "failure_handler_ready": failure_handler_ready,
        "formal_destructive_calls": destructive_calls,
        "run_guard_ready": run_guard_ready,
        "exclusive_run_directory_creation": exclusive_run_directory_creation,
        "formal_execution_cli_exposed": formal_execution_cli_exposed,
        "experiment_b_formal_execution_exposed": (
            formal_execution_cli_exposed and not experiment_b_guard
        ),
    }


def _transfer_learning_dry_run_static_safety_audit():
    """Prove the shared A/B dry-run contains no model lifecycle calls."""

    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    dry = functions["run_transfer_learning_dry_run"]

    def attribute_calls(name):
        return sum(
            1
            for child in ast.walk(dry)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == name
        )

    counts = {
        name: attribute_calls(name)
        for name in (
            "fit",
            "fit_generator",
            "train_on_batch",
            "predict",
            "predict_generator",
            "evaluate",
        )
    }
    if any(counts.values()):
        raise ValueError(f"TL dry-run lifecycle calls are not zero: {counts}")
    return counts


def run_transfer_learning_dry_run(*, corrected_r1_root=None, experiment="A"):
    """Run the shared A/B in-memory contract/weight-transfer audit with 0 epochs."""

    contract = resolve_r2_transfer_learning_contract(experiment)
    phase = "D4" if experiment == "B" else "C0"
    static_safety = _transfer_learning_dry_run_static_safety_audit()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTHONHASHSEED"] = str(contract["seed"])

    import keras.backend as K
    import tensorflow as tf
    from keras.models import load_model
    from utils.model import build_model

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError(
            f"TensorFlow initialized before Phase {phase} CPU policy"
        ) from exc
    random.seed(contract["seed"])
    np.random.seed(contract["seed"])
    tf.random.set_seed(contract["seed"])

    source = _source_checkpoint_gate(contract)
    target = load_target_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    if "X_test" in target or "y_test" in target:
        raise ValueError("Target Test arrays are exposed to the TL fit contract")
    expected_train_shape = (516, 5, 5)
    expected_validation_shape = (126, 5, 5)
    if target["X_train"].shape != expected_train_shape or target["y_train"].shape != (516,):
        raise ValueError("Target Training sequence shape mismatch")
    if (
        target["X_validation"].shape != expected_validation_shape
        or target["y_validation"].shape != (126,)
    ):
        raise ValueError("Target Validation sequence shape mismatch")
    if target["validation_duplicate_count"] or target["validation_padding_count"]:
        raise ValueError("Target Validation contains duplicate/padded samples")

    scaler = _scaler_isolation(
        corrected_r1_root,
        target,
        experiment=experiment,
    )
    baseline = _without_tl_anchor(
        target,
        scaler,
        experiment=experiment,
    )
    historical = _historical_contract(contract)
    baseline_state_before = (
        None
        if baseline["evidence_manifest"] is None
        else _file_state(baseline["evidence_manifest"].parent)
    )

    planned_directories = {
        mode: (
            _REPOSITORY_ROOT / R2_SOURCE_FORMAL_OUTPUT_ROOT / run_id
        ).resolve()
        for mode, run_id in contract["planned_run_ids"].items()
    }
    if any(path.exists() for path in planned_directories.values()):
        raise FileExistsError("A planned formal TL run directory already exists")

    source_model = fresh_model = freeze_model = unfreeze_model = None
    try:
        with tf.device(contract["device"]):
            source_model = load_model(source["checkpoint_path"], compile=False)
            fresh_model = build_model(
                contract["input_shape"],
                False,
                None,
                verbose=False,
                savefig=False,
                output_activation=contract["output_activation"],
                learning_rate=contract["learning_rate"],
            )
            freeze_model = build_model(
                contract["input_shape"],
                False,
                None,
                pre_model=source_model,
                freeze=True,
                verbose=False,
                savefig=False,
                output_activation=contract["output_activation"],
                learning_rate=contract["learning_rate"],
            )
            unfreeze_model = build_model(
                contract["input_shape"],
                False,
                None,
                pre_model=source_model,
                freeze=False,
                verbose=False,
                savefig=False,
                output_activation=contract["output_activation"],
                learning_rate=contract["learning_rate"],
            )

        if len(source_model.layers) != 7:
            raise ValueError("Unexpected sealed Source model layer count")
        if source_model.input_shape[1:] != contract["input_shape"]:
            raise ValueError("Sealed Source checkpoint input shape mismatch")
        if source_model.count_params() != contract["total_params"]:
            raise ValueError("Sealed Source checkpoint parameter count mismatch")
        if source_model.layers[-1].activation.__name__ != contract["output_activation"]:
            raise ValueError("Sealed Source output activation mismatch")
        source_optimizer_present = getattr(source_model, "optimizer", None) is not None
        if source_optimizer_present:
            raise ValueError("compile=False Source load unexpectedly restored optimizer")

        freeze_summary = _candidate_summary(freeze_model, K)
        unfreeze_summary = _candidate_summary(unfreeze_model, K)
        _validate_candidate(freeze_summary, "freeze", contract)
        _validate_candidate(unfreeze_summary, "unfreeze", contract)
        optimizer_isolation = {
            "source_final_learning_rate": source["final_learning_rate"],
            "source_optimizer_present": source_optimizer_present,
            "source_optimizer_state_transferred": False,
            "target_new_adam": (
                freeze_model.optimizer is not unfreeze_model.optimizer
                and freeze_summary["optimizer"] == "Adam"
                and unfreeze_summary["optimizer"] == "Adam"
            ),
            "freeze_initial_learning_rate": freeze_summary["initial_learning_rate"],
            "unfreeze_initial_learning_rate": unfreeze_summary["initial_learning_rate"],
            "target_optimizer_fresh": (
                freeze_summary["optimizer_iteration"] == 0
                and unfreeze_summary["optimizer_iteration"] == 0
                and freeze_summary["optimizer_variable_count"] == 1
                and unfreeze_summary["optimizer_variable_count"] == 1
            ),
        }
        if not optimizer_isolation["target_new_adam"] or not optimizer_isolation[
            "target_optimizer_fresh"
        ]:
            raise ValueError("Target optimizer isolation audit failed")

        layer_audit = []
        transfer_indices = set(contract["transfer_layer_indices"])
        for index, source_layer in enumerate(source_model.layers):
            freeze_layer = freeze_model.layers[index]
            unfreeze_layer = unfreeze_model.layers[index]
            fresh_layer = fresh_model.layers[index]
            transfer_expected = index in transfer_indices
            shape_compatible = (
                _weight_shapes(source_layer) == _weight_shapes(freeze_layer)
                == _weight_shapes(unfreeze_layer)
            )
            class_compatible = (
                source_layer.__class__ is freeze_layer.__class__
                and source_layer.__class__ is unfreeze_layer.__class__
            )
            freeze_source_exact = _weights_equal(source_layer, freeze_layer)
            unfreeze_source_exact = _weights_equal(source_layer, unfreeze_layer)
            freeze_fresh_exact = _weights_equal(fresh_layer, freeze_layer)
            unfreeze_fresh_exact = _weights_equal(fresh_layer, unfreeze_layer)
            if transfer_expected and not (
                class_compatible
                and shape_compatible
                and freeze_source_exact
                and unfreeze_source_exact
            ):
                raise ValueError(f"Transferred layer audit failed at index {index}")
            if index in (1, 6) and not (freeze_fresh_exact and unfreeze_fresh_exact):
                raise ValueError(f"Fresh Target initialization was overwritten at {index}")
            layer_audit.append(
                {
                    "layer_index": index,
                    "source_layer_name": source_layer.name,
                    "source_layer_class": source_layer.__class__.__name__,
                    "target_layer_name": freeze_layer.name,
                    "target_layer_class": freeze_layer.__class__.__name__,
                    "parameter_count": source_layer.count_params(),
                    "weight_shapes": _weight_shapes(source_layer),
                    "transfer_expected": transfer_expected,
                    "weight_equal_after_transfer": (
                        freeze_source_exact and unfreeze_source_exact
                        if transfer_expected
                        else None
                    ),
                    "freeze_trainable": freeze_layer.trainable,
                    "unfreeze_trainable": unfreeze_layer.trainable,
                    "fresh_initialization_preserved": (
                        freeze_fresh_exact and unfreeze_fresh_exact
                        if index in (1, 6)
                        else None
                    ),
                }
            )

        non_transferred = {
            "input_projection_transferred": False,
            "output_layer_transferred": False,
            "input_projection_fresh_initialization_preserved": layer_audit[1][
                "fresh_initialization_preserved"
            ],
            "output_fresh_initialization_preserved": layer_audit[6][
                "fresh_initialization_preserved"
            ],
        }
        if not all(
            (
                non_transferred["input_projection_fresh_initialization_preserved"],
                non_transferred["output_fresh_initialization_preserved"],
            )
        ):
            raise ValueError("Non-transferred layer initialization audit failed")

        source_modified = not (
            _sha256_file(source["checkpoint_path"])
            == source["checkpoint_sha256_before"]
            and _file_state(source["run_directory"])
            == source["artifact_state_before"]
        )
        baseline_modified = (
            False
            if baseline_state_before is None
            else _file_state(baseline["evidence_manifest"].parent)
            != baseline_state_before
        )
        if source_modified or baseline_modified:
            raise ValueError(
                f"A sealed Source/baseline artifact changed during {phase}"
            )
        if any(path.exists() for path in planned_directories.values()):
            raise ValueError(f"Phase {phase} created a formal TL run directory")

        notes = []
        if baseline["deferred"] or not baseline["canonical_directory_exists"]:
            notes.append(baseline["note"])
        return {
            "phase": phase,
            "status": "PASS WITH NOTES" if notes else "PASS",
            "notes": tuple(notes),
            "experiment": experiment,
            "direction": (
                "Plant2 -> Plant1" if experiment == "B" else "Plant1 -> Plant2"
            ),
            "source": {
                **source,
                "checkpoint_load": True,
                "input_shape": tuple(source_model.input_shape[1:]),
                "total_params": source_model.count_params(),
                "output_activation": source_model.layers[-1].activation.__name__,
                "modified": source_modified,
            },
            "target": {
                key: value
                for key, value in target.items()
                if key not in ("X_train", "y_train", "X_validation", "y_validation")
            },
            "scaler": scaler,
            "baseline": baseline,
            "historical": historical,
            "layers": tuple(layer_audit),
            "non_transferred": non_transferred,
            "freeze": {
                **freeze_summary,
                "batch_size": contract["freeze"]["batch_size"],
            },
            "unfreeze": {
                **unfreeze_summary,
                "batch_size": contract["unfreeze"]["batch_size"],
            },
            "optimizer_isolation": optimizer_isolation,
            "planned_run_directories": planned_directories,
            "allow_overwrite": contract["allow_overwrite"],
            "safety": {
                "epoch_executed": 0,
                "fit_calls": static_safety["fit"],
                "fit_generator_calls": static_safety["fit_generator"],
                "train_on_batch_calls": static_safety["train_on_batch"],
                "predict_calls": static_safety["predict"],
                "predict_generator_calls": static_safety["predict_generator"],
                "evaluate_calls": static_safety["evaluate"],
                "checkpoint_created": False,
                "prediction_created": False,
                "metrics_created": False,
                "formal_tl_run_created": False,
                "source_modified": source_modified,
                "baseline_modified": baseline_modified,
            },
        }
    finally:
        K.clear_session()


def run_transfer_learning_freeze_smoke(
    *,
    corrected_r1_root=None,
    experiment="A",
    seed=1234,
    device="/CPU:0",
    batch_size=64,
    epochs=1,
    verbose=2,
):
    """Run one approved in-memory A/B Freeze smoke epoch."""

    contract = resolve_r2_transfer_learning_contract(experiment)
    phase = "D5" if experiment == "B" else "C1"
    direction = "Plant2 -> Plant1" if experiment == "B" else "Plant1 -> Plant2"
    requested = {
        "experiment": experiment,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    expected = {
        "experiment": contract["experiment"],
        "seed": contract["seed"],
        "device": contract["device"],
        "batch_size": contract["freeze"]["batch_size"],
        "epochs": 1,
    }
    for key, value in requested.items():
        if value != expected[key]:
            raise ValueError(
                f"Phase {phase} requires {key}={expected[key]!r}, got {value!r}"
            )

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed not in (None, str(seed)):
        raise ValueError(
            f"PYTHONHASHSEED must be {seed} for Phase {phase}, got {hash_seed!r}"
        )
    os.environ["PYTHONHASHSEED"] = str(seed)

    import keras.backend as K
    import tensorflow as tf
    from keras.models import load_model
    from utils.model import build_model

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError(
            f"TensorFlow initialized before Phase {phase} CPU policy"
        ) from exc
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    source = _source_checkpoint_gate(contract)
    target = load_target_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    if "X_test" in target or "y_test" in target:
        raise ValueError(
            f"Target Test arrays are exposed to the Phase {phase} fit contract"
        )
    if any(
        (
            target["target_test_used_for_training"],
            target["target_test_used_for_validation"],
            target["target_test_used_for_callbacks"],
            target["target_test_used_for_selection"],
            target["target_test_metrics_computed"],
        )
    ):
        raise ValueError(f"Target Test is present in the Phase {phase} lifecycle")

    X_train = target["X_train"]
    y_train = target["y_train"]
    X_validation = target["X_validation"]
    y_validation = target["y_validation"]
    if X_train.shape != (516, 5, 5) or y_train.shape != (516,):
        raise ValueError(f"Phase {phase} Training sequence shape mismatch")
    if X_validation.shape != (126, 5, 5) or y_validation.shape != (126,):
        raise ValueError(f"Phase {phase} Validation sequence shape mismatch")
    if target["test_sequence_count"] != 2607:
        raise ValueError(f"Phase {phase} Target Test sequence count mismatch")
    if target["validation_duplicate_count"] or target["validation_padding_count"]:
        raise ValueError(
            f"Phase {phase} Validation contains duplicate/padded samples"
        )
    if not all(
        np.isfinite(array).all()
        for array in (X_train, y_train, X_validation, y_validation)
    ):
        raise ValueError(f"Phase {phase} fit arrays contain non-finite values")

    scaler = _scaler_isolation(
        corrected_r1_root,
        target,
        experiment=experiment,
    )
    baseline = _without_tl_anchor(
        target,
        scaler,
        experiment=experiment,
    )
    baseline_state_before = (
        None
        if baseline["evidence_manifest"] is None
        else _file_state(baseline["evidence_manifest"].parent)
    )
    corrected_root = Path(
        CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root
    ).resolve()
    corrected_state_before = _file_state(corrected_root)
    experiment_a_roots = (
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["source_run_id"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["freeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["unfreeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR["evidence_manifest"]
        ).resolve().parent,
    )
    experiment_a_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in experiment_a_roots
    }
    legacy_roots = (
        _REPOSITORY_ROOT / "dataset",
        _REPOSITORY_ROOT / "preprocess",
        _REPOSITORY_ROOT / "reports" / "Solar Energy Result",
    )
    legacy_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in legacy_roots
    }
    planned_directories = {
        mode: (
            _REPOSITORY_ROOT / R2_SOURCE_FORMAL_OUTPUT_ROOT / run_id
        ).resolve()
        for mode, run_id in contract["planned_run_ids"].items()
    }
    if any(path.exists() for path in planned_directories.values()):
        raise FileExistsError("A planned formal TL run directory already exists")
    experiment_b_formal_execution_exposed = False
    if experiment == "B":
        main_text = (_REPOSITORY_ROOT / "main.py").read_text(encoding="utf-8")
        formal_guards_present = (
            'if args["experiment"] != "A":' in main_text
            and "Formal TL Freeze run is Experiment A only" in main_text
            and "Formal TL Unfreeze run is Experiment A only" in main_text
        )
        if not formal_guards_present:
            experiment_b_formal_execution_exposed = True
            raise ValueError("Experiment-B formal TL execution is exposed")

    source_model = freeze_model = history = None
    try:
        with tf.device(device):
            source_model = load_model(source["checkpoint_path"], compile=False)
            freeze_model = build_model(
                contract["input_shape"],
                False,
                None,
                pre_model=source_model,
                freeze=True,
                verbose=False,
                savefig=False,
                output_activation=contract["output_activation"],
                learning_rate=contract["learning_rate"],
            )

        if getattr(source_model, "optimizer", None) is not None:
            raise ValueError("compile=False Source load restored an optimizer")
        summary_before = _candidate_summary(freeze_model, K)
        _validate_candidate(summary_before, "freeze", contract)
        transfer_indices = contract["transfer_layer_indices"]
        transferred_initial_weights_exact = all(
            _weights_equal(source_model.layers[index], freeze_model.layers[index])
            for index in transfer_indices
        )
        transferred_layers_frozen = all(
            freeze_model.layers[index].trainable is False
            for index in transfer_indices
        )
        if not transferred_initial_weights_exact:
            raise ValueError("Transferred weights do not exactly match Source before fit")
        if not transferred_layers_frozen:
            raise ValueError("A transferred layer is not frozen before fit")
        frozen_weights_before = {
            index: tuple(
                np.array(weight, copy=True)
                for weight in freeze_model.layers[index].get_weights()
            )
            for index in transfer_indices
        }
        optimizer_iteration_before = summary_before["optimizer_iteration"]

        # No callbacks are constructed or supplied.  Validation is the complete,
        # ordered, unpadded Corrected R1 Validation array; Target Test is absent.
        with tf.device(device):
            history = freeze_model.fit(
                X_train,
                y_train,
                validation_data=(X_validation, y_validation),
                batch_size=batch_size,
                epochs=epochs,
                shuffle=contract["training_shuffle"],
                callbacks=None,
                verbose=verbose,
            )

        executed_epochs = len(history.epoch)
        train_loss = float(history.history["loss"][0])
        validation_loss = float(history.history["val_loss"][0])
        train_loss_finite = bool(np.isfinite(train_loss))
        validation_loss_finite = bool(np.isfinite(validation_loss))
        optimizer_iteration_after = int(K.get_value(freeze_model.optimizer.iterations))
        frozen_transferred_weights_changed = any(
            not np.array_equal(before, after)
            for index in transfer_indices
            for before, after in zip(
                frozen_weights_before[index],
                freeze_model.layers[index].get_weights(),
            )
        )
        if executed_epochs != 1:
            raise RuntimeError(
                f"Phase {phase} executed {executed_epochs} epochs, expected 1"
            )
        if not train_loss_finite or not validation_loss_finite:
            raise RuntimeError(
                f"Phase {phase} produced non-finite train/validation loss"
            )
        if frozen_transferred_weights_changed:
            raise RuntimeError(
                f"A frozen transferred tensor changed during Phase {phase}"
            )
        if optimizer_iteration_before != 0 or optimizer_iteration_after <= 0:
            raise RuntimeError(f"Phase {phase} optimizer iteration lifecycle mismatch")

        source_modified = not (
            _sha256_file(source["checkpoint_path"])
            == source["checkpoint_sha256_before"]
            and _file_state(source["run_directory"])
            == source["artifact_state_before"]
        )
        baseline_modified = (
            False
            if baseline_state_before is None
            else _file_state(baseline["evidence_manifest"].parent)
            != baseline_state_before
        )
        experiment_a_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != experiment_a_state_before[str(path)]
            for path in experiment_a_roots
        )
        corrected_r1_evidence_modified = (
            _file_state(corrected_root) != corrected_state_before
        )
        legacy_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != legacy_state_before[str(path)]
            for path in legacy_roots
        )
        formal_tl_run_created = any(
            path.exists() for path in planned_directories.values()
        )
        if any(
            (
                source_modified,
                baseline_modified,
                experiment_a_modified,
                corrected_r1_evidence_modified,
                legacy_modified,
            )
        ):
            raise RuntimeError("A protected Source/baseline/Legacy artifact changed")
        if formal_tl_run_created:
            raise RuntimeError(f"Phase {phase} created a formal TL run directory")

        return {
            "phase": phase,
            "experiment": experiment,
            "direction": direction,
            "mode": "TL Freeze",
            "training_sequences": len(X_train),
            "validation_sequences": len(X_validation),
            "test_sequences": target["test_sequence_count"],
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "executed_epochs": executed_epochs,
            "device": device,
            "model_params": summary_before["total_params"],
            "trainable_params": summary_before["trainable_params"],
            "non_trainable_params": summary_before["non_trainable_params"],
            "output_activation": summary_before["output_activation"],
            "optimizer": summary_before["optimizer"],
            "initial_learning_rate": summary_before["initial_learning_rate"],
            "loss": summary_before["loss"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_loss_finite": train_loss_finite,
            "validation_loss_finite": validation_loss_finite,
            "transferred_initial_weights_exact": transferred_initial_weights_exact,
            "transferred_layers_frozen": transferred_layers_frozen,
            "frozen_transferred_weights_changed": (
                frozen_transferred_weights_changed
            ),
            "optimizer_iteration_before": optimizer_iteration_before,
            "optimizer_iteration_after": optimizer_iteration_after,
            "source_final_learning_rate": source["final_learning_rate"],
            "source_optimizer_state_transferred": False,
            "target_test_used_in_fit": False,
            "target_test_used_in_validation": False,
            "target_test_used_in_callbacks": False,
            "target_test_used_for_selection": False,
            "target_test_metrics_computed": False,
            "callbacks_created": False,
            "checkpoint_created": False,
            "formal_tl_run_created": formal_tl_run_created,
            "experiment_b_formal_execution_exposed": (
                experiment_b_formal_execution_exposed
            ),
            "outputs_written": False,
            "source_checkpoint_sha256_before": source[
                "checkpoint_sha256_before"
            ],
            "source_checkpoint_sha256_after": _sha256_file(
                source["checkpoint_path"]
            ),
            "source_checkpoint_modified": source_modified,
            "baseline_modified": baseline_modified,
            "experiment_a_modified": experiment_a_modified,
            "corrected_r1_evidence_modified": corrected_r1_evidence_modified,
            "legacy_modified": legacy_modified,
        }
    finally:
        K.clear_session()


def _sealed_freeze_run_gate(contract=None):
    """Validate the mapped sealed Freeze run without loading its checkpoint."""

    contract = R2_EXPERIMENT_A_TL_CONTRACT if contract is None else contract
    run_id = contract["planned_run_ids"]["freeze"]
    run_directory = (
        _REPOSITORY_ROOT / R2_SOURCE_FORMAL_OUTPUT_ROOT / run_id
    ).resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Sealed Freeze run missing: {run_directory}")
    manifest = _load_json(run_directory / "run_manifest.json")
    missing = [
        name
        for name in R2_EXPERIMENT_A_TL_FREEZE_FORMAL_OUTPUTS
        if not (run_directory / name).is_file()
        or (run_directory / name).stat().st_size == 0
    ]
    if (
        manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
        or manifest.get("artifact_count") != len(
            R2_EXPERIMENT_A_TL_FREEZE_FORMAL_OUTPUTS
        )
        or missing
    ):
        raise ValueError(f"Freeze run is not sealed/complete: missing={missing}")
    return {
        "run_id": run_id,
        "run_directory": run_directory,
        "artifact_state_before": _file_state(run_directory),
    }


def run_transfer_learning_unfreeze_smoke(
    *,
    corrected_r1_root=None,
    experiment="A",
    seed=1234,
    device="/CPU:0",
    batch_size=128,
    epochs=1,
    verbose=2,
):
    """Run the single approved in-memory A/B Unfreeze smoke epoch."""

    contract = resolve_r2_transfer_learning_contract(experiment)
    phase = "D8" if experiment == "B" else "C4"
    direction = _transfer_learning_direction(experiment)
    requested = {
        "experiment": experiment,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    expected = {
        "experiment": contract["experiment"],
        "seed": contract["seed"],
        "device": contract["device"],
        "batch_size": contract["unfreeze"]["batch_size"],
        "epochs": 1,
    }
    for key, value in requested.items():
        if value != expected[key]:
            raise ValueError(
                f"Phase {phase} requires {key}={expected[key]!r}, got {value!r}"
            )

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed not in (None, str(seed)):
        raise ValueError(
            f"PYTHONHASHSEED must be {seed} for Phase {phase}, got {hash_seed!r}"
        )
    os.environ["PYTHONHASHSEED"] = str(seed)

    import keras.backend as K
    import tensorflow as tf
    from keras.models import load_model
    from utils.model import build_model

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError(
            f"TensorFlow initialized before Phase {phase} CPU policy"
        ) from exc
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    source = _source_checkpoint_gate(contract)
    freeze = _sealed_freeze_run_gate(contract)
    target = load_target_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    if "X_test" in target or "y_test" in target:
        raise ValueError(
            f"Target Test arrays are exposed to the Phase {phase} fit contract"
        )
    if any(
        target[key]
        for key in (
            "target_test_used_for_training",
            "target_test_used_for_validation",
            "target_test_used_for_callbacks",
            "target_test_used_for_selection",
            "target_test_metrics_computed",
        )
    ):
        raise ValueError(f"Target Test is present in the Phase {phase} lifecycle")

    X_train = target["X_train"]
    y_train = target["y_train"]
    X_validation = target["X_validation"]
    y_validation = target["y_validation"]
    if X_train.shape != (516, 5, 5) or y_train.shape != (516,):
        raise ValueError(f"Phase {phase} Training sequence shape mismatch")
    if X_validation.shape != (126, 5, 5) or y_validation.shape != (126,):
        raise ValueError(f"Phase {phase} Validation sequence shape mismatch")
    if target["test_sequence_count"] != 2607:
        raise ValueError(f"Phase {phase} Target Test sequence count mismatch")
    if target["validation_duplicate_count"] or target["validation_padding_count"]:
        raise ValueError(
            f"Phase {phase} Validation contains duplicate/padded samples"
        )
    if not all(
        np.isfinite(array).all()
        for array in (X_train, y_train, X_validation, y_validation)
    ):
        raise ValueError(f"Phase {phase} fit arrays contain non-finite values")

    scaler = _scaler_isolation(
        corrected_r1_root,
        target,
        experiment=experiment,
    )
    baseline = _without_tl_anchor(
        target,
        scaler,
        experiment=experiment,
    )
    baseline_state_before = (
        None
        if baseline["evidence_manifest"] is None
        else _file_state(baseline["evidence_manifest"].parent)
    )
    target_evidence = _formal_target_data_evidence(corrected_r1_root, experiment)
    target_evidence_before = {
        name: _sha256_file(record["source_path"])
        for name, record in target_evidence["artifacts"].items()
    }
    corrected_root = Path(
        CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root
    ).resolve()
    corrected_state_before = _file_state(corrected_root)
    experiment_a_roots = (
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["source_run_id"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["freeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["unfreeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR["evidence_manifest"]
        ).resolve().parent,
    )
    experiment_a_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in experiment_a_roots
    }
    legacy_roots = (
        _REPOSITORY_ROOT / "dataset",
        _REPOSITORY_ROOT / "preprocess",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 A：Plant1 → Plant2",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 B：Plant2 → Plant1",
    )
    legacy_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in legacy_roots
    }
    unfreeze_run_directory = (
        _REPOSITORY_ROOT
        / R2_SOURCE_FORMAL_OUTPUT_ROOT
        / contract["planned_run_ids"]["unfreeze"]
    ).resolve()
    if unfreeze_run_directory.exists():
        raise FileExistsError(
            f"Formal Unfreeze run already exists: {unfreeze_run_directory}"
        )
    main_text = (_REPOSITORY_ROOT / "main.py").read_text(encoding="utf-8")
    experiment_b_formal_execution_exposed = not (
        'if args["experiment"] != "A":' in main_text
        and "Formal TL Unfreeze run is Experiment A only" in main_text
    )
    unfreeze_formal_dry_run_source = inspect.getsource(
        run_transfer_learning_unfreeze_formal_dry_run
    )
    experiment_b_formal_dry_run_exposed = not (
        "formal TL Unfreeze dry-run is Experiment A only"
        in unfreeze_formal_dry_run_source
    )
    if experiment == "B" and (
        experiment_b_formal_execution_exposed
        or experiment_b_formal_dry_run_exposed
    ):
        raise ValueError("Experiment-B formal Unfreeze lifecycle is exposed")
    experiment_a_contract = resolve_r2_transfer_learning_contract("A")
    experiment_a_contract_unchanged = (
        experiment_a_contract["source"] == "Plant1/source_profile"
        and experiment_a_contract["target"] == "Plant2/target_profile"
        and np.isclose(
            experiment_a_contract["learning_rate"],
            3e-5,
            rtol=0.0,
            atol=1e-15,
        )
        and experiment_a_contract["unfreeze"]["batch_size"] == 128
    )
    if not experiment_a_contract_unchanged:
        raise ValueError("Experiment-A Unfreeze smoke contract regressed")

    source_model = fresh_model = unfreeze_model = history = None
    try:
        with tf.device(device):
            source_model = load_model(source["checkpoint_path"], compile=False)
            fresh_model = build_model(
                contract["input_shape"],
                False,
                None,
                verbose=False,
                savefig=False,
                output_activation=contract["output_activation"],
                learning_rate=contract["learning_rate"],
            )
            unfreeze_model = build_model(
                contract["input_shape"],
                False,
                None,
                pre_model=source_model,
                freeze=False,
                verbose=False,
                savefig=False,
                output_activation=contract["output_activation"],
                learning_rate=contract["learning_rate"],
            )

        if getattr(source_model, "optimizer", None) is not None:
            raise ValueError("compile=False Source load restored an optimizer")
        summary_before = _candidate_summary(unfreeze_model, K)
        _validate_candidate(summary_before, "unfreeze", contract)
        transfer_indices = contract["transfer_layer_indices"]
        transferred_initial_integrity = {}
        for index in transfer_indices:
            source_layer = source_model.layers[index]
            unfreeze_layer = unfreeze_model.layers[index]
            source_weights = source_layer.get_weights()
            unfreeze_weights = unfreeze_layer.get_weights()
            tensor_exact = tuple(
                np.array_equal(source_weight, unfreeze_weight)
                for source_weight, unfreeze_weight in zip(
                    source_weights,
                    unfreeze_weights,
                )
            )
            transferred_initial_integrity[str(index)] = {
                "source_layer": source_layer.name,
                "unfreeze_layer": unfreeze_layer.name,
                "class_compatible": source_layer.__class__ is unfreeze_layer.__class__,
                "weight_shapes_compatible": (
                    _weight_shapes(source_layer) == _weight_shapes(unfreeze_layer)
                ),
                "tensor_exact": tensor_exact,
                "all_tensors_exact": bool(tensor_exact) and all(tensor_exact),
            }
        transferred_initial_weights_exact = all(
            record["class_compatible"]
            and record["weight_shapes_compatible"]
            and record["all_tensors_exact"]
            for record in transferred_initial_integrity.values()
        )
        transferred_layers_trainable = all(
            unfreeze_model.layers[index].trainable for index in transfer_indices
        )
        input_projection_transferred = not _weights_equal(
            fresh_model.layers[1], unfreeze_model.layers[1]
        )
        output_transferred = not _weights_equal(
            fresh_model.layers[6], unfreeze_model.layers[6]
        )
        if not transferred_initial_weights_exact:
            raise ValueError(f"Phase {phase} transferred initial weights mismatch")
        if not transferred_layers_trainable:
            raise ValueError(f"Phase {phase} transferred layers are not trainable")
        if input_projection_transferred or output_transferred:
            raise ValueError(
                f"Phase {phase} Target-fresh layer initialization was overwritten"
            )

        trainable_weights_before = {
            index: tuple(
                np.array(K.get_value(weight), copy=True)
                for weight in unfreeze_model.layers[index].trainable_weights
            )
            for index in transfer_indices
        }
        non_trainable_weights_before = {
            index: tuple(
                np.array(K.get_value(weight), copy=True)
                for weight in unfreeze_model.layers[index].non_trainable_weights
            )
            for index in transfer_indices
        }
        optimizer_iteration_before = summary_before["optimizer_iteration"]

        # No callbacks are constructed or supplied.  Target Test arrays are not
        # returned by the training contract and cannot enter this fit call.
        with tf.device(device):
            history = unfreeze_model.fit(
                X_train,
                y_train,
                validation_data=(X_validation, y_validation),
                batch_size=batch_size,
                epochs=epochs,
                shuffle=contract["training_shuffle"],
                callbacks=None,
                verbose=verbose,
            )

        executed_epochs = len(history.epoch)
        train_loss = float(history.history["loss"][0])
        validation_loss = float(history.history["val_loss"][0])
        train_loss_finite = bool(np.isfinite(train_loss))
        validation_loss_finite = bool(np.isfinite(validation_loss))
        optimizer_iteration_after = int(K.get_value(unfreeze_model.optimizer.iterations))
        changed_trainable_tensors = sum(
            not np.array_equal(before, K.get_value(after))
            for index in transfer_indices
            for before, after in zip(
                trainable_weights_before[index],
                unfreeze_model.layers[index].trainable_weights,
            )
        )
        changed_bn_moving_tensors = sum(
            not np.array_equal(before, K.get_value(after))
            for index in transfer_indices
            for before, after in zip(
                non_trainable_weights_before[index],
                unfreeze_model.layers[index].non_trainable_weights,
            )
        )
        any_transferred_trainable_tensor_changed = changed_trainable_tensors > 0
        if executed_epochs != 1:
            raise RuntimeError(
                f"Phase {phase} executed {executed_epochs} epochs, expected 1"
            )
        if not train_loss_finite or not validation_loss_finite:
            raise RuntimeError(
                f"Phase {phase} produced non-finite train/validation loss"
            )
        if optimizer_iteration_before != 0 or optimizer_iteration_after <= 0:
            raise RuntimeError(
                f"Phase {phase} optimizer iteration lifecycle mismatch"
            )
        if not any_transferred_trainable_tensor_changed:
            raise RuntimeError(
                f"No transferred trainable tensor changed in Phase {phase}"
            )

        source_modified = not (
            _sha256_file(source["checkpoint_path"])
            == source["checkpoint_sha256_before"]
            and _file_state(source["run_directory"])
            == source["artifact_state_before"]
        )
        freeze_modified = (
            _file_state(freeze["run_directory"])
            != freeze["artifact_state_before"]
        )
        baseline_modified = (
            False
            if baseline_state_before is None
            else _file_state(baseline["evidence_manifest"].parent)
            != baseline_state_before
        )
        experiment_a_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != experiment_a_state_before[str(path)]
            for path in experiment_a_roots
        )
        corrected_r1_evidence_modified = (
            _file_state(corrected_root) != corrected_state_before
        )
        target_evidence_modified = any(
            _sha256_file(target_evidence["artifacts"][name]["source_path"])
            != digest
            for name, digest in target_evidence_before.items()
        )
        legacy_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != legacy_state_before[str(path)]
            for path in legacy_roots
        )
        formal_unfreeze_run_created = unfreeze_run_directory.exists()
        if any(
            (
                source_modified,
                freeze_modified,
                baseline_modified,
                experiment_a_modified,
                corrected_r1_evidence_modified,
                target_evidence_modified,
                legacy_modified,
            )
        ):
            raise RuntimeError(f"Phase {phase} changed a protected artifact")
        if formal_unfreeze_run_created:
            raise RuntimeError(f"Phase {phase} created the formal Unfreeze run")

        return {
            "phase": phase,
            "experiment": experiment,
            "direction": direction,
            "mode": "TL Unfreeze",
            "source_run_id": source["run_id"],
            "source_checkpoint_path": source["checkpoint_path"],
            "source_final_learning_rate": source["final_learning_rate"],
            "freeze_run_id": freeze["run_id"],
            "unfreeze_initialized_from_source": True,
            "unfreeze_initialized_from_freeze": False,
            "training_sequences": len(X_train),
            "validation_sequences": len(X_validation),
            "test_sequences": target["test_sequence_count"],
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "executed_epochs": executed_epochs,
            "device": device,
            "model_params": summary_before["total_params"],
            "trainable_params": summary_before["trainable_params"],
            "non_trainable_params": summary_before["non_trainable_params"],
            "output_activation": summary_before["output_activation"],
            "optimizer": summary_before["optimizer"],
            "initial_learning_rate": summary_before["initial_learning_rate"],
            "loss": summary_before["loss"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_loss_finite": train_loss_finite,
            "validation_loss_finite": validation_loss_finite,
            "transferred_initial_weights_exact": transferred_initial_weights_exact,
            "transferred_initial_integrity": transferred_initial_integrity,
            "transferred_layers_trainable": transferred_layers_trainable,
            "input_projection_transferred": input_projection_transferred,
            "output_transferred": output_transferred,
            "any_transferred_trainable_tensor_changed": (
                any_transferred_trainable_tensor_changed
            ),
            "changed_trainable_tensor_count": changed_trainable_tensors,
            "changed_bn_moving_tensor_count": changed_bn_moving_tensors,
            "optimizer_iteration_before": optimizer_iteration_before,
            "optimizer_iteration_after": optimizer_iteration_after,
            "target_new_adam": True,
            "source_optimizer_state_transferred": False,
            "freeze_optimizer_state_transferred": False,
            "training_shuffle": contract["training_shuffle"],
            "validation_shuffle": contract["validation_shuffle"],
            "target_test_used_in_fit": False,
            "target_test_used_in_validation": False,
            "target_test_used_in_callbacks": False,
            "target_test_used_for_selection": False,
            "target_test_metrics_computed": False,
            "callbacks_created": False,
            "checkpoint_created": False,
            "formal_unfreeze_run_created": formal_unfreeze_run_created,
            "formal_unfreeze_run_directory": unfreeze_run_directory,
            "experiment_b_formal_execution_exposed": (
                experiment_b_formal_execution_exposed
                or experiment_b_formal_dry_run_exposed
            ),
            "experiment_a_contract_unchanged": experiment_a_contract_unchanged,
            "outputs_written": False,
            "source_checkpoint_sha256_before": source[
                "checkpoint_sha256_before"
            ],
            "source_checkpoint_sha256_after": _sha256_file(
                source["checkpoint_path"]
            ),
            "source_modified": source_modified,
            "freeze_modified": freeze_modified,
            "baseline_modified": baseline_modified,
            "without_tl_anchor_status": baseline["evidence_status"],
            "experiment_a_modified": experiment_a_modified,
            "corrected_r1_evidence_modified": corrected_r1_evidence_modified,
            "target_evidence_modified": target_evidence_modified,
            "legacy_modified": legacy_modified,
        }
    finally:
        K.clear_session()


def run_transfer_learning_freeze_formal_dry_run(
    *,
    corrected_r1_root=None,
    experiment="A",
    run_id=None,
    output_root=None,
):
    """Audit the shared A/B formal Freeze runner with zero lifecycle calls."""

    contract = resolve_r2_transfer_learning_contract(experiment)
    phase = "D6" if experiment == "B" else "C2"
    direction = _transfer_learning_direction(experiment)
    expected_run_id = contract["planned_run_ids"]["freeze"]
    run_id = expected_run_id if run_id is None else run_id
    if run_id != expected_run_id:
        raise ValueError(
            f"Phase {phase} formal TL Freeze run ID must be "
            f"{expected_run_id!r}, got {run_id!r}"
        )
    run_directory = resolve_transfer_learning_freeze_formal_run_directory(
        run_id=run_id,
        output_root=output_root,
    )
    assert_transfer_learning_freeze_formal_run_available(
        run_directory,
        allow_overwrite=contract["allow_overwrite"],
    )
    if run_directory.exists():
        raise AssertionError("Dry-run must not create the formal TL directory")
    overwrite_guard = _audit_transfer_learning_freeze_overwrite_guard()

    source = _source_checkpoint_gate(contract)
    training = load_target_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    _validate_formal_target_training_data(training)
    scaler = _scaler_isolation(
        corrected_r1_root,
        training,
        experiment=experiment,
    )
    baseline = _without_tl_anchor(
        training,
        scaler,
        experiment=experiment,
    )
    evaluation = load_target_evaluation_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    if evaluation["X_test"].shape != (2607, 5, 5):
        raise ValueError("Formal TL evaluation X_test shape mismatch")
    if not evaluation["round_trip_ok"]:
        raise ValueError("Formal TL Target scaler round-trip failed")

    baseline_state_before = (
        None
        if baseline["evidence_manifest"] is None
        else _file_state(baseline["evidence_manifest"].parent)
    )
    corrected_root = Path(
        CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root
    ).resolve()
    corrected_state_before = _file_state(corrected_root)
    experiment_a_roots = (
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["source_run_id"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["freeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["unfreeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR["evidence_manifest"]
        ).resolve().parent,
    )
    experiment_a_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in experiment_a_roots
    }
    legacy_roots = (
        _REPOSITORY_ROOT / "dataset",
        _REPOSITORY_ROOT / "preprocess",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 A：Plant1 → Plant2",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 B：Plant2 → Plant1",
    )
    legacy_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in legacy_roots
    }

    keras, K, tf = _configure_tl_cpu_runtime(contract["seed"])
    from keras.models import load_model

    source_model = freeze_model = fresh_model = None
    try:
        with tf.device(contract["device"]):
            source_model = load_model(source["checkpoint_path"], compile=False)
        model_gate = _build_and_validate_freeze_candidate(
            source_model,
            K,
            tf,
            contract=contract,
        )
        freeze_model = model_gate["model"]
        fresh_model = model_gate["fresh_model"]

        callbacks = make_transfer_learning_freeze_formal_callbacks(
            run_directory,
            contract=contract,
        )
        callback_snapshot = _callback_snapshot(callbacks)
        _validate_callback_snapshot(
            callback_snapshot,
            run_directory,
            contract=contract,
        )
        if run_directory.exists():
            raise AssertionError("Callback construction created the formal TL directory")
        callback_files_written = sum(
            path.exists()
            for path in (
                run_directory / "epoch_log.csv",
                run_directory / "checkpoints" / "best_model.hdf5",
            )
        )
        if callback_files_written:
            raise AssertionError("Callback construction wrote formal TL artifacts")

        git_evidence = _git_evidence()
        core_code_evidence = _formal_tl_core_code_evidence()
        target_data_evidence = _formal_target_data_evidence(
            corrected_r1_root,
            experiment,
        )
        environment = _environment_evidence(keras, tf, contract=contract)
        source_reference = {
            "source_run_id": source["run_id"],
            "source_checkpoint_path": str(source["checkpoint_path"]),
            "source_checkpoint_sha256": source["checkpoint_sha256_before"],
            "source_final_learning_rate": source["final_learning_rate"],
            "source_optimizer_transferred": False,
        }
        comparison_reference = {
            "canonical_run_id": baseline["canonical_run_id"],
            "actual_evidence_run_id": baseline["evidence_run_id"],
            "actual_evidence_manifest": (
                None
                if baseline["evidence_manifest"] is None
                else str(baseline["evidence_manifest"])
            ),
            "canonical_alias_directory_exists": baseline[
                "canonical_directory_exists"
            ],
            "status": baseline["evidence_status"],
            "note": baseline["note"],
        }
        static_safety = _formal_static_safety_audit()
        baseline_ready = (
            baseline["deferred"] is True or baseline["compatible"] is True
        )
        evidence_package_ready = (
            set(core_code_evidence) == set(_FORMAL_TL_CORE_EXECUTION_FILES)
            and set(target_data_evidence["artifacts"])
            == set(_FORMAL_TARGET_EVIDENCE_FILES)
            and target_data_evidence["scalers_refit"] is False
            and source_reference["source_checkpoint_sha256"]
            == _sha256_file(source["checkpoint_path"])
            and baseline_ready
            and environment["device"] == contract["device"]
            and "scikit-learn" in environment
            and "joblib" in environment
        )
        callbacks_pass = True
        failure_handler_pass = (
            static_safety["failure_handler_ready"]
            and static_safety["formal_destructive_calls"] == 0
            and static_safety["run_guard_ready"]
            and static_safety["exclusive_run_directory_creation"]
            and static_safety["formal_fit_calls"] == 1
            and static_safety["formal_predict_calls"] == 1
            and static_safety["formal_evaluate_calls"] == 0
            and static_safety["test_predict_after_best_checkpoint_fixed"]
        )
        dry_lifecycle_safe = all(
            static_safety[key] == 0
            for key in (
                "dry_run_fit_calls",
                "dry_run_fit_generator_calls",
                "dry_run_train_on_batch_calls",
                "dry_run_predict_calls",
                "dry_run_predict_generator_calls",
                "dry_run_evaluate_calls",
            )
        )
        experiment_a_contract = resolve_r2_transfer_learning_contract("A")
        experiment_a_contract_unchanged = (
            experiment_a_contract["source"] == "Plant1/source_profile"
            and experiment_a_contract["target"] == "Plant2/target_profile"
            and _transfer_learning_direction("A") == "Plant1 -> Plant2"
            and experiment_a_contract["planned_run_ids"]["freeze"]
            == R2_EXPERIMENT_A_TL_FREEZE_FORMAL_RUN_ID
            and np.isclose(
                experiment_a_contract["learning_rate"],
                3e-5,
                rtol=0.0,
                atol=1e-15,
            )
            and experiment_a_contract["freeze"]["batch_size"] == 64
            and experiment_a_contract["freeze"]["trainable_params"] == 121
            and experiment_a_contract["freeze"]["non_trainable_params"] == 46560
        )
        if not evidence_package_ready:
            raise ValueError("Formal TL evidence package is incomplete")
        if not callbacks_pass:
            raise ValueError("Formal TL callback contract failed")
        if not failure_handler_pass:
            raise ValueError("Formal TL failure handler/static isolation failed")
        if not dry_lifecycle_safe:
            raise ValueError(f"Phase {phase} dry-run contains a model lifecycle call")
        if not experiment_a_contract_unchanged:
            raise ValueError("Experiment-A formal Freeze contract regressed")
        if experiment == "B" and static_safety[
            "experiment_b_formal_execution_exposed"
        ]:
            raise ValueError("Experiment-B formal Freeze execution is exposed")

        source_modified = not (
            _sha256_file(source["checkpoint_path"])
            == source["checkpoint_sha256_before"]
            and _file_state(source["run_directory"])
            == source["artifact_state_before"]
        )
        baseline_modified = (
            False
            if baseline_state_before is None
            else _file_state(baseline["evidence_manifest"].parent)
            != baseline_state_before
        )
        experiment_a_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != experiment_a_state_before[str(path)]
            for path in experiment_a_roots
        )
        corrected_r1_evidence_modified = (
            _file_state(corrected_root) != corrected_state_before
        )
        legacy_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != legacy_state_before[str(path)]
            for path in legacy_roots
        )
        if any(
            (
                source_modified,
                baseline_modified,
                experiment_a_modified,
                corrected_r1_evidence_modified,
                legacy_modified,
            )
        ):
            raise RuntimeError(f"Phase {phase} changed a protected artifact")
        if run_directory.exists():
            raise RuntimeError(f"Phase {phase} created the formal TL run directory")

        notes = () if baseline["canonical_directory_exists"] else (baseline["note"],)
        return {
            "phase": phase,
            "status": "PASS WITH NOTES" if notes else "PASS",
            "notes": notes,
            "experiment": experiment,
            "direction": direction,
            "mode": "TL Freeze",
            "run_id": run_id,
            "formal_runner_ready": True,
            "run_directory": run_directory,
            "formal_run_directory_exists": False,
            "allow_overwrite": contract["allow_overwrite"],
            "overwrite_guard": overwrite_guard,
            "source": {
                **source_reference,
                **model_gate["source_model_contract"],
                "checkpoint_load": True,
                "modified": source_modified,
            },
            "target_rows": dict(training["rows"]),
            "target_sequences": dict(training["sequences"]),
            "target_scaler_fit_rows": evaluation["target_scaler_fit_rows"],
            "target_feature_scaler_fit_rows": scaler[
                "target_feature_scaler_fit_rows"
            ],
            "target_target_scaler_fit_rows": scaler[
                "target_target_scaler_fit_rows"
            ],
            "target_validation_transform_only": scaler[
                "validation_transform_only"
            ],
            "target_test_transform_only": scaler["test_transform_only"],
            "target_clipping_applied": scaler["target_clipping_applied"],
            "target_scaler_round_trip": evaluation["round_trip_ok"],
            "source_scaler_used_on_target": scaler["source_scaler_used_on_target"],
            "model": model_gate["model_snapshot"],
            "frozen_transferred_layer_state": model_gate[
                "frozen_transferred_layer_state"
            ],
            "transferred_layer_integrity": model_gate[
                "transferred_layer_integrity"
            ],
            "non_transferred_fresh_initialization": model_gate[
                "non_transferred_fresh_initialization"
            ],
            "fresh_layer_integrity": model_gate["fresh_layer_integrity"],
            "batch_size": contract["freeze"]["batch_size"],
            "maximum_epochs": contract["maximum_epochs"],
            "training_shuffle": contract["training_shuffle"],
            "validation_shuffle": contract["validation_shuffle"],
            "callbacks": callback_snapshot,
            "callbacks_pass": callbacks_pass,
            "callback_files_written": callback_files_written,
            "planned_outputs": _FORMAL_TL_FREEZE_OUTPUTS,
            "evidence_package": {
                "ready": evidence_package_ready,
                "identity": {
                    "experiment": experiment,
                    "direction": direction,
                    "mode": "TL Freeze",
                    "source": contract["source"],
                    "target": contract["target"],
                },
                "git": git_evidence,
                "core_code_sha256": core_code_evidence,
                "target_data_evidence": target_data_evidence,
                "source_checkpoint_reference": source_reference,
                "without_tl_comparison_reference": comparison_reference,
                "environment": environment,
            },
            "failure_manifest": failure_handler_pass,
            "static_safety": static_safety,
            "without_tl_anchor_status": baseline["evidence_status"],
            "experiment_b_formal_execution_exposed": static_safety[
                "experiment_b_formal_execution_exposed"
            ],
            "experiment_a_contract_unchanged": experiment_a_contract_unchanged,
            "epoch_executed": 0,
            "fit_calls": static_safety["dry_run_fit_calls"],
            "fit_generator_calls": static_safety["dry_run_fit_generator_calls"],
            "train_on_batch_calls": static_safety[
                "dry_run_train_on_batch_calls"
            ],
            "predict_calls": static_safety["dry_run_predict_calls"],
            "predict_generator_calls": static_safety[
                "dry_run_predict_generator_calls"
            ],
            "evaluate_calls": static_safety["dry_run_evaluate_calls"],
            "target_test_used_for_selection": False,
            "target_test_prediction_executed": False,
            "target_test_metrics_computed": False,
            "checkpoint_created": False,
            "metrics_created": False,
            "prediction_created": False,
            "formal_run_created": False,
            "source_modified": source_modified,
            "baseline_modified": baseline_modified,
            "experiment_a_modified": experiment_a_modified,
            "corrected_r1_evidence_modified": corrected_r1_evidence_modified,
            "legacy_modified": legacy_modified,
        }
    finally:
        K.clear_session()


def run_transfer_learning_unfreeze_formal_dry_run(
    *,
    corrected_r1_root=None,
    experiment="A",
    run_id=None,
    output_root=None,
):
    """Audit the shared A/B formal Unfreeze runner with zero lifecycle calls."""

    contract = resolve_r2_transfer_learning_contract(experiment)
    phase = "D9" if experiment == "B" else "C5"
    direction = _transfer_learning_direction(experiment)
    expected_run_id = contract["planned_run_ids"]["unfreeze"]
    run_id = expected_run_id if run_id is None else run_id
    if run_id != expected_run_id:
        raise ValueError(
            f"Phase {phase} formal TL Unfreeze run ID must be "
            f"{expected_run_id!r}, got {run_id!r}"
        )
    run_directory = resolve_transfer_learning_unfreeze_formal_run_directory(
        run_id=run_id,
        output_root=output_root,
    )
    assert_transfer_learning_unfreeze_formal_run_available(
        run_directory,
        allow_overwrite=contract["allow_overwrite"],
    )
    if run_directory.exists():
        raise AssertionError("Dry-run must not create the formal Unfreeze directory")
    overwrite_guard = _audit_transfer_learning_unfreeze_overwrite_guard()

    source = _source_checkpoint_gate(contract)
    if source["checkpoint_sha256_before"] != contract["source_checkpoint_sha256"]:
        raise ValueError("Formal Unfreeze Source checkpoint SHA256 mismatch")
    freeze = _sealed_freeze_run_gate(contract)
    training = load_target_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    _validate_formal_target_training_data(training)
    scaler = _scaler_isolation(
        corrected_r1_root,
        training,
        experiment=experiment,
    )
    baseline = _without_tl_anchor(
        training,
        scaler,
        experiment=experiment,
    )
    evaluation = load_target_evaluation_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    if evaluation["X_test"].shape != (2607, 5, 5):
        raise ValueError("Formal Unfreeze evaluation X_test shape mismatch")
    if not evaluation["round_trip_ok"]:
        raise ValueError("Formal Unfreeze Target scaler round-trip failed")

    baseline_state_before = (
        None
        if baseline["evidence_manifest"] is None
        else _file_state(baseline["evidence_manifest"].parent)
    )
    corrected_root = Path(
        CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root
    ).resolve()
    corrected_state_before = _file_state(corrected_root)
    experiment_a_roots = (
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["source_run_id"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["freeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["unfreeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR["evidence_manifest"]
        ).resolve().parent,
    )
    experiment_a_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in experiment_a_roots
    }
    target_data_evidence = _formal_target_data_evidence(
        corrected_r1_root,
        experiment,
    )
    target_evidence_state_before = {
        name: _sha256_file(record["source_path"])
        for name, record in target_data_evidence["artifacts"].items()
    }
    legacy_roots = (
        _REPOSITORY_ROOT / "dataset",
        _REPOSITORY_ROOT / "preprocess",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 A：Plant1 → Plant2",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 B：Plant2 → Plant1",
    )
    legacy_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in legacy_roots
    }

    keras, K, tf = _configure_tl_cpu_runtime(contract["seed"])
    from keras.models import load_model

    source_model = unfreeze_model = fresh_model = None
    try:
        with tf.device(contract["device"]):
            source_model = load_model(source["checkpoint_path"], compile=False)
        model_gate = _build_and_validate_unfreeze_candidate(
            source_model,
            K,
            tf,
            contract=contract,
        )
        unfreeze_model = model_gate["model"]
        fresh_model = model_gate["fresh_model"]

        callbacks = make_transfer_learning_unfreeze_formal_callbacks(
            run_directory,
            contract=contract,
        )
        callback_snapshot = _callback_snapshot(callbacks)
        _validate_callback_snapshot(
            callback_snapshot,
            run_directory,
            contract=contract,
        )
        if run_directory.exists():
            raise AssertionError("Callback construction created the Unfreeze run directory")
        callback_files_written = sum(
            path.exists()
            for path in (
                run_directory / "epoch_log.csv",
                run_directory / "checkpoints" / "best_model.hdf5",
            )
        )
        if callback_files_written:
            raise AssertionError("Callback construction wrote formal TL artifacts")

        git_evidence = _git_evidence()
        core_code_evidence = _formal_tl_core_code_evidence()
        environment = _environment_evidence(keras, tf, contract=contract)
        source_reference = {
            "source_run_id": source["run_id"],
            "source_checkpoint_path": str(source["checkpoint_path"]),
            "source_checkpoint_sha256": source["checkpoint_sha256_before"],
            "source_final_learning_rate": source["final_learning_rate"],
            "source_optimizer_transferred": False,
            "used_for_initialization": True,
        }
        freeze_reference = {
            "run_id": freeze["run_id"],
            "status": "complete",
            "sealed": True,
            "artifact_count": len(_FORMAL_TL_FREEZE_OUTPUTS),
            "role": "protected-comparison-only",
            "used_for_initialization": False,
            "optimizer_transferred": False,
        }
        comparison_reference = {
            "canonical_run_id": baseline["canonical_run_id"],
            "actual_evidence_run_id": baseline["evidence_run_id"],
            "actual_evidence_manifest": (
                None
                if baseline["evidence_manifest"] is None
                else str(baseline["evidence_manifest"])
            ),
            "canonical_alias_directory_exists": baseline[
                "canonical_directory_exists"
            ],
            "status": baseline["evidence_status"],
            "note": baseline["note"],
        }
        static_safety = _formal_unfreeze_static_safety_audit()
        baseline_ready = (
            baseline["deferred"] is True or baseline["compatible"] is True
        )
        evidence_package_ready = (
            set(core_code_evidence) == set(_FORMAL_TL_CORE_EXECUTION_FILES)
            and set(target_data_evidence["artifacts"])
            == set(_FORMAL_TARGET_EVIDENCE_FILES)
            and target_data_evidence["scalers_refit"] is False
            and source_reference["source_checkpoint_sha256"]
            == contract["source_checkpoint_sha256"]
            and source_reference["used_for_initialization"] is True
            and freeze_reference["used_for_initialization"] is False
            and baseline_ready
            and environment["device"] == contract["device"]
            and "scikit-learn" in environment
            and "joblib" in environment
        )
        callbacks_pass = True
        failure_handler_pass = (
            static_safety["failure_handler_ready"]
            and static_safety["formal_destructive_calls"] == 0
            and static_safety["run_guard_ready"]
            and static_safety["exclusive_run_directory_creation"]
            and static_safety["formal_fit_calls"] == 1
            and static_safety["formal_fit_generator_calls"] == 0
            and static_safety["formal_train_on_batch_calls"] == 0
            and static_safety["formal_predict_calls"] == 1
            and static_safety["formal_predict_generator_calls"] == 0
            and static_safety["formal_evaluate_calls"] == 0
            and static_safety["test_predict_after_best_checkpoint_fixed"]
        )
        dry_lifecycle_safe = all(
            static_safety[key] == 0
            for key in (
                "dry_run_fit_calls",
                "dry_run_fit_generator_calls",
                "dry_run_train_on_batch_calls",
                "dry_run_predict_calls",
                "dry_run_predict_generator_calls",
                "dry_run_evaluate_calls",
            )
        )
        experiment_a_contract = resolve_r2_transfer_learning_contract("A")
        experiment_a_contract_unchanged = (
            experiment_a_contract["source"] == "Plant1/source_profile"
            and experiment_a_contract["target"] == "Plant2/target_profile"
            and _transfer_learning_direction("A") == "Plant1 -> Plant2"
            and experiment_a_contract["planned_run_ids"]["unfreeze"]
            == R2_EXPERIMENT_A_TL_UNFREEZE_FORMAL_RUN_ID
            and np.isclose(
                experiment_a_contract["learning_rate"],
                3e-5,
                rtol=0.0,
                atol=1e-15,
            )
            and experiment_a_contract["unfreeze"]["batch_size"] == 128
            and experiment_a_contract["unfreeze"]["trainable_params"] == 46441
            and experiment_a_contract["unfreeze"]["non_trainable_params"] == 240
        )
        if not evidence_package_ready:
            raise ValueError("Formal Unfreeze evidence package is incomplete")
        if not callbacks_pass:
            raise ValueError("Formal Unfreeze callback contract failed")
        if not failure_handler_pass:
            raise ValueError("Formal Unfreeze failure handler/static isolation failed")
        if not dry_lifecycle_safe:
            raise ValueError(f"Phase {phase} dry-run contains a model lifecycle call")
        if not experiment_a_contract_unchanged:
            raise ValueError("Experiment-A formal Unfreeze contract regressed")
        if experiment == "B" and static_safety[
            "experiment_b_formal_execution_exposed"
        ]:
            raise ValueError("Experiment-B formal Unfreeze execution is exposed")

        source_modified = not (
            _sha256_file(source["checkpoint_path"])
            == source["checkpoint_sha256_before"]
            and _file_state(source["run_directory"])
            == source["artifact_state_before"]
        )
        freeze_modified = (
            _file_state(freeze["run_directory"])
            != freeze["artifact_state_before"]
        )
        baseline_modified = (
            False
            if baseline_state_before is None
            else _file_state(baseline["evidence_manifest"].parent)
            != baseline_state_before
        )
        experiment_a_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != experiment_a_state_before[str(path)]
            for path in experiment_a_roots
        )
        corrected_r1_evidence_modified = (
            _file_state(corrected_root) != corrected_state_before
        )
        target_evidence_modified = any(
            _sha256_file(target_data_evidence["artifacts"][name]["source_path"])
            != digest
            for name, digest in target_evidence_state_before.items()
        )
        legacy_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != legacy_state_before[str(path)]
            for path in legacy_roots
        )
        if any(
            (
                source_modified,
                freeze_modified,
                baseline_modified,
                experiment_a_modified,
                corrected_r1_evidence_modified,
                target_evidence_modified,
                legacy_modified,
            )
        ):
            raise RuntimeError(f"Phase {phase} changed a protected artifact")
        if run_directory.exists():
            raise RuntimeError(
                f"Phase {phase} created the formal Unfreeze run directory"
            )

        notes = () if baseline["canonical_directory_exists"] else (baseline["note"],)
        return {
            "phase": phase,
            "status": "PASS WITH NOTES" if notes else "PASS",
            "notes": notes,
            "experiment": experiment,
            "direction": direction,
            "mode": "TL Unfreeze",
            "run_id": run_id,
            "formal_runner_ready": True,
            "run_directory": run_directory,
            "formal_run_directory_exists": False,
            "allow_overwrite": contract["allow_overwrite"],
            "overwrite_guard": overwrite_guard,
            "source": {
                **source_reference,
                **model_gate["source_model_contract"],
                "checkpoint_load": True,
                "modified": source_modified,
            },
            "unfreeze_initialized_from_source": True,
            "unfreeze_initialized_from_freeze": False,
            "freeze_reference": freeze_reference,
            "target_rows": dict(training["rows"]),
            "target_sequences": dict(training["sequences"]),
            "target_scaler_fit_rows": evaluation["target_scaler_fit_rows"],
            "target_feature_scaler_fit_rows": scaler[
                "target_feature_scaler_fit_rows"
            ],
            "target_target_scaler_fit_rows": scaler[
                "target_target_scaler_fit_rows"
            ],
            "target_validation_transform_only": scaler[
                "validation_transform_only"
            ],
            "target_test_transform_only": scaler["test_transform_only"],
            "target_clipping_applied": scaler["target_clipping_applied"],
            "target_scaler_round_trip": evaluation["round_trip_ok"],
            "source_scaler_used_on_target": scaler["source_scaler_used_on_target"],
            "model": model_gate["model_snapshot"],
            "transferred_initial_weights_exact": model_gate[
                "transferred_initial_weights_exact"
            ],
            "transferred_layer_integrity": model_gate[
                "transferred_layer_integrity"
            ],
            "transferred_layers_trainable": model_gate[
                "transferred_layers_trainable"
            ],
            "input_projection_transferred": model_gate[
                "input_projection_transferred"
            ],
            "output_transferred": model_gate["output_transferred"],
            "fresh_layer_integrity": model_gate["fresh_layer_integrity"],
            "source_optimizer_transferred": model_gate[
                "source_optimizer_transferred"
            ],
            "freeze_optimizer_transferred": model_gate[
                "freeze_optimizer_transferred"
            ],
            "batch_size": contract["unfreeze"]["batch_size"],
            "maximum_epochs": contract["maximum_epochs"],
            "training_shuffle": contract["training_shuffle"],
            "validation_shuffle": contract["validation_shuffle"],
            "callbacks": callback_snapshot,
            "callbacks_pass": callbacks_pass,
            "callback_files_written": callback_files_written,
            "planned_outputs": _FORMAL_TL_UNFREEZE_OUTPUTS,
            "evidence_package": {
                "ready": evidence_package_ready,
                "identity": {
                    "experiment": experiment,
                    "direction": direction,
                    "mode": "TL Unfreeze",
                    "source": contract["source"],
                    "target": contract["target"],
                },
                "git": git_evidence,
                "core_code_sha256": core_code_evidence,
                "target_data_evidence": target_data_evidence,
                "source_checkpoint_reference": source_reference,
                "freeze_reference": freeze_reference,
                "without_tl_comparison_reference": comparison_reference,
                "environment": environment,
            },
            "failure_manifest": failure_handler_pass,
            "static_safety": static_safety,
            "without_tl_anchor_status": baseline["evidence_status"],
            "experiment_b_formal_execution_exposed": static_safety[
                "experiment_b_formal_execution_exposed"
            ],
            "experiment_a_contract_unchanged": experiment_a_contract_unchanged,
            "epoch_executed": 0,
            "fit_calls": static_safety["dry_run_fit_calls"],
            "fit_generator_calls": static_safety["dry_run_fit_generator_calls"],
            "train_on_batch_calls": static_safety[
                "dry_run_train_on_batch_calls"
            ],
            "predict_calls": static_safety["dry_run_predict_calls"],
            "predict_generator_calls": static_safety[
                "dry_run_predict_generator_calls"
            ],
            "evaluate_calls": static_safety["dry_run_evaluate_calls"],
            "target_test_used_for_selection": False,
            "target_test_prediction_executed": False,
            "target_test_metrics_computed": False,
            "checkpoint_created": False,
            "metrics_created": False,
            "prediction_created": False,
            "formal_run_created": False,
            "source_modified": source_modified,
            "freeze_modified": freeze_modified,
            "baseline_modified": baseline_modified,
            "experiment_a_modified": experiment_a_modified,
            "corrected_r1_evidence_modified": corrected_r1_evidence_modified,
            "target_evidence_modified": target_evidence_modified,
            "legacy_modified": legacy_modified,
        }
    finally:
        K.clear_session()


def _regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    return {
        "n": len(y_true),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": float(1.0 - np.sum(np.square(residual)) / denominator),
    }


def _write_predictions(path, timestamps, indices, y_true, y_pred, scale):
    true_name = f"y_true_{scale}"
    pred_name = f"y_pred_{scale}"
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("target_timestamp", "target_index", true_name, pred_name),
        )
        writer.writeheader()
        for timestamp, index, observed, predicted in zip(
            timestamps,
            indices,
            y_true,
            y_pred,
        ):
            writer.writerow(
                {
                    "target_timestamp": str(timestamp),
                    "target_index": int(index),
                    true_name: float(observed),
                    pred_name: float(predicted),
                }
            )


def _save_learning_curve(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history.history["loss"], label="Training loss")
    axis.plot(history.history["val_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Compiled Loss (MSE + regularization)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_original_scale_diagnostics(
    run_directory,
    timestamps,
    y_true,
    y_pred,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_directory = Path(run_directory)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = y_true - y_pred

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(timestamps, y_true, label="Observed", linewidth=1)
    axis.plot(timestamps, y_pred, label="Predicted", linewidth=1)
    axis.set_xlabel("Target timestamp")
    axis.set_ylabel("DC_POWER (kW)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(
        run_directory / "prediction_plot_original_scale.png",
        dpi=150,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_true, y_pred, s=10, alpha=0.45)
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    axis.plot((lower, upper), (lower, upper), linestyle="--", color="black")
    axis.set_xlabel("Observed DC_POWER (kW)")
    axis.set_ylabel("Predicted DC_POWER (kW)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_directory / "yy_plot_original_scale.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(y_pred, residual, s=10, alpha=0.45)
    axis.axhline(0.0, linestyle="--", color="black")
    axis.set_xlabel("Predicted DC_POWER (kW)")
    axis.set_ylabel("Residual = True - Predicted (kW)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_directory / "residual_plot_original_scale.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(residual, bins=40, alpha=0.8)
    axis.set_xlabel("Prediction Error = True - Predicted (kW)")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        run_directory / "error_histogram_original_scale.png",
        dpi=150,
    )
    plt.close(figure)


def run_transfer_learning_freeze_formal(
    *,
    corrected_r1_root=None,
    experiment="A",
    run_id=None,
    output_root=None,
    maximum_epochs=500,
    allow_overwrite=False,
    verbose=1,
):
    """Execute the locked formal Freeze lifecycle after explicit authorization.

    Phase C2/D6 define and audit this shared implementation.  Experiment-B CLI
    dispatch remains blocked until separately authorized.  Target Test is loaded
    only after the best Validation checkpoint exists and is fixed.
    """

    contract = resolve_r2_transfer_learning_contract(experiment)
    direction = _transfer_learning_direction(experiment)
    expected_run_id = contract["planned_run_ids"]["freeze"]
    run_id = expected_run_id if run_id is None else run_id
    requested = {
        "experiment": experiment,
        "run_id": run_id,
        "maximum_epochs": maximum_epochs,
        "allow_overwrite": allow_overwrite,
    }
    expected = {
        "experiment": contract["experiment"],
        "run_id": expected_run_id,
        "maximum_epochs": contract["maximum_epochs"],
        "allow_overwrite": contract["allow_overwrite"],
    }
    for key, value in requested.items():
        if value != expected[key]:
            raise ValueError(
                f"Formal TL Freeze requires {key}={expected[key]!r}, got {value!r}"
            )

    run_directory = resolve_transfer_learning_freeze_formal_run_directory(
        run_id=run_id,
        output_root=output_root,
    )
    assert_transfer_learning_freeze_formal_run_available(
        run_directory,
        allow_overwrite=allow_overwrite,
    )
    training = load_target_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    _validate_formal_target_training_data(training)
    scaler = _scaler_isolation(
        corrected_r1_root,
        training,
        experiment=experiment,
    )
    baseline = _without_tl_anchor(
        training,
        scaler,
        experiment=experiment,
    )
    source = _source_checkpoint_gate(contract)

    keras, K, tf = _configure_tl_cpu_runtime(contract["seed"])
    from keras.models import load_model
    from utils.model import rmse

    git_evidence = _git_evidence()
    core_code_evidence = _formal_tl_core_code_evidence()
    target_data_evidence = _formal_target_data_evidence(
        corrected_r1_root,
        experiment,
    )
    target_evidence_state_before = {
        name: _sha256_file(record["source_path"])
        for name, record in target_data_evidence["artifacts"].items()
    }
    environment = _environment_evidence(keras, tf, contract=contract)
    source_reference = {
        "source_run_id": source["run_id"],
        "source_checkpoint_path": str(source["checkpoint_path"]),
        "source_checkpoint_sha256": source["checkpoint_sha256_before"],
        "source_final_learning_rate": source["final_learning_rate"],
        "source_optimizer_transferred": False,
    }
    comparison_reference = {
        "canonical_run_id": baseline["canonical_run_id"],
        "actual_evidence_run_id": baseline["evidence_run_id"],
        "actual_evidence_manifest": (
            None
            if baseline["evidence_manifest"] is None
            else str(baseline["evidence_manifest"])
        ),
        "canonical_alias_directory_exists": baseline["canonical_directory_exists"],
        "status": baseline["evidence_status"],
        "note": baseline["note"],
    }
    baseline_state_before = (
        None
        if baseline["evidence_manifest"] is None
        else _file_state(baseline["evidence_manifest"].parent)
    )

    started_at = datetime.now(timezone.utc).isoformat()
    checkpoint_path = run_directory / "checkpoints" / "best_model.hdf5"
    manifest_path = run_directory / "run_manifest.json"
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at,
        "run_directory": str(run_directory),
        "allow_overwrite": False,
        "experiment": experiment,
        "direction": direction,
        "mode": "TL Freeze",
        "source": contract["source"],
        "target": contract["target"],
        "git": git_evidence,
        "core_code_sha256": core_code_evidence,
        "source_checkpoint_reference": source_reference,
        "target_data_evidence": target_data_evidence,
        "without_tl_comparison_reference": comparison_reference,
        "selection_split": "validation",
        "target_test_used_for_selection": False,
        "target_test_evaluation_stage": "after-best-checkpoint-fixed",
        "metric_units": R2_EXPERIMENT_A_TL_FREEZE_OUTPUT_UNITS,
        "planned_outputs": list(_FORMAL_TL_FREEZE_OUTPUTS),
    }
    params = {
        **{key: value for key, value in contract.items() if key != "callbacks"},
        "run_id": run_id,
        "mode": "TL Freeze",
        "batch_size": contract["freeze"]["batch_size"],
        "trainable_params": contract["freeze"]["trainable_params"],
        "non_trainable_params": contract["freeze"]["non_trainable_params"],
        "callbacks": contract["callbacks"],
        "training_sequences": len(training["X_train"]),
        "validation_sequences": len(training["X_validation"]),
        "test_sequences": training["test_sequence_count"],
        "metric_units": R2_EXPERIMENT_A_TL_FREEZE_OUTPUT_UNITS,
    }

    source_model = fresh_model = model = best_model = None
    run_directory.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(manifest_path, manifest)
        (run_directory / "checkpoints").mkdir(exist_ok=False)
        _write_text_exclusive(
            run_directory / "git_commit.txt",
            f"commit={git_evidence['commit']}\n"
            f"dirty={str(git_evidence['dirty']).lower()}\n",
        )
        target_data_evidence = _snapshot_target_evidence(
            run_directory,
            target_data_evidence,
        )
        manifest["target_data_evidence"] = target_data_evidence
        _write_json(manifest_path, manifest, replace=True)
        _write_json(
            run_directory / "source_checkpoint_reference.json",
            source_reference,
        )
        _write_json(run_directory / "params.json", params)
        _write_json(run_directory / "environment.json", environment)

        with (run_directory / "training_log.txt").open(
            "x", encoding="utf-8"
        ) as training_log, redirect_stdout(training_log):
            with tf.device(contract["device"]):
                source_model = load_model(source["checkpoint_path"], compile=False)
            model_gate = _build_and_validate_freeze_candidate(
                source_model,
                K,
                tf,
                contract=contract,
                architecture_directory=str(run_directory),
                savefig=True,
            )
            fresh_model = model_gate["fresh_model"]
            model = model_gate["model"]
            callbacks = make_transfer_learning_freeze_formal_callbacks(
                run_directory,
                contract=contract,
            )
            callback_snapshot = _callback_snapshot(callbacks)
            _validate_callback_snapshot(
                callback_snapshot,
                run_directory,
                contract=contract,
            )

            with tf.device(contract["device"]):
                history = model.fit(
                    training["X_train"],
                    training["y_train"],
                    validation_data=(
                        training["X_validation"],
                        training["y_validation"],
                    ),
                    batch_size=contract["freeze"]["batch_size"],
                    epochs=contract["maximum_epochs"],
                    shuffle=contract["training_shuffle"],
                    callbacks=list(callbacks),
                    verbose=verbose,
                )

        frozen_weights_changed = any(
            not np.array_equal(before, after)
            for index in contract["transfer_layer_indices"]
            for before, after in zip(
                model_gate["frozen_weights_before_fit"][index],
                model.layers[index].get_weights(),
            )
        )
        if frozen_weights_changed:
            raise RuntimeError("A frozen transferred tensor changed during formal TL")
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Formal TL best Validation checkpoint missing")
        _save_learning_curve(history, run_directory / "learning_curve.png")

        best_model = load_model(
            checkpoint_path,
            custom_objects={"rmse": rmse},
            compile=False,
        )
        evaluation = load_target_evaluation_contract(
            corrected_r1_root=corrected_r1_root,
            experiment=experiment,
        )
        with tf.device(contract["device"]):
            prediction_normalized = best_model.predict(
                evaluation["X_test"],
                batch_size=contract["freeze"]["batch_size"],
                verbose=0,
            ).reshape(-1)
        prediction_original = evaluation["target_scaler"].inverse_transform(
            prediction_normalized.reshape(-1, 1)
        ).reshape(-1)
        _write_predictions(
            run_directory / "predictions_normalized.csv",
            evaluation["target_timestamps"],
            evaluation["target_indices"],
            evaluation["y_test_normalized"],
            prediction_normalized,
            "normalized",
        )
        _write_predictions(
            run_directory / "predictions_original_scale.csv",
            evaluation["target_timestamps"],
            evaluation["target_indices"],
            evaluation["y_test_original"],
            prediction_original,
            "original_scale",
        )
        metrics_normalized = _regression_metrics(
            evaluation["y_test_normalized"],
            prediction_normalized,
        )
        metrics_original = _regression_metrics(
            evaluation["y_test_original"],
            prediction_original,
        )
        _write_json(run_directory / "metrics_normalized.json", metrics_normalized)
        _write_json(
            run_directory / "metrics_original_scale.json",
            metrics_original,
        )
        _save_original_scale_diagnostics(
            run_directory,
            evaluation["target_timestamps"],
            evaluation["y_test_original"],
            prediction_original,
        )

        source_modified = not (
            _sha256_file(source["checkpoint_path"])
            == source["checkpoint_sha256_before"]
            and _file_state(source["run_directory"])
            == source["artifact_state_before"]
        )
        baseline_modified = (
            False
            if baseline_state_before is None
            else _file_state(baseline["evidence_manifest"].parent)
            != baseline_state_before
        )
        target_evidence_modified = any(
            _sha256_file(target_data_evidence["artifacts"][name]["source_path"])
            != digest
            for name, digest in target_evidence_state_before.items()
        )
        if source_modified or baseline_modified or target_evidence_modified:
            raise RuntimeError("Formal TL changed protected source evidence")

        missing = [
            name
            for name in _FORMAL_TL_FREEZE_OUTPUTS
            if not (run_directory / name).is_file()
            or (run_directory / name).stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(f"Formal TL artifact package incomplete: {missing}")
        validation_losses = np.asarray(history.history["val_loss"], dtype=np.float64)
        best_epoch_zero_based = int(np.argmin(validation_losses))
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "executed_epochs": len(history.epoch),
                "best_epoch_zero_based": best_epoch_zero_based,
                "best_epoch": best_epoch_zero_based + 1,
                "best_val_loss": float(validation_losses[best_epoch_zero_based]),
                "final_learning_rate": float(
                    K.get_value(model.optimizer.learning_rate)
                ),
                "best_checkpoint": str(checkpoint_path),
                "frozen_transferred_weights_changed": frozen_weights_changed,
                "target_test_used_for_selection": False,
                "target_test_metrics_role": "checkpoint-diagnostic-only",
                "prediction_sample_count": len(prediction_normalized),
                "evidence_package_complete": True,
                "artifact_count": len(_FORMAL_TL_FREEZE_OUTPUTS),
            }
        )
        _write_json(manifest_path, manifest, replace=True)
        K.clear_session()
        return {
            "run_directory": run_directory,
            "executed_epochs": len(history.epoch),
            "best_checkpoint": checkpoint_path,
            "metrics_normalized": metrics_normalized,
            "metrics_original_scale": metrics_original,
            "target_test_used_for_selection": False,
        }
    except Exception as error:
        try:
            _mark_transfer_learning_formal_failed(manifest_path, manifest, error)
        except Exception as manifest_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"Additionally failed to save failure manifest: {manifest_error}"
                )
        try:
            K.clear_session()
        except Exception as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"Additionally failed to clear Keras session: {cleanup_error}"
                )
        raise


def run_transfer_learning_unfreeze_formal(
    *,
    corrected_r1_root=None,
    experiment="A",
    run_id=None,
    output_root=None,
    maximum_epochs=500,
    allow_overwrite=False,
    verbose=1,
):
    """Execute locked formal Unfreeze only after explicit future authorization.

    The model is initialized directly from the mapped sealed Source checkpoint.
    The mapped sealed Freeze run is protected comparison evidence only and is
    never an initialization source.  Target Test is loaded only after the best
    Validation checkpoint is fixed.
    """

    contract = resolve_r2_transfer_learning_contract(experiment)
    direction = _transfer_learning_direction(experiment)
    expected_run_id = contract["planned_run_ids"]["unfreeze"]
    run_id = expected_run_id if run_id is None else run_id
    requested = {
        "experiment": experiment,
        "run_id": run_id,
        "maximum_epochs": maximum_epochs,
        "allow_overwrite": allow_overwrite,
    }
    expected = {
        "experiment": contract["experiment"],
        "run_id": expected_run_id,
        "maximum_epochs": contract["maximum_epochs"],
        "allow_overwrite": contract["allow_overwrite"],
    }
    for key, value in requested.items():
        if value != expected[key]:
            raise ValueError(
                f"Formal TL Unfreeze requires {key}={expected[key]!r}, got {value!r}"
            )

    run_directory = resolve_transfer_learning_unfreeze_formal_run_directory(
        run_id=run_id,
        output_root=output_root,
    )
    assert_transfer_learning_unfreeze_formal_run_available(
        run_directory,
        allow_overwrite=allow_overwrite,
    )
    training = load_target_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    _validate_formal_target_training_data(training)
    scaler = _scaler_isolation(
        corrected_r1_root,
        training,
        experiment=experiment,
    )
    baseline = _without_tl_anchor(
        training,
        scaler,
        experiment=experiment,
    )
    source = _source_checkpoint_gate(contract)
    if source["checkpoint_sha256_before"] != contract["source_checkpoint_sha256"]:
        raise ValueError("Formal Unfreeze Source checkpoint SHA256 mismatch")
    freeze = _sealed_freeze_run_gate(contract)

    keras, K, tf = _configure_tl_cpu_runtime(contract["seed"])
    from keras.models import load_model
    from utils.model import rmse

    git_evidence = _git_evidence()
    core_code_evidence = _formal_tl_core_code_evidence()
    target_data_evidence = _formal_target_data_evidence(
        corrected_r1_root,
        experiment,
    )
    target_evidence_state_before = {
        name: _sha256_file(record["source_path"])
        for name, record in target_data_evidence["artifacts"].items()
    }
    environment = _environment_evidence(keras, tf, contract=contract)
    source_reference = {
        "source_run_id": source["run_id"],
        "source_checkpoint_path": str(source["checkpoint_path"]),
        "source_checkpoint_sha256": source["checkpoint_sha256_before"],
        "source_final_learning_rate": source["final_learning_rate"],
        "source_optimizer_transferred": False,
        "used_for_initialization": True,
    }
    freeze_reference = {
        "run_id": freeze["run_id"],
        "status": "complete",
        "sealed": True,
        "artifact_count": len(_FORMAL_TL_FREEZE_OUTPUTS),
        "role": "protected-comparison-only",
        "used_for_initialization": False,
        "optimizer_transferred": False,
    }
    comparison_reference = {
        "canonical_run_id": baseline["canonical_run_id"],
        "actual_evidence_run_id": baseline["evidence_run_id"],
        "actual_evidence_manifest": (
            None
            if baseline["evidence_manifest"] is None
            else str(baseline["evidence_manifest"])
        ),
        "canonical_alias_directory_exists": baseline["canonical_directory_exists"],
        "status": baseline["evidence_status"],
        "note": baseline["note"],
    }
    baseline_state_before = (
        None
        if baseline["evidence_manifest"] is None
        else _file_state(baseline["evidence_manifest"].parent)
    )
    corrected_root = Path(
        CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root
    ).resolve()
    corrected_state_before = _file_state(corrected_root)
    experiment_a_roots = (
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["source_run_id"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["freeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_SOURCE_FORMAL_OUTPUT_ROOT
            / R2_EXPERIMENT_A_TL_CONTRACT["planned_run_ids"]["unfreeze"]
        ).resolve(),
        (
            _REPOSITORY_ROOT
            / R2_EXPERIMENT_A_WITHOUT_TL_ANCHOR["evidence_manifest"]
        ).resolve().parent,
    )
    experiment_a_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in experiment_a_roots
    }
    legacy_roots = (
        _REPOSITORY_ROOT / "dataset",
        _REPOSITORY_ROOT / "preprocess",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 A：Plant1 → Plant2",
        _REPOSITORY_ROOT
        / "reports"
        / "Solar Energy Result"
        / "實驗 B：Plant2 → Plant1",
    )
    legacy_state_before = {
        str(path): _file_state(path) if path.is_dir() else None
        for path in legacy_roots
    }

    started_at = datetime.now(timezone.utc).isoformat()
    checkpoint_path = run_directory / "checkpoints" / "best_model.hdf5"
    manifest_path = run_directory / "run_manifest.json"
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at,
        "run_directory": str(run_directory),
        "allow_overwrite": False,
        "experiment": experiment,
        "direction": direction,
        "mode": "TL Unfreeze",
        "source": contract["source"],
        "target": contract["target"],
        "git": git_evidence,
        "core_code_sha256": core_code_evidence,
        "source_checkpoint_reference": source_reference,
        "freeze_reference": freeze_reference,
        "target_data_evidence": target_data_evidence,
        "without_tl_comparison_reference": comparison_reference,
        "selection_split": "validation",
        "target_test_used_for_selection": False,
        "target_test_evaluation_stage": "after-best-checkpoint-fixed",
        "metric_units": R2_EXPERIMENT_A_TL_FREEZE_OUTPUT_UNITS,
        "planned_outputs": list(_FORMAL_TL_UNFREEZE_OUTPUTS),
    }
    params = {
        **{key: value for key, value in contract.items() if key != "callbacks"},
        "run_id": run_id,
        "mode": "TL Unfreeze",
        "batch_size": contract["unfreeze"]["batch_size"],
        "trainable_params": contract["unfreeze"]["trainable_params"],
        "non_trainable_params": contract["unfreeze"]["non_trainable_params"],
        "callbacks": contract["callbacks"],
        "training_sequences": len(training["X_train"]),
        "validation_sequences": len(training["X_validation"]),
        "test_sequences": training["test_sequence_count"],
        "metric_units": R2_EXPERIMENT_A_TL_FREEZE_OUTPUT_UNITS,
        "unfreeze_initialized_from_source": True,
        "unfreeze_initialized_from_freeze": False,
    }

    source_model = fresh_model = model = best_model = None
    run_directory.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(manifest_path, manifest)
        (run_directory / "checkpoints").mkdir(exist_ok=False)
        _write_text_exclusive(
            run_directory / "git_commit.txt",
            f"commit={git_evidence['commit']}\n"
            f"dirty={str(git_evidence['dirty']).lower()}\n",
        )
        target_data_evidence = _snapshot_target_evidence(
            run_directory,
            target_data_evidence,
        )
        manifest["target_data_evidence"] = target_data_evidence
        _write_json(manifest_path, manifest, replace=True)
        _write_json(
            run_directory / "source_checkpoint_reference.json",
            source_reference,
        )
        _write_json(run_directory / "params.json", params)
        _write_json(run_directory / "environment.json", environment)

        with (run_directory / "training_log.txt").open(
            "x", encoding="utf-8"
        ) as training_log, redirect_stdout(training_log):
            with tf.device(contract["device"]):
                source_model = load_model(source["checkpoint_path"], compile=False)
            model_gate = _build_and_validate_unfreeze_candidate(
                source_model,
                K,
                tf,
                contract=contract,
                architecture_directory=str(run_directory),
                savefig=True,
            )
            fresh_model = model_gate["fresh_model"]
            model = model_gate["model"]
            callbacks = make_transfer_learning_unfreeze_formal_callbacks(
                run_directory,
                contract=contract,
            )
            callback_snapshot = _callback_snapshot(callbacks)
            _validate_callback_snapshot(
                callback_snapshot,
                run_directory,
                contract=contract,
            )

            with tf.device(contract["device"]):
                history = model.fit(
                    training["X_train"],
                    training["y_train"],
                    validation_data=(
                        training["X_validation"],
                        training["y_validation"],
                    ),
                    batch_size=contract["unfreeze"]["batch_size"],
                    epochs=contract["maximum_epochs"],
                    shuffle=contract["training_shuffle"],
                    callbacks=list(callbacks),
                    verbose=verbose,
                )

        if not checkpoint_path.is_file():
            raise FileNotFoundError("Formal Unfreeze best Validation checkpoint missing")
        _save_learning_curve(history, run_directory / "learning_curve.png")

        best_model = load_model(
            checkpoint_path,
            custom_objects={"rmse": rmse},
            compile=False,
        )
        evaluation = load_target_evaluation_contract(
            corrected_r1_root=corrected_r1_root,
            experiment=experiment,
        )
        with tf.device(contract["device"]):
            prediction_normalized = best_model.predict(
                evaluation["X_test"],
                batch_size=contract["unfreeze"]["batch_size"],
                verbose=0,
            ).reshape(-1)
        prediction_original = evaluation["target_scaler"].inverse_transform(
            prediction_normalized.reshape(-1, 1)
        ).reshape(-1)
        _write_predictions(
            run_directory / "predictions_normalized.csv",
            evaluation["target_timestamps"],
            evaluation["target_indices"],
            evaluation["y_test_normalized"],
            prediction_normalized,
            "normalized",
        )
        _write_predictions(
            run_directory / "predictions_original_scale.csv",
            evaluation["target_timestamps"],
            evaluation["target_indices"],
            evaluation["y_test_original"],
            prediction_original,
            "original_scale",
        )
        metrics_normalized = _regression_metrics(
            evaluation["y_test_normalized"],
            prediction_normalized,
        )
        metrics_original = _regression_metrics(
            evaluation["y_test_original"],
            prediction_original,
        )
        _write_json(run_directory / "metrics_normalized.json", metrics_normalized)
        _write_json(
            run_directory / "metrics_original_scale.json",
            metrics_original,
        )
        _save_original_scale_diagnostics(
            run_directory,
            evaluation["target_timestamps"],
            evaluation["y_test_original"],
            prediction_original,
        )

        source_modified = not (
            _sha256_file(source["checkpoint_path"])
            == source["checkpoint_sha256_before"]
            and _file_state(source["run_directory"])
            == source["artifact_state_before"]
        )
        freeze_modified = (
            _file_state(freeze["run_directory"])
            != freeze["artifact_state_before"]
        )
        baseline_modified = (
            False
            if baseline_state_before is None
            else _file_state(baseline["evidence_manifest"].parent)
            != baseline_state_before
        )
        experiment_a_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != experiment_a_state_before[str(path)]
            for path in experiment_a_roots
        )
        corrected_r1_evidence_modified = (
            _file_state(corrected_root) != corrected_state_before
        )
        target_evidence_modified = any(
            _sha256_file(target_data_evidence["artifacts"][name]["source_path"])
            != digest
            for name, digest in target_evidence_state_before.items()
        )
        legacy_modified = any(
            (_file_state(path) if path.is_dir() else None)
            != legacy_state_before[str(path)]
            for path in legacy_roots
        )
        if any(
            (
                source_modified,
                freeze_modified,
                baseline_modified,
                experiment_a_modified,
                corrected_r1_evidence_modified,
                target_evidence_modified,
                legacy_modified,
            )
        ):
            raise RuntimeError("Formal Unfreeze changed a protected artifact")

        missing = [
            name
            for name in _FORMAL_TL_UNFREEZE_OUTPUTS
            if not (run_directory / name).is_file()
            or (run_directory / name).stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                f"Formal Unfreeze artifact package incomplete: {missing}"
            )
        validation_losses = np.asarray(history.history["val_loss"], dtype=np.float64)
        best_epoch_zero_based = int(np.argmin(validation_losses))
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "executed_epochs": len(history.epoch),
                "best_epoch_zero_based": best_epoch_zero_based,
                "best_epoch": best_epoch_zero_based + 1,
                "best_val_loss": float(validation_losses[best_epoch_zero_based]),
                "final_learning_rate": float(
                    K.get_value(model.optimizer.learning_rate)
                ),
                "best_checkpoint": str(checkpoint_path),
                "unfreeze_initialized_from_source": True,
                "unfreeze_initialized_from_freeze": False,
                "transferred_initial_weights_exact": model_gate[
                    "transferred_initial_weights_exact"
                ],
                "transferred_layers_trainable": model_gate[
                    "transferred_layers_trainable"
                ],
                "target_test_used_for_selection": False,
                "target_test_metrics_role": "checkpoint-diagnostic-only",
                "prediction_sample_count": len(prediction_normalized),
                "evidence_package_complete": True,
                "artifact_count": len(
                    _FORMAL_TL_UNFREEZE_OUTPUTS
                ),
            }
        )
        _write_json(manifest_path, manifest, replace=True)
        K.clear_session()
        return {
            "run_directory": run_directory,
            "executed_epochs": len(history.epoch),
            "best_checkpoint": checkpoint_path,
            "metrics_normalized": metrics_normalized,
            "metrics_original_scale": metrics_original,
            "target_test_used_for_selection": False,
        }
    except Exception as error:
        try:
            _mark_transfer_learning_formal_failed(manifest_path, manifest, error)
        except Exception as manifest_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"Additionally failed to save failure manifest: {manifest_error}"
                )
        try:
            K.clear_session()
        except Exception as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"Additionally failed to clear Keras session: {cleanup_error}"
                )
        raise


def format_transfer_learning_dry_run(result):
    """Format the shared C0/D4 audit as a stable A–K readout."""

    if result.get("phase") == "D4":
        target = result["target"]
        scaler = result["scaler"]
        freeze = result["freeze"]
        unfreeze = result["unfreeze"]
        optimizer = result["optimizer_isolation"]
        baseline = result["baseline"]
        historical = result["historical"]
        safety = result["safety"]
        lines = [
            f"Phase D4 = {result['status']}",
            f"Experiment = {result['experiment']}",
            f"Direction = {result['direction']}",
            "A. Source checkpoint",
            f"Run ID = {result['source']['run_id']}",
            f"Status = {result['source']['status']}",
            f"Checkpoint load = {result['source']['checkpoint_load']}",
            f"Checkpoint SHA256 = {result['source']['checkpoint_sha256']}",
            f"Input shape = {result['source']['input_shape']}",
            f"Params = {result['source']['total_params']}",
            f"Output = {result['source']['output_activation']}",
            f"Source final LR = {result['source']['final_learning_rate']:.0e}",
            f"Modified = {result['source']['modified']}",
            "B. Target contract",
            "Rows = "
            + " / ".join(
                str(target["rows"][name])
                for name in ("training", "validation", "test")
            ),
            "Sequences = "
            + " / ".join(
                str(target["sequences"][name])
                for name in ("training", "validation", "test")
            ),
            f"Alignment = {target['alignment']}",
            f"Target Test used for training = {target['target_test_used_for_training']}",
            f"Target Test used for validation = {target['target_test_used_for_validation']}",
            f"Target Test used for callbacks = {target['target_test_used_for_callbacks']}",
            f"Target Test used for selection = {target['target_test_used_for_selection']}",
            f"Target Test metrics computed = {target['target_test_metrics_computed']}",
            "C. Scaler isolation",
            f"Source scaler used on Target = {scaler['source_scaler_used_on_target']}",
            f"Target feature scaler fit rows = {scaler['target_feature_scaler_fit_rows']}",
            f"Target target scaler fit rows = {scaler['target_target_scaler_fit_rows']}",
            f"Target scaler identity = {scaler['target_scaler_identity']}",
            "Target target-scaler round-trip = "
            f"{scaler['target_target_scaler_round_trip']}",
            "D. Weight transfer table",
            "Index | Source Layer | Source Class | Target Layer | Target Class | "
            "Params | Weight Shapes | Transferred | Exact Match | Freeze Trainable | "
            "Unfreeze Trainable",
        ]
        for layer in result["layers"]:
            lines.append(
                f"{layer['layer_index']} | {layer['source_layer_name']} | "
                f"{layer['source_layer_class']} | {layer['target_layer_name']} | "
                f"{layer['target_layer_class']} | {layer['parameter_count']} | "
                f"{layer['weight_shapes']} | {layer['transfer_expected']} | "
                f"{layer['weight_equal_after_transfer']} | "
                f"{layer['freeze_trainable']} | {layer['unfreeze_trainable']}"
            )
        lines.extend(
            (
                "E. Non-transferred",
                "Input projection transferred = "
                f"{result['non_transferred']['input_projection_transferred']}",
                "Output transferred = "
                f"{result['non_transferred']['output_layer_transferred']}",
                "Fresh initialization preserved = "
                f"{result['non_transferred']['input_projection_fresh_initialization_preserved'] and result['non_transferred']['output_fresh_initialization_preserved']}",
                "F. Freeze",
                f"LR = {freeze['initial_learning_rate']:.0e}",
                f"Batch = {freeze['batch_size']}",
                f"Trainable params = {freeze['trainable_params']}",
                f"Non-trainable params = {freeze['non_trainable_params']}",
                "G. Unfreeze",
                f"LR = {unfreeze['initial_learning_rate']:.0e}",
                f"Batch = {unfreeze['batch_size']}",
                f"Trainable params = {unfreeze['trainable_params']}",
                f"Non-trainable params = {unfreeze['non_trainable_params']}",
                "H. Optimizer isolation",
                f"Source final LR = {optimizer['source_final_learning_rate']:.0e}",
                f"Target Freeze LR = {optimizer['freeze_initial_learning_rate']:.0e}",
                f"Target Unfreeze LR = {optimizer['unfreeze_initial_learning_rate']:.0e}",
                f"Target new Adam = {optimizer['target_new_adam']}",
                "Source optimizer transferred = "
                f"{optimizer['source_optimizer_state_transferred']}",
                f"Target optimizer fresh = {optimizer['target_optimizer_fresh']}",
                "I. Plant1 Without-TL anchor",
                f"Canonical run ID = {baseline['canonical_run_id']}",
                f"Status = {baseline['evidence_status']}",
                f"Canonical directory exists = {baseline['canonical_directory_exists']}",
                "J. Historical contract",
                f"Maximum epochs = {historical['maximum_epochs']}",
                f"Callbacks = {tuple(historical['callbacks'])}",
                f"Training shuffle = {historical['training_shuffle']}",
                f"Validation shuffle = {historical['validation_shuffle']}",
                "K. Safety",
                f"Epoch executed = {safety['epoch_executed']}",
                f"fit = {safety['fit_calls']}",
                f"fit_generator = {safety['fit_generator_calls']}",
                f"train_on_batch = {safety['train_on_batch_calls']}",
                f"predict = {safety['predict_calls']}",
                f"predict_generator = {safety['predict_generator_calls']}",
                f"evaluate = {safety['evaluate_calls']}",
                f"Checkpoint created = {safety['checkpoint_created']}",
                f"Formal TL run created = {safety['formal_tl_run_created']}",
            )
        )
        for mode, directory in result["planned_run_directories"].items():
            lines.append(f"Planned {mode} directory exists = {directory.exists()}")
        for note in result["notes"]:
            lines.append(f"NOTE = {note}")
        return "\n".join(lines)

    target = result["target"]
    scaler = result["scaler"]
    freeze = result["freeze"]
    unfreeze = result["unfreeze"]
    optimizer = result["optimizer_isolation"]
    baseline = result["baseline"]
    historical = result["historical"]
    safety = result["safety"]
    lines = [
        f"Phase C0 = {result['status']}",
        "A. Source checkpoint",
        f"Run ID = {result['source']['run_id']}",
        f"Status = {result['source']['status']}",
        f"Checkpoint load = {result['source']['checkpoint_load']}",
        f"Input shape = {result['source']['input_shape']}",
        f"Params = {result['source']['total_params']}",
        f"Source final LR = {result['source']['final_learning_rate']:.0e}",
        f"Modified = {result['source']['modified']}",
        "B. Target data",
        "Rows = " + " / ".join(str(target["rows"][name]) for name in ("training", "validation", "test")),
        "Sequences = " + " / ".join(str(target["sequences"][name]) for name in ("training", "validation", "test")),
        f"Alignment = {target['alignment_ok']}",
        "Validation duplicate/padding = "
        f"{target['validation_duplicate_count'] + target['validation_padding_count']}",
        f"Target Test used for training = {target['target_test_used_for_training']}",
        f"Target Test used for validation = {target['target_test_used_for_validation']}",
        f"Target Test used for callbacks = {target['target_test_used_for_callbacks']}",
        f"Target Test used for selection = {target['target_test_used_for_selection']}",
        f"Target Test metrics computed = {target['target_test_metrics_computed']}",
        "C. Scaler isolation",
        f"Source scaler used on Target = {scaler['source_scaler_used_on_target']}",
        f"Target feature scaler fit rows = {scaler['target_feature_scaler_fit_rows']}",
        f"Target target scaler fit rows = {scaler['target_target_scaler_fit_rows']}",
        f"Target scaler identity = {scaler['target_scaler_identity']}",
        f"Baseline scaler compatibility = {baseline['scaler_compatible']}",
        "D. Weight transfer table",
        "Layer | Class | Params | Transferred? | Weights exact match? | Freeze trainable? | Unfreeze trainable?",
    ]
    for layer in result["layers"]:
        lines.append(
            f"{layer['layer_index']}:{layer['source_layer_name']} | "
            f"{layer['source_layer_class']} | {layer['parameter_count']} | "
            f"{layer['transfer_expected']} | {layer['weight_equal_after_transfer']} | "
            f"{layer['freeze_trainable']} | {layer['unfreeze_trainable']}"
        )
    lines.extend(
        (
            "E. Non-transferred layers",
            "Input projection transferred = "
            f"{result['non_transferred']['input_projection_transferred']}",
            f"Output transferred = {result['non_transferred']['output_layer_transferred']}",
            "Target fresh initialization preserved = "
            f"{result['non_transferred']['input_projection_fresh_initialization_preserved'] and result['non_transferred']['output_fresh_initialization_preserved']}",
            "F. Freeze contract",
            f"LR = {freeze['initial_learning_rate']:.0e}",
            f"Batch = {freeze['batch_size']}",
            f"Trainable params = {freeze['trainable_params']}",
            f"Non-trainable params = {freeze['non_trainable_params']}",
            f"Trainable layer names = {freeze['trainable_layer_names']}",
            "G. Unfreeze contract",
            f"LR = {unfreeze['initial_learning_rate']:.0e}",
            f"Batch = {unfreeze['batch_size']}",
            f"Trainable params = {unfreeze['trainable_params']}",
            f"Non-trainable params = {unfreeze['non_trainable_params']}",
            f"Trainable layer names = {unfreeze['trainable_layer_names']}",
            "H. Optimizer isolation",
            f"Source final LR = {optimizer['source_final_learning_rate']:.0e}",
            f"Target new Adam = {optimizer['target_new_adam']}",
            f"Target initial LR = {optimizer['freeze_initial_learning_rate']:.0e}",
            "Source optimizer state transferred = "
            f"{optimizer['source_optimizer_state_transferred']}",
            f"Target optimizer state initialized fresh = {optimizer['target_optimizer_fresh']}",
            "I. Future Without-TL comparison anchor",
            f"Canonical run ID = {baseline['canonical_run_id']}",
            f"Evidence run ID = {baseline['evidence_run_id']}",
            f"Sealed baseline = {baseline['compatible']}",
            f"Target profile = {baseline['target_profile']}",
            "Test interval = "
            f"{baseline['test_first_target_timestamp']} / {baseline['test_last_target_timestamp']}",
            f"Scaler compatibility = {baseline['scaler_compatible']}",
            "J. Historical contract",
            f"Maximum epochs = {historical['maximum_epochs']}",
            f"Callbacks = {tuple(historical['callbacks'])}",
            f"Training shuffle = {historical['training_shuffle']}",
            f"Validation shuffle = {historical['validation_shuffle']}",
            "Requires human confirmation = "
            f"{historical['requires_human_confirmation']}",
            "K. Safety",
            f"Epoch executed = {safety['epoch_executed']}",
            f"fit = {safety['fit_calls']}",
            f"fit_generator = {safety['fit_generator_calls']}",
            f"predict = {safety['predict_calls']}",
            f"evaluate = {safety['evaluate_calls']}",
            f"Checkpoint created = {safety['checkpoint_created']}",
            f"Formal TL run created = {safety['formal_tl_run_created']}",
            "Legacy/sealed artifacts modified = "
            f"{safety['source_modified'] or safety['baseline_modified']}",
        )
    )
    for note in result["notes"]:
        lines.append(f"NOTE = {note}")
    return "\n".join(lines)


def format_transfer_learning_freeze_smoke(result):
    """Format the in-memory Phase C1/D5 Freeze smoke result."""

    return "\n".join(
        (
            f"Phase {result['phase']} = PASS",
            f"Experiment = {result['experiment']}",
            f"Direction = {result['direction']}",
            f"Mode = {result['mode']}",
            f"Training sequences = {result['training_sequences']}",
            f"Validation sequences = {result['validation_sequences']}",
            f"Test sequences = {result['test_sequences']}",
            f"Batch size = {result['batch_size']}",
            f"Epochs requested = {result['epochs_requested']}",
            f"Executed epochs = {result['executed_epochs']}",
            f"Device = {result['device']}",
            f"Model params = {result['model_params']}",
            f"Trainable params = {result['trainable_params']}",
            f"Non-trainable params = {result['non_trainable_params']}",
            f"Output = {result['output_activation']}",
            f"Optimizer = {result['optimizer']}",
            f"Initial LR = {result['initial_learning_rate']:.8g}",
            f"Loss = {result['loss']}",
            f"Training loss = {result['train_loss']:.10g}",
            f"Validation loss = {result['validation_loss']:.10g}",
            f"Train loss finite = {result['train_loss_finite']}",
            f"Validation loss finite = {result['validation_loss_finite']}",
            "Transferred initial weights exact = "
            f"{result['transferred_initial_weights_exact']}",
            "Transferred layers frozen = "
            f"{result['transferred_layers_frozen']}",
            "Frozen transferred weights changed = "
            f"{result['frozen_transferred_weights_changed']}",
            "Optimizer iteration before = "
            f"{result['optimizer_iteration_before']}",
            "Optimizer iteration after = "
            f"{result['optimizer_iteration_after']}",
            f"Source final LR = {result['source_final_learning_rate']:.8g}",
            "Source optimizer transferred = "
            f"{result['source_optimizer_state_transferred']}",
            f"Target Test used in fit = {result['target_test_used_in_fit']}",
            "Target Test used in validation = "
            f"{result['target_test_used_in_validation']}",
            "Target Test used in callbacks = "
            f"{result['target_test_used_in_callbacks']}",
            "Target Test used for selection = "
            f"{result['target_test_used_for_selection']}",
            "Target Test metrics computed = "
            f"{result['target_test_metrics_computed']}",
            f"Checkpoint created = {result['checkpoint_created']}",
            f"Formal TL run created = {result['formal_tl_run_created']}",
            "Experiment B formal execution exposed = "
            f"{result['experiment_b_formal_execution_exposed']}",
            f"Source checkpoint modified = {result['source_checkpoint_modified']}",
            f"Baseline modified = {result['baseline_modified']}",
            f"Experiment A modified = {result['experiment_a_modified']}",
            "Corrected R1 evidence modified = "
            f"{result['corrected_r1_evidence_modified']}",
            f"Legacy modified = {result['legacy_modified']}",
            f"Outputs written = {result['outputs_written']}",
        )
    )


def format_transfer_learning_unfreeze_smoke(result):
    """Format the shared in-memory Phase C4/D8 Unfreeze smoke result."""

    return "\n".join(
        (
            f"Phase {result['phase']} = PASS",
            f"Experiment = {result['experiment']}",
            f"Direction = {result['direction']}",
            f"Mode = {result['mode']}",
            f"Source run ID = {result['source_run_id']}",
            f"Freeze protected run ID = {result['freeze_run_id']}",
            f"Training sequences = {result['training_sequences']}",
            f"Validation sequences = {result['validation_sequences']}",
            f"Test sequences = {result['test_sequences']}",
            f"Batch size = {result['batch_size']}",
            f"Epochs requested = {result['epochs_requested']}",
            f"Executed epochs = {result['executed_epochs']}",
            f"Device = {result['device']}",
            f"Model params = {result['model_params']}",
            f"Trainable params = {result['trainable_params']}",
            f"Non-trainable params = {result['non_trainable_params']}",
            f"Output = {result['output_activation']}",
            f"Optimizer = {result['optimizer']}",
            f"Initial LR = {result['initial_learning_rate']:.8g}",
            f"Loss = {result['loss']}",
            f"Training loss = {result['train_loss']:.10g}",
            f"Validation loss = {result['validation_loss']:.10g}",
            f"Train loss finite = {result['train_loss_finite']}",
            f"Validation loss finite = {result['validation_loss_finite']}",
            "Transferred initial weights exact = "
            f"{result['transferred_initial_weights_exact']}",
            "Transferred layers trainable = "
            f"{result['transferred_layers_trainable']}",
            "Any transferred trainable tensor changed = "
            f"{result['any_transferred_trainable_tensor_changed']}",
            "Changed transferred trainable tensors = "
            f"{result['changed_trainable_tensor_count']}",
            "Changed BN moving tensors = "
            f"{result['changed_bn_moving_tensor_count']}",
            "Optimizer iteration before = "
            f"{result['optimizer_iteration_before']}",
            "Optimizer iteration after = "
            f"{result['optimizer_iteration_after']}",
            "Unfreeze initialized from Source = "
            f"{result['unfreeze_initialized_from_source']}",
            "Unfreeze initialized from Freeze = "
            f"{result['unfreeze_initialized_from_freeze']}",
            "Source optimizer transferred = "
            f"{result['source_optimizer_state_transferred']}",
            "Freeze optimizer transferred = "
            f"{result['freeze_optimizer_state_transferred']}",
            f"Training shuffle = {result['training_shuffle']}",
            f"Validation shuffle = {result['validation_shuffle']}",
            f"Target Test used in fit = {result['target_test_used_in_fit']}",
            "Target Test used in validation = "
            f"{result['target_test_used_in_validation']}",
            "Target Test used in callbacks = "
            f"{result['target_test_used_in_callbacks']}",
            "Target Test used for selection = "
            f"{result['target_test_used_for_selection']}",
            "Target Test metrics computed = "
            f"{result['target_test_metrics_computed']}",
            f"Checkpoint created = {result['checkpoint_created']}",
            "Formal Unfreeze run created = "
            f"{result['formal_unfreeze_run_created']}",
            "Experiment B formal execution exposed = "
            f"{result['experiment_b_formal_execution_exposed']}",
            "Experiment A contract unchanged = "
            f"{result['experiment_a_contract_unchanged']}",
            "Without-TL anchor status = "
            f"{result['without_tl_anchor_status']}",
            f"Source modified = {result['source_modified']}",
            f"Freeze modified = {result['freeze_modified']}",
            f"Baseline modified = {result['baseline_modified']}",
            f"Experiment A modified = {result['experiment_a_modified']}",
            "Corrected R1 evidence modified = "
            f"{result['corrected_r1_evidence_modified']}",
            f"Legacy modified = {result['legacy_modified']}",
            f"Outputs written = {result['outputs_written']}",
        )
    )


def format_transfer_learning_freeze_formal_dry_run(result):
    """Format the shared Phase C2/D6 zero-epoch formal Freeze audit."""

    rows = result["target_rows"]
    sequences = result["target_sequences"]
    model = result["model"]
    lines = [
        f"Phase {result['phase']} = {result['status']}",
        f"Experiment = {result['experiment']}",
        f"Direction = {result['direction']}",
        f"Mode = {result['mode']}",
        f"Run ID = {result['run_id']}",
        f"Formal runner ready = {result['formal_runner_ready']}",
        f"Source run ID = {result['source']['source_run_id']}",
        "Source checkpoint reference = "
        f"{result['source']['source_checkpoint_path']}",
        f"Source checkpoint load = {result['source']['checkpoint_load']}",
        "Source checkpoint SHA256 = "
        f"{result['source']['source_checkpoint_sha256']}",
        f"Source input shape = {result['source']['input_shape']}",
        f"Source model params = {result['source']['total_params']}",
        f"Source output = {result['source']['output_activation']}",
        f"Source final LR = {result['source']['source_final_learning_rate']:.0e}",
        "Source optimizer transferred = "
        f"{result['source']['source_optimizer_transferred']}",
        "Target rows = "
        + " / ".join(str(rows[name]) for name in ("training", "validation", "test")),
        "Target sequences = "
        + " / ".join(
            str(sequences[name]) for name in ("training", "validation", "test")
        ),
        "Target feature/target scaler fit rows = "
        f"{result['target_feature_scaler_fit_rows']} / "
        f"{result['target_target_scaler_fit_rows']}",
        "Target Validation/Test transform only = "
        f"{result['target_validation_transform_only']} / "
        f"{result['target_test_transform_only']}",
        f"Target clipping applied = {result['target_clipping_applied']}",
        f"Target scaler round-trip = {result['target_scaler_round_trip']}",
        f"Source scaler used on Target = {result['source_scaler_used_on_target']}",
        f"Model params = {model['total_params']}",
        f"Trainable params = {model['trainable_params']}",
        f"Non-trainable params = {model['non_trainable_params']}",
        f"Output = {model['output_activation']}",
        f"Optimizer = {model['optimizer']}",
        f"LR = {model['initial_learning_rate']:.8g}",
        f"Loss = {model['loss']}",
        f"Batch = {result['batch_size']}",
        f"Maximum epochs = {result['maximum_epochs']}",
        f"Training shuffle = {result['training_shuffle']}",
        f"Validation shuffle = {result['validation_shuffle']}",
        "Frozen transferred layer state = "
        f"{result['frozen_transferred_layer_state']}",
        "Transferred tensors exact = "
        f"{all(all(record['tensor_exact']) for record in result['transferred_layer_integrity'].values())}",
        "Non-transferred fresh initialization = "
        f"{result['non_transferred_fresh_initialization']}",
        f"Callback contract = {result['callbacks_pass']}",
        f"Callback files written = {result['callback_files_written']}",
        f"Planned artifact count = {len(result['planned_outputs'])}",
        "Planned artifacts = " + " / ".join(result["planned_outputs"]),
        f"Evidence package = {result['evidence_package']['ready']}",
        f"Failure manifest = {result['failure_manifest']}",
        f"Overwrite guard = {result['overwrite_guard']}",
        "Without-TL anchor status = "
        f"{result['without_tl_anchor_status']}",
        "Experiment B formal execution exposed = "
        f"{result['experiment_b_formal_execution_exposed']}",
        "Experiment A contract unchanged = "
        f"{result['experiment_a_contract_unchanged']}",
        f"Epoch executed = {result['epoch_executed']}",
        f"fit = {result['fit_calls']}",
        f"fit_generator = {result['fit_generator_calls']}",
        f"train_on_batch = {result['train_on_batch_calls']}",
        f"predict = {result['predict_calls']}",
        f"predict_generator = {result['predict_generator_calls']}",
        f"evaluate = {result['evaluate_calls']}",
        "Formal run directory exists = "
        f"{result['formal_run_directory_exists']}",
        f"Checkpoint created = {result['checkpoint_created']}",
        f"Metrics created = {result['metrics_created']}",
        f"Prediction created = {result['prediction_created']}",
        f"Formal run created = {result['formal_run_created']}",
        f"Source modified = {result['source_modified']}",
        f"Baseline modified = {result['baseline_modified']}",
        f"Experiment A modified = {result['experiment_a_modified']}",
        "Corrected R1 evidence modified = "
        f"{result['corrected_r1_evidence_modified']}",
        f"Legacy modified = {result['legacy_modified']}",
    ]
    for note in result["notes"]:
        lines.append(f"NOTE = {note}")
    return "\n".join(lines)


def format_transfer_learning_unfreeze_formal_dry_run(result):
    """Format the shared C5/D9 zero-epoch formal Unfreeze audit."""

    rows = result["target_rows"]
    sequences = result["target_sequences"]
    model = result["model"]
    lines = [
        f"Phase {result['phase']} = {result['status']}",
        f"Experiment = {result['experiment']}",
        f"Direction = {result['direction']}",
        f"Run ID = {result['run_id']}",
        f"Formal runner ready = {result['formal_runner_ready']}",
        f"Source checkpoint = {result['source']['source_checkpoint_path']}",
        "Source checkpoint load = "
        f"{'PASS' if result['source']['checkpoint_load'] else 'FAIL'}",
        "Source checkpoint SHA256 = "
        f"{result['source']['source_checkpoint_sha256'].upper()}",
        f"Source final LR = {result['source']['source_final_learning_rate']:.0e}",
        "Source optimizer transferred = "
        f"{result['source']['source_optimizer_transferred']}",
        "Unfreeze initialized from Source checkpoint = "
        f"{result['unfreeze_initialized_from_source']}",
        "Unfreeze initialized from Freeze checkpoint = "
        f"{result['unfreeze_initialized_from_freeze']}",
        f"Freeze sealed reference = {result['freeze_reference']['run_id']}",
        f"Freeze status = {result['freeze_reference']['status']}",
        f"Freeze role = {result['freeze_reference']['role']}",
        "Target rows = "
        + " / ".join(str(rows[name]) for name in ("training", "validation", "test")),
        "Target sequences = "
        + " / ".join(
            str(sequences[name]) for name in ("training", "validation", "test")
        ),
        f"Target scaler fit rows = {result['target_scaler_fit_rows']}",
        "Target feature/target scaler fit rows = "
        f"{result['target_feature_scaler_fit_rows']} / "
        f"{result['target_target_scaler_fit_rows']}",
        "Target validation/test transform only = "
        f"{result['target_validation_transform_only']} / "
        f"{result['target_test_transform_only']}",
        f"Target clipping applied = {result['target_clipping_applied']}",
        "Target scaler round-trip = "
        f"{'PASS' if result['target_scaler_round_trip'] else 'FAIL'}",
        f"Source scaler used on Target = {result['source_scaler_used_on_target']}",
        f"Model params = {model['total_params']}",
        f"Trainable params = {model['trainable_params']}",
        f"Non-trainable params = {model['non_trainable_params']}",
        f"Output = {model['output_activation']}",
        f"Optimizer = {model['optimizer']}",
        f"LR = {model['initial_learning_rate']:.8g}",
        f"Optimizer iteration = {model['optimizer_iteration']}",
        f"New optimizer = {model['optimizer_iteration'] == 0}",
        f"Loss = {model['loss'].upper()}",
        f"Batch = {result['batch_size']}",
        f"Maximum epochs = {result['maximum_epochs']}",
        f"Training shuffle = {result['training_shuffle']}",
        f"Validation shuffle = {result['validation_shuffle']}",
        "Transferred initial Source weights exact = "
        f"{result['transferred_initial_weights_exact']}",
        "Transferred layers trainable = "
        f"{result['transferred_layers_trainable']}",
        "Input projection transferred = "
        f"{result['input_projection_transferred']}",
        f"Output transferred = {result['output_transferred']}",
        "Source optimizer transferred = "
        f"{result['source_optimizer_transferred']}",
        "Freeze optimizer transferred = "
        f"{result['freeze_optimizer_transferred']}",
        f"Callback contract = {'PASS' if result['callbacks_pass'] else 'FAIL'}",
        f"Callback files written = {result['callback_files_written']}",
        f"Planned artifact count = {len(result['planned_outputs'])}",
        "Planned artifacts = " + " / ".join(result["planned_outputs"]),
        "Evidence package = "
        f"{'PASS' if result['evidence_package']['ready'] else 'FAIL'}",
        f"Failure manifest = {'PASS' if result['failure_manifest'] else 'FAIL'}",
        f"Overwrite guard = {'PASS' if result['overwrite_guard'] else 'FAIL'}",
        f"Without-TL anchor status = {result['without_tl_anchor_status']}",
        "Experiment B formal execution exposed = "
        f"{result['experiment_b_formal_execution_exposed']}",
        "Experiment A contract unchanged = "
        f"{result['experiment_a_contract_unchanged']}",
        f"Epoch executed = {result['epoch_executed']}",
        f"fit = {result['fit_calls']}",
        f"fit_generator = {result['fit_generator_calls']}",
        f"train_on_batch = {result['train_on_batch_calls']}",
        f"predict = {result['predict_calls']}",
        f"predict_generator = {result['predict_generator_calls']}",
        f"evaluate = {result['evaluate_calls']}",
        "Formal Unfreeze directory exists = "
        f"{result['formal_run_directory_exists']}",
        f"Checkpoint created = {result['checkpoint_created']}",
        f"Prediction created = {result['prediction_created']}",
        f"Metrics created = {result['metrics_created']}",
        f"Formal run created = {result['formal_run_created']}",
        f"Source modified = {result['source_modified']}",
        f"Freeze modified = {result['freeze_modified']}",
        f"Baseline modified = {result['baseline_modified']}",
        f"Experiment A modified = {result['experiment_a_modified']}",
        "Corrected R1 evidence modified = "
        f"{result['corrected_r1_evidence_modified']}",
        f"Target evidence modified = {result['target_evidence_modified']}",
        f"Legacy modified = {result['legacy_modified']}",
    ]
    for note in result["notes"]:
        lines.append(f"NOTE = {note}")
    return "\n".join(lines)
