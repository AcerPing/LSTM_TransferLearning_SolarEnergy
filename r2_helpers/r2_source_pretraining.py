"""Corrected R1 Source pre-training lifecycle helpers.

This module contains the read-only Phase A data preflight, the explicitly
one-epoch/in-memory Phase B1 smoke path, the Phase B2 Experiment-A formal
runner, and the Phase D1 guarded A/B contract selection.  Importing this
module or running either formal dry-run never calls ``fit`` and never creates
a formal run directory.
"""

import csv
import hashlib
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
    R2_SOURCE_DATA_USAGE,
    R2_SOURCE_FORMAL_CONTRACT,
    R2_SOURCE_FORMAL_EVALUATION_OUTPUTS,
    R2_SOURCE_FORMAL_OUTPUT_ROOT,
    R2_SOURCE_FORMAL_OUTPUTS,
    R2_SOURCE_FORMAL_RUN_ID,
    R2_SOURCE_MODEL_CONTRACT,
    corrected_r1_profile_path,
    resolve_r2_source_formal_contract,
    resolve_r2_source_smoke_contract,
)
from utils.data_io import (
    read_corrected_r1_profile,
    validate_corrected_r1_profile,
)


_FORMAL_CORE_EXECUTION_FILES = (
    "main.py",
    "utils/data_io.py",
    "utils/model.py",
    "r2_config/corrected_r1.py",
    "r2_helpers/r2_source_pretraining.py",
)
_FORMAL_SOURCE_EVIDENCE_FILES = (
    "split_manifest.json",
    "scaler_manifest.json",
    "feature_scaler.joblib",
    "target_scaler.joblib",
)


def load_source_training_contract(
    *,
    corrected_r1_root=None,
    experiment="A",
):
    """Return only future fit/validation arrays; never expose Test as fit data."""

    if experiment not in CORRECTED_R1_EXPERIMENTS:
        raise ValueError(f"Corrected R1 experiment must be A or B, got {experiment!r}")
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    source_mapping = CORRECTED_R1_EXPERIMENTS[experiment]["source"]
    source_dir = corrected_r1_profile_path(root, experiment, "source")
    profile = read_corrected_r1_profile(source_dir)
    report = validate_corrected_r1_profile(
        profile,
        expected_plant=source_mapping["manifest_plant"],
        expected_profile=source_mapping["manifest_profile"],
        expected_rows=CORRECTED_R1_EXPECTED_ROWS["source"],
        expected_sequences=CORRECTED_R1_EXPECTED_SEQUENCES["source"],
        feature_order=CORRECTED_R1_FEATURE_ORDER,
        target_column=CORRECTED_R1_TARGET,
        window=CORRECTED_R1_WINDOW,
        horizon=CORRECTED_R1_HORIZON,
    )
    training = report["sequence_data"]["training"]
    validation = report["sequence_data"]["validation"]
    return {
        "usage": "formal-source-training",
        "source_plant": source_mapping["manifest_plant"],
        "source_profile": source_mapping["profile_dir"],
        "source_rows": dict(report["rows"]),
        "source_sequences": dict(report["sequences"]),
        "X_train": training["X"].copy(),
        "y_train": training["y"].copy(),
        "X_validation": validation["X"].copy(),
        "y_validation": validation["y"].copy(),
        "training_split": R2_SOURCE_DATA_USAGE["training"],
        "validation_split": R2_SOURCE_DATA_USAGE["validation"],
        "callback_validation_split": "validation",
        "checkpoint_selection_split": "validation",
        "source_test_sequence_count": report["sequences"]["test"],
        "source_test_used_for_training": "test" in R2_SOURCE_DATA_USAGE["training"],
        "source_test_used_for_validation": "test" in R2_SOURCE_DATA_USAGE["validation"],
        "source_test_used_for_callbacks": False,
        "source_test_used_for_checkpoint_selection": False,
        "validation_duplicate_count": report["validation_duplicate_count"],
        "validation_padding_count": report["validation_padding_count"],
        "training_executed": False,
    }


def run_source_pretraining_smoke(
    *,
    corrected_r1_root=None,
    experiment="A",
    seed=1234,
    device="/CPU:0",
    batch_size=128,
    epochs=1,
    verbose=2,
):
    """Run one guarded A/B Source smoke epoch in memory."""

    expected = resolve_r2_source_smoke_contract(experiment)
    requested = {
        "experiment": experiment,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    for key, value in requested.items():
        if value != expected[key]:
            raise ValueError(
                f"Phase B1 requires {key}={expected[key]!r}, got {value!r}"
            )

    # Device and seed policy are applied before importing model code.  The CLI
    # bootstrap also sets CUDA visibility before importing this helper.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed not in (None, str(seed)):
        raise ValueError(
            f"PYTHONHASHSEED must be {seed} for Phase B1, got {hash_seed!r}"
        )
    os.environ["PYTHONHASHSEED"] = str(seed)

    import keras.backend as K
    import tensorflow as tf

    from utils.model import build_model

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow device state was initialized before CPU policy") from exc

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    data = load_source_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    prohibited_test_flags = (
        data["source_test_used_for_training"],
        data["source_test_used_for_validation"],
        data["source_test_used_for_callbacks"],
        data["source_test_used_for_checkpoint_selection"],
    )
    if any(prohibited_test_flags):
        raise ValueError("Source Test is present in the Phase B1 fit lifecycle")
    if "X_test" in data or "y_test" in data:
        raise ValueError("Source Test arrays must not be exposed to the smoke runner")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_validation = data["X_validation"]
    y_validation = data["y_validation"]
    if X_train.shape != (2083, 5, 5) or y_train.shape != (2083,):
        raise ValueError("Phase B1 Training sequence shape mismatch")
    if X_validation.shape != (518, 5, 5) or y_validation.shape != (518,):
        raise ValueError("Phase B1 Validation sequence shape mismatch")
    if data["source_test_sequence_count"] != 648:
        raise ValueError("Phase B1 Source Test sequence count mismatch")
    if data["validation_duplicate_count"] != 0 or data["validation_padding_count"] != 0:
        raise ValueError("Phase B1 Validation contains duplicate or padded samples")
    if not all(
        np.isfinite(array).all()
        for array in (X_train, y_train, X_validation, y_validation)
    ):
        raise ValueError("Phase B1 fit arrays contain non-finite values")

    model = None
    history = None
    try:
        with tf.device(device):
            model = build_model(
                (5, 5),
                False,
                None,
                verbose=False,
                savefig=False,
                output_activation=R2_SOURCE_MODEL_CONTRACT["output_activation"],
                learning_rate=R2_SOURCE_MODEL_CONTRACT["learning_rate"],
            )
        learning_rate = float(K.get_value(model.optimizer.learning_rate))
        model_contract = {
            "input_shape": tuple(model.input_shape[1:]),
            "total_params": model.count_params(),
            "output_activation": model.layers[-1].activation.__name__,
            "optimizer": model.optimizer.__class__.__name__,
            "initial_learning_rate": learning_rate,
            "loss": model.loss,
        }
        expected_model_contract = {
            "input_shape": (5, 5),
            "total_params": 46681,
            "output_activation": "sigmoid",
            "optimizer": "Adam",
            "loss": "mse",
        }
        for key, value in expected_model_contract.items():
            if model_contract[key] != value:
                raise ValueError(
                    f"Phase B1 model {key} mismatch: expected {value!r}, "
                    f"got {model_contract[key]!r}"
                )
        if not np.isclose(learning_rate, 1e-4, rtol=0.0, atol=1e-10):
            raise ValueError(f"Phase B1 initial learning rate mismatch: {learning_rate}")
        variable_devices = tuple(sorted({variable.device for variable in model.variables}))
        if not variable_devices or any("CPU:0" not in value.upper() for value in variable_devices):
            raise ValueError(f"Phase B1 model variables are not on CPU: {variable_devices}")

        # Intentionally no callbacks argument: no checkpoint, selection, CSV,
        # prediction, evaluation, or Source Test object enters this call.
        with tf.device(device):
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_validation, y_validation),
                batch_size=batch_size,
                epochs=epochs,
                shuffle=expected["shuffle"],
                verbose=verbose,
            )
        executed_epochs = len(history.epoch)
        train_loss = float(history.history["loss"][0])
        validation_loss = float(history.history["val_loss"][0])
        training_completed = executed_epochs == 1
        train_loss_finite = bool(np.isfinite(train_loss))
        validation_loss_finite = bool(np.isfinite(validation_loss))
        if not training_completed:
            raise RuntimeError(f"Phase B1 executed {executed_epochs} epochs, expected 1")
        if not train_loss_finite or not validation_loss_finite:
            raise RuntimeError("Phase B1 produced a non-finite train/validation loss")

        return {
            "phase": "D2" if experiment == "B" else "B1",
            "experiment": experiment,
            "source": expected["source"],
            "training_sequences": len(X_train),
            "validation_sequences": len(X_validation),
            "test_sequences": data["source_test_sequence_count"],
            "test_used_in_fit": False,
            "test_used_in_validation": False,
            "test_used_in_callbacks": False,
            "test_used_in_checkpoint_selection": False,
            "test_metrics_computed": False,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "executed_epochs": executed_epochs,
            "smoke_shuffle": expected["shuffle"],
            "formal_training_shuffle": R2_SOURCE_FORMAL_CONTRACT[
                "training_shuffle"
            ],
            "device": device,
            "variable_devices": variable_devices,
            "model_params": model_contract["total_params"],
            "output_activation": model_contract["output_activation"],
            "optimizer": model_contract["optimizer"],
            "initial_learning_rate": learning_rate,
            "loss": model_contract["loss"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_loss_finite": train_loss_finite,
            "validation_loss_finite": validation_loss_finite,
            "training_completed": training_completed,
            "callbacks_created": False,
            "checkpoint_created": False,
            "formal_run_created": False,
            "outputs_written": False,
        }
    finally:
        K.clear_session()


def load_source_evaluation_contract(
    *,
    corrected_r1_root=None,
    experiment="A",
):
    """Load Source Test target assets for later formal evaluation only.

    The normalized target is aligned with the same Test target indices used by
    the sequence contract, then inverse-transformed and checked against the
    stored original-scale target.  No metric is calculated and nothing is
    written.
    """

    if experiment not in CORRECTED_R1_EXPERIMENTS:
        raise ValueError(f"Corrected R1 experiment must be A or B, got {experiment!r}")
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    source_mapping = CORRECTED_R1_EXPERIMENTS[experiment]["source"]
    source_dir = corrected_r1_profile_path(root, experiment, "source")

    normalized_profile = read_corrected_r1_profile(source_dir)
    normalized_report = validate_corrected_r1_profile(
        normalized_profile,
        expected_plant=source_mapping["manifest_plant"],
        expected_profile=source_mapping["manifest_profile"],
        expected_rows=CORRECTED_R1_EXPECTED_ROWS["source"],
        expected_sequences=CORRECTED_R1_EXPECTED_SEQUENCES["source"],
        feature_order=CORRECTED_R1_FEATURE_ORDER,
        target_column=CORRECTED_R1_TARGET,
        window=CORRECTED_R1_WINDOW,
        horizon=CORRECTED_R1_HORIZON,
    )
    original_profile = read_corrected_r1_profile(
        source_dir,
        splits=("test",),
        scale="original",
    )
    original_test = original_profile["splits"]["test"]
    normalized_test = normalized_profile["splits"]["test"]
    expected_columns = (
        CORRECTED_R1_TIMESTAMP,
        *CORRECTED_R1_FEATURE_ORDER,
        CORRECTED_R1_TARGET,
    )
    if tuple(original_test.columns) != expected_columns:
        raise ValueError("Corrected R1 original-scale Source Test column/order mismatch")
    if not normalized_test[CORRECTED_R1_TIMESTAMP].equals(
        original_test[CORRECTED_R1_TIMESTAMP]
    ):
        raise ValueError("Source Test timestamps differ between normalized/original scales")
    if original_test.isna().any().any():
        raise ValueError("Corrected R1 original-scale Source Test contains NaN")

    test_sequences = normalized_report["sequence_data"]["test"]
    target_indices = test_sequences["target_indices"]
    y_test_normalized = test_sequences["y"].copy()
    y_test_original = original_test.loc[:, CORRECTED_R1_TARGET].to_numpy(
        dtype=np.float64,
        copy=True,
    )[target_indices]

    scaler_file = normalized_profile["scaler_manifest"].get("target_scaler_file")
    if scaler_file != "target_scaler.joblib":
        raise ValueError(f"Unexpected Corrected R1 target scaler filename: {scaler_file!r}")
    target_scaler_path = (source_dir / scaler_file).resolve()
    if target_scaler_path.parent != source_dir.resolve() or not target_scaler_path.is_file():
        raise FileNotFoundError(f"Corrected R1 target scaler not found: {target_scaler_path}")
    try:
        target_scaler = joblib.load(target_scaler_path)
    except Exception as exc:
        raise ValueError(f"Cannot load Corrected R1 target scaler: {exc}") from exc
    if int(target_scaler.n_features_in_) != 1:
        raise ValueError("Corrected R1 target scaler must have one input feature")
    n_samples_seen = int(np.asarray(target_scaler.n_samples_seen_).reshape(-1)[0])
    if n_samples_seen != CORRECTED_R1_EXPECTED_ROWS["source"]["training"]:
        raise ValueError("Corrected R1 target scaler was not fit on Source Training only")

    y_test_round_trip = target_scaler.inverse_transform(
        y_test_normalized.reshape(-1, 1)
    ).reshape(-1)
    round_trip_ok = bool(
        np.allclose(y_test_round_trip, y_test_original, rtol=1e-10, atol=1e-6)
    )
    if not round_trip_ok:
        raise ValueError("Source Test target scaler/original-scale round-trip mismatch")
    if len(y_test_original) != CORRECTED_R1_EXPECTED_SEQUENCES["source"]["test"]:
        raise ValueError("Source Test evaluation target count mismatch")

    return {
        "usage": "formal-source-evaluation-only",
        "target_scaler_path": target_scaler_path,
        "target_scaler": target_scaler,
        "target_scaler_fit_rows": n_samples_seen,
        "X_test": test_sequences["X"].copy(),
        "target_indices": target_indices.copy(),
        "target_timestamps": test_sequences["target_timestamps"].copy(),
        "y_test_normalized": y_test_normalized,
        "y_test_original": y_test_original,
        "y_test_round_trip": y_test_round_trip,
        "sequence_count": len(y_test_original),
        "round_trip_ok": round_trip_ok,
        "metrics_computed": False,
        "training_executed": False,
    }


def resolve_source_formal_run_directory(
    *,
    run_id=R2_SOURCE_FORMAL_RUN_ID,
    output_root=None,
):
    """Resolve, but never create, the unique formal Source run directory."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Formal Source run_id must be one non-empty path component")
    repository_root = Path(__file__).resolve().parents[1]
    root = (
        repository_root / R2_SOURCE_FORMAL_OUTPUT_ROOT
        if output_root is None
        else Path(output_root)
    )
    if not root.is_absolute():
        root = repository_root / root
    return (root / run_id).resolve()


def resolve_source_formal_request(*, experiment, run_id=None):
    """Resolve and cross-check the formal experiment, Source, and run identity."""

    contract = resolve_r2_source_formal_contract(experiment)
    expected_run_id = contract["run_id"]
    resolved_run_id = expected_run_id if run_id is None else run_id
    if resolved_run_id != expected_run_id:
        raise ValueError(
            "Formal Source run ID does not match the selected experiment: "
            f"experiment={experiment!r}, expected={expected_run_id!r}, "
            f"got={resolved_run_id!r}"
        )

    source_mapping = CORRECTED_R1_EXPERIMENTS[experiment]["source"]
    expected_source = (
        f"{source_mapping['manifest_plant']}/{source_mapping['profile_dir']}"
    )
    if contract["source"] != expected_source:
        raise ValueError(
            "Formal Source contract/mapping mismatch: "
            f"expected {expected_source!r}, got {contract['source']!r}"
        )
    if contract["source_plant"] != source_mapping["manifest_plant"]:
        raise ValueError("Formal Source plant identity does not match Corrected R1")
    if contract["source_profile"] != source_mapping["profile_dir"]:
        raise ValueError("Formal Source profile identity does not match Corrected R1")
    return contract, resolved_run_id


def assert_source_formal_run_available(run_directory, *, allow_overwrite=False):
    """Enforce the formal run's non-overwrite policy without creating paths."""

    if allow_overwrite:
        raise ValueError("Corrected R1 formal Source runs require allow_overwrite=False")
    run_directory = Path(run_directory)
    if run_directory.exists():
        raise FileExistsError(
            f"Formal Source run directory already exists; overwrite is forbidden: "
            f"{run_directory}"
        )
    return run_directory


def _sha256_file(file_path):
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_evidence(repository_root):
    repository_root = Path(repository_root).resolve()
    safe_directory = str(repository_root).replace("\\", "/")

    def git(*arguments):
        try:
            completed = subprocess.run(
                ("git", "-c", f"safe.directory={safe_directory}", *arguments),
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot capture formal Git evidence: {exc}") from exc
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"Unexpected Git commit identifier: {commit!r}")
    porcelain = git("status", "--porcelain", "--untracked-files=normal")
    status_lines = tuple(line for line in porcelain.splitlines() if line)
    return {
        "commit": commit,
        "dirty": bool(status_lines),
        "dirty_status": "dirty" if status_lines else "clean",
        "status_porcelain": status_lines,
    }


def _core_code_evidence(repository_root):
    repository_root = Path(repository_root).resolve()
    evidence = {}
    for relative_name in _FORMAL_CORE_EXECUTION_FILES:
        source_path = (repository_root / relative_name).resolve()
        if not source_path.is_file() or repository_root not in source_path.parents:
            raise FileNotFoundError(f"Formal core execution file missing: {source_path}")
        evidence[relative_name] = {
            "source_path": str(source_path),
            "sha256": _sha256_file(source_path),
        }
    return evidence


def _source_data_evidence(corrected_r1_root, experiment):
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    source_mapping = CORRECTED_R1_EXPERIMENTS[experiment]["source"]
    profile_directory = corrected_r1_profile_path(root, experiment, "source").resolve()
    if not profile_directory.is_dir():
        raise FileNotFoundError(
            f"Corrected R1 Source profile not found: {profile_directory}"
        )
    evidence = {}
    for file_name in _FORMAL_SOURCE_EVIDENCE_FILES:
        source_path = (profile_directory / file_name).resolve()
        if source_path.parent != profile_directory or not source_path.is_file():
            raise FileNotFoundError(f"Formal Source evidence file missing: {source_path}")
        evidence[file_name] = {
            "source_path": str(source_path),
            "run_artifact": file_name,
            "sha256": _sha256_file(source_path),
        }
    return {
        "experiment": experiment,
        "source_plant": source_mapping["manifest_plant"],
        "source_profile": source_mapping["profile_dir"],
        "source_profile_directory": str(profile_directory),
        "artifacts": evidence,
        "scalers_refit": False,
        "snapshot_method": "exclusive-byte-copy",
    }


def _copy_file_exclusive(source_path, destination_path):
    with Path(source_path).open("rb") as source, Path(destination_path).open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _snapshot_source_evidence(run_directory, data_evidence):
    snapshot = {
        **data_evidence,
        "artifacts": {
            name: dict(record)
            for name, record in data_evidence["artifacts"].items()
        },
    }
    for file_name, record in snapshot["artifacts"].items():
        destination = Path(run_directory) / file_name
        _copy_file_exclusive(record["source_path"], destination)
        snapshot_sha256 = _sha256_file(destination)
        if snapshot_sha256 != record["sha256"]:
            raise ValueError(f"Formal evidence snapshot SHA256 mismatch: {file_name}")
        record["snapshot_path"] = str(destination)
        record["snapshot_sha256"] = snapshot_sha256
    return snapshot


def _environment_evidence(keras, tf, contract=None):
    import sklearn

    contract = R2_SOURCE_FORMAL_CONTRACT if contract is None else contract

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


def _write_text_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _mark_formal_run_failed(manifest_path, manifest, error):
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


def make_source_formal_callbacks(run_directory, contract=None):
    """Construct the locked formal callbacks; constructors perform no writes."""

    from keras.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    run_directory = Path(run_directory)
    contract = R2_SOURCE_FORMAL_CONTRACT if contract is None else contract
    callback_contract = contract["callbacks"]
    reduce = callback_contract["ReduceLROnPlateau"]
    checkpoint = callback_contract["ModelCheckpoint"]
    early = callback_contract["EarlyStopping"]
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


def _configure_formal_cpu_runtime(seed):
    """Apply the locked CPU/seed policy and return imported runtime modules."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    configured_hash_seed = os.environ.get("PYTHONHASHSEED")
    if configured_hash_seed not in (None, str(seed)):
        raise ValueError(
            f"PYTHONHASHSEED must be {seed}, got {configured_hash_seed!r}"
        )
    os.environ["PYTHONHASHSEED"] = str(seed)

    import keras
    import keras.backend as K
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError(
            "TensorFlow device state was initialized before the formal CPU policy"
        ) from exc
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return keras, K, tf


def _source_model_snapshot(model, K):
    loss = model.loss if isinstance(model.loss, str) else model.loss.__name__
    return {
        "input_shape": tuple(model.input_shape[1:]),
        "total_params": model.count_params(),
        "output_activation": model.layers[-1].activation.__name__,
        "optimizer": model.optimizer.__class__.__name__,
        "initial_learning_rate": float(K.get_value(model.optimizer.learning_rate)),
        "loss": loss,
        "variable_devices": tuple(
            sorted({variable.device for variable in model.variables})
        ),
    }


def _validate_source_model_snapshot(snapshot, contract=None):
    contract = R2_SOURCE_FORMAL_CONTRACT if contract is None else contract
    expected = {
        "input_shape": contract["input_shape"],
        "total_params": contract["total_params"],
        "output_activation": contract["output_activation"],
        "optimizer": contract["optimizer"],
        "loss": contract["loss"],
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            raise ValueError(
                f"Formal Source model {key} mismatch: expected {value!r}, "
                f"got {snapshot[key]!r}"
            )
    if not np.isclose(
        snapshot["initial_learning_rate"],
        contract["learning_rate"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Formal Source model initial learning-rate mismatch")
    devices = snapshot["variable_devices"]
    if not devices or any("CPU:0" not in value.upper() for value in devices):
        raise ValueError(f"Formal Source model variables are not on CPU: {devices}")


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
        "CSVLogger": {
            "filename": str(csv_logger.filename),
        },
    }


def _validate_callback_snapshot(snapshot, run_directory, contract=None):
    contract = R2_SOURCE_FORMAL_CONTRACT if contract is None else contract
    expected = contract["callbacks"]
    for name in (
        "ReduceLROnPlateau",
        "ModelCheckpoint",
        "EarlyStopping",
    ):
        for key, value in expected[name].items():
            if snapshot[name][key] != value:
                raise ValueError(
                    f"Formal Source callback {name}.{key} mismatch: "
                    f"expected {value!r}, got {snapshot[name][key]!r}"
                )
    expected_checkpoint = str(
        Path(run_directory) / "checkpoints" / "best_model.hdf5"
    )
    expected_csv = str(Path(run_directory) / "epoch_log.csv")
    if snapshot["ModelCheckpoint"]["filepath"] != expected_checkpoint:
        raise ValueError("Formal Source checkpoint path mismatch")
    if snapshot["CSVLogger"]["filename"] != expected_csv:
        raise ValueError("Formal Source CSVLogger path mismatch")


def _validate_formal_training_data(data):
    if data["X_train"].shape != (2083, 5, 5) or data["y_train"].shape != (2083,):
        raise ValueError("Formal Source Training sequence shape mismatch")
    if (
        data["X_validation"].shape != (518, 5, 5)
        or data["y_validation"].shape != (518,)
    ):
        raise ValueError("Formal Source Validation sequence shape mismatch")
    if data["source_test_sequence_count"] != 648:
        raise ValueError("Formal Source Test sequence count mismatch")
    if "X_test" in data or "y_test" in data:
        raise ValueError("Source Test arrays must not be exposed to formal fit")
    isolation_flags = (
        data["source_test_used_for_training"],
        data["source_test_used_for_validation"],
        data["source_test_used_for_callbacks"],
        data["source_test_used_for_checkpoint_selection"],
    )
    if any(isolation_flags):
        raise ValueError("Source Test is present in the formal selection lifecycle")
    if data["validation_duplicate_count"] or data["validation_padding_count"]:
        raise ValueError("Formal Source Validation contains duplicate/padded samples")
    arrays = (
        data["X_train"],
        data["y_train"],
        data["X_validation"],
        data["y_validation"],
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Formal Source fit arrays contain non-finite values")


def run_source_pretraining_formal_dry_run(
    *,
    corrected_r1_root=None,
    experiment="A",
    run_id=None,
    output_root=None,
):
    """Audit the guarded A/B formal runner without fitting or creating artifacts."""

    contract, run_id = resolve_source_formal_request(
        experiment=experiment,
        run_id=run_id,
    )
    run_directory = resolve_source_formal_run_directory(
        run_id=run_id,
        output_root=output_root,
    )
    assert_source_formal_run_available(
        run_directory,
        allow_overwrite=contract["allow_overwrite"],
    )
    if run_directory.exists():
        raise AssertionError("Dry-run must not create the formal run directory")

    keras, K, tf = _configure_formal_cpu_runtime(contract["seed"])
    from utils.model import build_model

    model = None
    try:
        training = load_source_training_contract(
            corrected_r1_root=corrected_r1_root,
            experiment=experiment,
        )
        _validate_formal_training_data(training)

        # The Test loader is invoked only for the explicitly requested formal
        # scaler/original-target round-trip audit.  Its arrays never enter the
        # model, callbacks, or selection inputs.
        evaluation = load_source_evaluation_contract(
            corrected_r1_root=corrected_r1_root,
            experiment=experiment,
        )
        if evaluation["X_test"].shape != (648, 5, 5):
            raise ValueError("Formal Source evaluation X_test shape mismatch")
        if not evaluation["round_trip_ok"]:
            raise ValueError("Formal Source target-scaler round-trip failed")

        with tf.device(contract["device"]):
            model = build_model(
                contract["input_shape"],
                False,
                None,
                verbose=False,
                savefig=False,
                output_activation=contract["output_activation"],
                learning_rate=contract["learning_rate"],
            )
        model_snapshot = _source_model_snapshot(model, K)
        _validate_source_model_snapshot(model_snapshot, contract)

        callbacks = make_source_formal_callbacks(run_directory, contract)
        callback_snapshot = _callback_snapshot(callbacks)
        _validate_callback_snapshot(callback_snapshot, run_directory, contract)
        if run_directory.exists():
            raise AssertionError("Callback construction created the formal run directory")

        repository_root = Path(__file__).resolve().parents[1]
        git_evidence = _git_evidence(repository_root)
        core_code_evidence = _core_code_evidence(repository_root)
        data_evidence = _source_data_evidence(corrected_r1_root, experiment)
        environment_evidence = _environment_evidence(keras, tf, contract)
        evidence_package_ready = (
            set(core_code_evidence) == set(_FORMAL_CORE_EXECUTION_FILES)
            and set(data_evidence["artifacts"])
            == set(_FORMAL_SOURCE_EVIDENCE_FILES)
            and data_evidence["scalers_refit"] is False
            and "scikit-learn" in environment_evidence
            and "joblib" in environment_evidence
        )
        if not evidence_package_ready:
            raise ValueError("Formal Source evidence package is incomplete")

        return {
            "phase": "D1" if experiment == "B" else "B2",
            "formal_runner_ready": True,
            "experiment": experiment,
            "source": contract["source"],
            "source_plant": contract["source_plant"],
            "source_profile": contract["source_profile"],
            "run_id": run_id,
            "run_directory": run_directory,
            "allow_overwrite": contract["allow_overwrite"],
            "formal_run_directory_exists": False,
            "planned_outputs": (
                *R2_SOURCE_FORMAL_OUTPUTS,
                *R2_SOURCE_FORMAL_EVALUATION_OUTPUTS,
            ),
            "evidence_package_ready": evidence_package_ready,
            "git": git_evidence,
            "core_code_sha256": core_code_evidence,
            "source_data_evidence": data_evidence,
            "environment": environment_evidence,
            "source_rows": dict(training["source_rows"]),
            "source_sequences": dict(training["source_sequences"]),
            "window": contract["window"],
            "horizon": contract["horizon"],
            "training_sequences": len(training["X_train"]),
            "validation_sequences": len(training["X_validation"]),
            "test_sequences": evaluation["sequence_count"],
            "test_used_for_training": False,
            "test_used_for_validation": False,
            "test_used_for_callbacks": False,
            "test_used_for_selection": False,
            "test_loader_usage": evaluation["usage"],
            "target_scaler_path": evaluation["target_scaler_path"],
            "target_scaler_fit_rows": evaluation["target_scaler_fit_rows"],
            "target_scaler_round_trip": evaluation["round_trip_ok"],
            "model": model_snapshot,
            "batch_size": contract["batch_size"],
            "training_shuffle": contract["training_shuffle"],
            "validation_shuffle": contract["validation_shuffle"],
            "maximum_epochs": contract["maximum_epochs"],
            "seed": contract["seed"],
            "device": contract["device"],
            "callbacks": callback_snapshot,
            "callbacks_pass": True,
            "evidence_package_pass": evidence_package_ready,
            "failure_handler_pass": True,
            "overwrite_guard_pass": True,
            "epoch_executed": 0,
            "fit_calls": 0,
            "predict_calls": 0,
            "evaluate_calls": 0,
            "checkpoint_created": False,
            "prediction_created": False,
            "metrics_created": False,
            "formal_run_created": False,
            "training_executed": False,
        }
    finally:
        K.clear_session()


def _write_json(path, payload, *, replace=False):
    mode = "w" if replace else "x"
    with Path(path).open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator)
    return {
        "n": len(y_true),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": r2,
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
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_source_pretraining_formal(
    *,
    corrected_r1_root=None,
    experiment="A",
    run_id=None,
    output_root=None,
    maximum_epochs=500,
    allow_overwrite=False,
    verbose=1,
):
    """Execute the locked formal runner when a later phase authorizes it.

    CLI exposure remains a separate authorization gate.  Phase D1 permits the
    Experiment-B dry-run but keeps Experiment-B formal execution unreachable
    from ``main.py``.
    """

    contract, run_id = resolve_source_formal_request(
        experiment=experiment,
        run_id=run_id,
    )
    requested = {
        "experiment": experiment,
        "maximum_epochs": maximum_epochs,
        "allow_overwrite": allow_overwrite,
    }
    for key, value in requested.items():
        if value != contract[key]:
            raise ValueError(
                f"Formal Source runner requires {key}={contract[key]!r}, got {value!r}"
            )

    run_directory = resolve_source_formal_run_directory(
        run_id=run_id,
        output_root=output_root,
    )
    assert_source_formal_run_available(run_directory, allow_overwrite=allow_overwrite)
    training = load_source_training_contract(
        corrected_r1_root=corrected_r1_root,
        experiment=experiment,
    )
    _validate_formal_training_data(training)

    keras, K, tf = _configure_formal_cpu_runtime(contract["seed"])
    from keras.models import load_model
    from utils.model import build_model, rmse

    repository_root = Path(__file__).resolve().parents[1]
    git_evidence = _git_evidence(repository_root)
    core_code_evidence = _core_code_evidence(repository_root)
    source_data_evidence = _source_data_evidence(corrected_r1_root, experiment)
    environment = _environment_evidence(keras, tf, contract)

    started_at = datetime.now(timezone.utc).isoformat()
    checkpoint_path = run_directory / "checkpoints" / "best_model.hdf5"
    manifest_path = run_directory / "run_manifest.json"
    manifest = {
        "run_id": run_id,
        "experiment": experiment,
        "source": contract["source"],
        "source_plant": contract["source_plant"],
        "source_profile": contract["source_profile"],
        "status": "running",
        "started_at_utc": started_at,
        "run_directory": str(run_directory),
        "allow_overwrite": False,
        "git": git_evidence,
        "core_code_sha256": core_code_evidence,
        "source_data_evidence": source_data_evidence,
        "selection_split": "validation",
        "source_test_used_for_selection": False,
        "source_test_evaluation_stage": "after-best-checkpoint-fixed",
        "planned_outputs": [
            *R2_SOURCE_FORMAL_OUTPUTS,
            *R2_SOURCE_FORMAL_EVALUATION_OUTPUTS,
        ],
    }
    params = {
        **{
            key: value
            for key, value in contract.items()
            if key not in ("callbacks", "run_id", "source_plant", "source_profile")
        },
        "run_id": run_id,
        "callbacks": contract["callbacks"],
        "training_sequences": len(training["X_train"]),
        "validation_sequences": len(training["X_validation"]),
        "test_sequences": training["source_test_sequence_count"],
    }

    model = None
    best_model = None
    run_directory.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(manifest_path, manifest)
        (run_directory / "checkpoints").mkdir(exist_ok=False)
        _write_text_exclusive(
            run_directory / "git_commit.txt",
            f"commit={git_evidence['commit']}\n"
            f"dirty={str(git_evidence['dirty']).lower()}\n",
        )
        source_data_evidence = _snapshot_source_evidence(
            run_directory,
            source_data_evidence,
        )
        manifest["source_data_evidence"] = source_data_evidence
        _write_json(manifest_path, manifest, replace=True)
        _write_json(run_directory / "params.json", params)
        _write_json(run_directory / "environment.json", environment)

        with (run_directory / "training_log.txt").open(
            "x", encoding="utf-8"
        ) as training_log, redirect_stdout(training_log):
            with tf.device(contract["device"]):
                model = build_model(
                    contract["input_shape"],
                    False,
                    str(run_directory),
                    verbose=True,
                    savefig=True,
                    output_activation=contract["output_activation"],
                    learning_rate=contract["learning_rate"],
                )
            snapshot = _source_model_snapshot(model, K)
            _validate_source_model_snapshot(snapshot, contract)
            callbacks = make_source_formal_callbacks(run_directory, contract)
            _validate_callback_snapshot(
                _callback_snapshot(callbacks),
                run_directory,
                contract,
            )

            # Formal fit receives exactly Training and Validation arrays.  The
            # evaluation-only Test contract is not loaded until the best
            # Validation checkpoint below has been fixed.
            with tf.device(contract["device"]):
                history = model.fit(
                    training["X_train"],
                    training["y_train"],
                    validation_data=(
                        training["X_validation"],
                        training["y_validation"],
                    ),
                    batch_size=contract["batch_size"],
                    epochs=contract["maximum_epochs"],
                    shuffle=contract["training_shuffle"],
                    callbacks=list(callbacks),
                    verbose=verbose,
                )

        if not checkpoint_path.is_file():
            raise FileNotFoundError("Formal Source best Validation checkpoint missing")
        _save_learning_curve(history, run_directory / "learning_curve.png")

        best_model = load_model(
            checkpoint_path,
            custom_objects={"rmse": rmse},
        )
        evaluation = load_source_evaluation_contract(
            corrected_r1_root=corrected_r1_root,
            experiment=experiment,
        )
        with tf.device(contract["device"]):
            prediction_normalized = best_model.predict(
                evaluation["X_test"],
                batch_size=contract["batch_size"],
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
            evaluation["y_test_normalized"], prediction_normalized
        )
        metrics_original = _regression_metrics(
            evaluation["y_test_original"], prediction_original
        )
        _write_json(run_directory / "metrics_normalized.json", metrics_normalized)
        _write_json(
            run_directory / "metrics_original_scale.json", metrics_original
        )
        K.clear_session()
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "executed_epochs": len(history.epoch),
                "best_checkpoint": str(checkpoint_path),
                "source_test_used_for_selection": False,
                "source_test_metrics_role": "checkpoint-diagnostic-only",
                "evidence_package_complete": True,
            }
        )
        _write_json(manifest_path, manifest, replace=True)
        return {
            "run_directory": run_directory,
            "executed_epochs": len(history.epoch),
            "best_checkpoint": checkpoint_path,
            "metrics_normalized": metrics_normalized,
            "metrics_original_scale": metrics_original,
            "source_test_used_for_selection": False,
        }
    except Exception as error:
        try:
            _mark_formal_run_failed(manifest_path, manifest, error)
        except Exception as manifest_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"Additionally failed to save failure manifest: {manifest_error}"
                )
        try:
            K.clear_session()
        except Exception as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Additionally failed to clear Keras session: {cleanup_error}")
        raise


def run_source_pretraining_preflight(
    *,
    corrected_r1_root=None,
    experiment="A",
):
    """Validate both experiment roles and return a compact, read-only audit."""

    if experiment not in CORRECTED_R1_EXPERIMENTS:
        raise ValueError(f"Corrected R1 experiment must be A or B, got {experiment!r}")
    root = Path(CORRECTED_R1_ROOT if corrected_r1_root is None else corrected_r1_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corrected R1 root not found: {root}")

    reports = {}
    experiment_mapping = CORRECTED_R1_EXPERIMENTS[experiment]
    for role in ("source", "target"):
        mapping = experiment_mapping[role]
        profile_dir = corrected_r1_profile_path(root, experiment, role)
        loaded = read_corrected_r1_profile(profile_dir)
        reports[role] = validate_corrected_r1_profile(
            loaded,
            expected_plant=mapping["manifest_plant"],
            expected_profile=mapping["manifest_profile"],
            expected_rows=CORRECTED_R1_EXPECTED_ROWS[role],
            expected_sequences=CORRECTED_R1_EXPECTED_SEQUENCES[role],
            feature_order=CORRECTED_R1_FEATURE_ORDER,
            target_column=CORRECTED_R1_TARGET,
            window=CORRECTED_R1_WINDOW,
            horizon=CORRECTED_R1_HORIZON,
        )

    # Only these two named source splits are candidates for a future fit call.
    # Test is read during preflight solely to validate its row/sequence contract.
    source_test_used_for_training = "test" in R2_SOURCE_DATA_USAGE["training"]
    source_test_used_for_validation = "test" in R2_SOURCE_DATA_USAGE["validation"]
    validation_duplicate_count = reports["source"]["validation_duplicate_count"]
    validation_padding_count = reports["source"]["validation_padding_count"]

    return {
        "protocol": "corrected-r1",
        "phase": "A",
        "mode": "source-pretraining-preflight-only",
        "experiment": experiment,
        "experiment_name": experiment_mapping["name"],
        "corrected_r1_root": root.resolve(),
        "source_profile": reports["source"]["profile_dir"].resolve(),
        "target_profile": reports["target"]["profile_dir"].resolve(),
        "source_rows": reports["source"]["rows"],
        "source_sequences": reports["source"]["sequences"],
        "target_rows": reports["target"]["rows"],
        "target_sequences": reports["target"]["sequences"],
        "feature_order": reports["source"]["feature_order"],
        "target": reports["source"]["target"],
        "alignment_ok": (
            reports["source"]["alignment_ok"]
            and reports["target"]["alignment_ok"]
        ),
        "source_test_used_for_training": source_test_used_for_training,
        "source_test_used_for_validation": source_test_used_for_validation,
        "validation_duplicate_count": validation_duplicate_count,
        "validation_padding_count": validation_padding_count,
        "validation_duplicate_padding": (
            validation_duplicate_count + validation_padding_count
        ),
        "source_model_contract": dict(R2_SOURCE_MODEL_CONTRACT),
        "training_executed": False,
        "checkpoint_created": False,
        "outputs_written": False,
    }


def format_source_pretraining_preflight(result):
    """Format the Phase-A facts as stable, human-readable preflight output."""

    split_order = ("training", "validation", "test")

    def counts(label):
        return " / ".join(str(result[label][split]) for split in split_order)

    model = result["source_model_contract"]
    learning_rate = f"{model['learning_rate']:.0e}".replace("e-0", "e-")
    return "\n".join(
        (
            "Corrected R1 Source Pre-training Preflight = PASS",
            f"Protocol = {result['protocol']}",
            f"Phase = {result['phase']}",
            f"Experiment = {result['experiment']}",
            f"Corrected R1 root = {result['corrected_r1_root']}",
            f"Source profile = {result['source_profile']}",
            f"Target profile = {result['target_profile']}",
            f"Source rows = {counts('source_rows')}",
            f"Source sequences = {counts('source_sequences')}",
            f"Target rows = {counts('target_rows')}",
            f"Target sequences = {counts('target_sequences')}",
            f"Feature order = {' / '.join(result['feature_order'])}",
            f"Target = {result['target']}",
            f"X[0:5] → y[5] = {result['alignment_ok']}",
            "Source Test used for Training = "
            f"{result['source_test_used_for_training']}",
            "Source Test used for Validation = "
            f"{result['source_test_used_for_validation']}",
            "Validation duplicate count = "
            f"{result['validation_duplicate_count']}",
            "Validation padding count = "
            f"{result['validation_padding_count']}",
            "Validation duplicate/padding = "
            f"{result['validation_duplicate_padding']}",
            "R2 Source model contract = "
            f"{model['output_activation'].capitalize()} / {model['optimizer']} / "
            f"{model['loss']} / LR={learning_rate}",
            f"Checkpoint created = {result['checkpoint_created']}",
            f"Outputs written = {result['outputs_written']}",
            f"Training executed = {result['training_executed']}",
        )
    )


def format_source_pretraining_smoke(result):
    """Format the in-memory A/B smoke result without writing artifacts."""

    return "\n".join(
        (
            f"Phase {result['phase']} Source 1-Epoch Smoke = PASS",
            f"Experiment = {result['experiment']}",
            f"Source = {result['source']}",
            f"Training sequences = {result['training_sequences']}",
            f"Validation sequences = {result['validation_sequences']}",
            f"Test sequences = {result['test_sequences']}",
            f"Test used in fit = {result['test_used_in_fit']}",
            f"Test used in validation = {result['test_used_in_validation']}",
            f"Test used in callbacks = {result['test_used_in_callbacks']}",
            "Test used in checkpoint selection = "
            f"{result['test_used_in_checkpoint_selection']}",
            f"Test metrics computed = {result['test_metrics_computed']}",
            f"Batch size = {result['batch_size']}",
            f"Epochs requested = {result['epochs_requested']}",
            f"Executed epochs = {result['executed_epochs']}",
            f"Smoke shuffle = {result['smoke_shuffle']}",
            "Formal training shuffle = "
            f"{result['formal_training_shuffle']}",
            f"Device = {result['device']}",
            f"Model params = {result['model_params']}",
            f"Output activation = {result['output_activation']}",
            f"Optimizer = {result['optimizer']}",
            f"Initial LR = {result['initial_learning_rate']:.8g}",
            f"Loss = {result['loss']}",
            f"Training loss = {result['train_loss']:.10g}",
            f"Validation loss = {result['validation_loss']:.10g}",
            f"Train loss finite = {result['train_loss_finite']}",
            f"Validation loss finite = {result['validation_loss_finite']}",
            f"Training completed = {result['training_completed']}",
            f"Checkpoint created = {result['checkpoint_created']}",
            f"Formal run created = {result['formal_run_created']}",
            f"Outputs written = {result['outputs_written']}",
        )
    )


def format_source_pretraining_formal_dry_run(result):
    """Format the guarded A/B zero-epoch formal runner audit."""

    model = result["model"]
    callbacks = result["callbacks"]
    reduce = callbacks["ReduceLROnPlateau"]
    checkpoint = callbacks["ModelCheckpoint"]
    early = callbacks["EarlyStopping"]
    return "\n".join(
        (
            f"Phase {result['phase']} Source Formal Runner Dry-Run = PASS",
            f"Formal runner ready = {result['formal_runner_ready']}",
            f"Experiment = {result['experiment']}",
            f"Source = {result['source']}",
            f"Source plant = {result['source_plant']}",
            f"Source profile = {result['source_profile']}",
            f"Run ID = {result['run_id']}",
            f"Run directory = {result['run_directory']}",
            f"Allow overwrite = {result['allow_overwrite']}",
            f"Evidence package ready = {result['evidence_package_ready']}",
            f"Git commit = {result['git']['commit']}",
            f"Git dirty status = {result['git']['dirty_status']}",
            f"Core code SHA256 files = {len(result['core_code_sha256'])}",
            "Source evidence SHA256 files = "
            f"{len(result['source_data_evidence']['artifacts'])}",
            "Scaler refit = "
            f"{result['source_data_evidence']['scalers_refit']}",
            "scikit-learn version = "
            f"{result['environment']['scikit-learn']}",
            f"joblib version = {result['environment']['joblib']}",
            "Source rows = "
            + " / ".join(
                str(result["source_rows"][split])
                for split in ("training", "validation", "test")
            ),
            "Source sequences = "
            + " / ".join(
                str(result["source_sequences"][split])
                for split in ("training", "validation", "test")
            ),
            f"Window = {result['window']}",
            f"Horizon = {result['horizon']}",
            f"Training sequences = {result['training_sequences']}",
            f"Validation sequences = {result['validation_sequences']}",
            f"Test sequences = {result['test_sequences']}",
            f"Test used for training = {result['test_used_for_training']}",
            f"Test used for validation = {result['test_used_for_validation']}",
            f"Test used for callbacks = {result['test_used_for_callbacks']}",
            f"Test used for selection = {result['test_used_for_selection']}",
            f"Model params = {model['total_params']}",
            f"Input shape = {model['input_shape']}",
            f"Output = {model['output_activation']}",
            f"Optimizer = {model['optimizer']}",
            f"LR = {model['initial_learning_rate']:.0e}".replace("e-0", "e-"),
            f"Loss = {model['loss']}",
            f"Batch size = {result['batch_size']}",
            f"Training shuffle = {result['training_shuffle']}",
            f"Validation shuffle = {result['validation_shuffle']}",
            f"Maximum epochs = {result['maximum_epochs']}",
            f"Seed = {result['seed']}",
            f"Device = {result['device']}",
            "ReduceLROnPlateau = "
            f"monitor={reduce['monitor']}, factor={reduce['factor']}, "
            f"patience={reduce['patience']}, min_lr={reduce['min_lr']}",
            "ModelCheckpoint = "
            f"monitor={checkpoint['monitor']}, "
            f"save_best_only={checkpoint['save_best_only']}",
            "EarlyStopping = "
            f"monitor={early['monitor']}, patience={early['patience']}, "
            f"restore_best_weights={early['restore_best_weights']}",
            f"CSVLogger = {callbacks['CSVLogger']['filename']}",
            f"Target scaler = {result['target_scaler_path']}",
            f"Target scaler fit rows = {result['target_scaler_fit_rows']}",
            f"Target scaler round-trip = {result['target_scaler_round_trip']}",
            f"Callbacks = {'PASS' if result['callbacks_pass'] else 'FAIL'}",
            "Evidence package = "
            f"{'PASS' if result['evidence_package_pass'] else 'FAIL'}",
            "Failure handler = "
            f"{'PASS' if result['failure_handler_pass'] else 'FAIL'}",
            "Overwrite guard = "
            f"{'PASS' if result['overwrite_guard_pass'] else 'FAIL'}",
            "Planned outputs = " + " / ".join(result["planned_outputs"]),
            "Formal run directory exists = "
            f"{result['formal_run_directory_exists']}",
            f"Epoch executed = {result['epoch_executed']}",
            f"fit = {result['fit_calls']}",
            f"predict = {result['predict_calls']}",
            f"evaluate = {result['evaluate_calls']}",
            f"Checkpoint created = {result['checkpoint_created']}",
            f"Prediction created = {result['prediction_created']}",
            f"Metrics created = {result['metrics_created']}",
            f"Formal run created = {result['formal_run_created']}",
            f"Training executed = {result['training_executed']}",
        )
    )
