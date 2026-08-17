"""R2.5 Phase-E one-time authorized Target Test evaluation."""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from r2_config.solar_r2 import EXPERIMENTS, EXPECTED_SEQUENCE_COUNTS, profile_path
from r2_config.solar_r25 import (
    BASELINE_RUN_ROOT,
    BASELINE_SOURCE_CHECKPOINT_SHA256,
    PRIMARY_STRATEGY_ID,
    R25_BATCH_SIZE,
)
from r2_helpers.solar_data import (
    build_sequences,
    load_profile_metadata,
    load_split,
    validate_feature_transform_consistency,
    validate_manifest,
    validate_scalers,
    validate_target_round_trip,
)
from r2_helpers.solar_formal import compute_metrics
from r2_helpers.solar_partial_ft import (
    SelectedTestAuthorization,
    authorize_selected_test,
    inverse_transform_original,
    reload_best_checkpoint,
    sha256_file,
)
from r2_helpers.solar_runtime import NamedSequenceSplit


PHASE_D_RUN_ID = "20260816T073233Z_seed1234"
PHASE_D_RUN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "Solar Energy Result"
    / "R2.5"
    / "Experiment_A"
    / "Partial_FT"
    / PHASE_D_RUN_ID
)
SELECTION_PATH = PHASE_D_RUN_ROOT / "selection.json"
SELECTED_CHECKPOINT_RELATIVE = Path(
    "candidate/partial_target_adapters_last_lstm/checkpoint_epoch_0500.hdf5"
)
SELECTED_CHECKPOINT_SHA256 = (
    "4dca61375c80d609f73ffc44e4cf829a4d1b20aab7e91e22f38072257e4679a6"
)
WITHOUT_TL_BASELINE = {
    "MAE": 2866.1041600963354,
    "MSE": 13705909.923006574,
    "RMSE": 3702.1493653020775,
    "R2": 0.6570718304225046,
    "n_samples": 2607,
}


class PhaseEError(RuntimeError):
    def __init__(self, message: str, *, output_dir: Path | None = None):
        super().__init__(message)
        self.output_dir = output_dir


@dataclass(frozen=True)
class PhaseEPreflight:
    phase_d_manifest: Mapping[str, Any]
    selection: Mapping[str, Any]
    phase_d_manifest_sha256: str
    selection_sha256: str
    selected_checkpoint: Path
    selected_checkpoint_sha256: str
    source_checkpoint_sha256: str
    baseline: Mapping[str, Any]


@dataclass(frozen=True)
class PhaseEResult:
    output_dir: Path
    manifest_path: Path
    classification: str
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class FinalTestData:
    named: NamedSequenceSplit
    split_data: Any
    sequences: Any
    target_scaler: Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseEError(message)


def _read_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"Required JSON is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseEError(f"Invalid JSON {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_record(root: Path, record: Mapping[str, Any]) -> None:
    path = root / str(record["path"])
    _require(path.is_file(), f"Phase D artifact missing: {path}")
    _require(sha256_file(path) == record["sha256"], f"Phase D artifact changed: {path}")


def validate_phase_e_preflight() -> PhaseEPreflight:
    """Verify immutable selection and artifacts without touching Target Test."""

    root = PHASE_D_RUN_ROOT.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    selection = _read_json(SELECTION_PATH)
    _require(manifest.get("status") == "PASS", "Phase D status must be PASS")
    _require(manifest.get("run_id") == PHASE_D_RUN_ID, "Phase D run ID mismatch")
    _require(manifest.get("selection_locked") is True, "Phase D selection is not locked")
    _require(selection.get("locked") is True, "selection.json is not locked")
    _require(selection.get("selected_strategy_id") == PRIMARY_STRATEGY_ID, "Strategy mismatch")
    _require(selection.get("best_epoch") == 500, "Selected epoch must be 500")
    _require(selection.get("test_metrics_used_for_selection") is False, "Test influenced selection")
    _require(
        manifest.get("source_checkpoint", {}).get("sha256")
        == BASELINE_SOURCE_CHECKPOINT_SHA256,
        "Source checkpoint SHA mismatch",
    )
    selected = (root / SELECTED_CHECKPOINT_RELATIVE).resolve()
    _require(selected.is_file(), "Selected checkpoint is missing")
    _require(Path(selection["selected_checkpoint_path"]).resolve() == selected, "Locked path mismatch")
    selected_sha = sha256_file(selected)
    _require(selected_sha == SELECTED_CHECKPOINT_SHA256, "Selected checkpoint SHA mismatch")
    _require(
        manifest.get("selected_checkpoint", {}).get("sha256") == selected_sha,
        "Manifest checkpoint SHA mismatch",
    )
    for record in (manifest["selected_checkpoint"], *manifest["artifacts"].values()):
        _verify_record(root, record)

    baseline_path = BASELINE_RUN_ROOT / "target" / "without_tl" / "metrics_original.json"
    baseline = _read_json(baseline_path)
    for name, expected in WITHOUT_TL_BASELINE.items():
        _require(np.isclose(baseline[name], expected, rtol=0, atol=1e-12), f"Baseline {name} changed")
    return PhaseEPreflight(
        phase_d_manifest=manifest,
        selection=selection,
        phase_d_manifest_sha256=sha256_file(manifest_path),
        selection_sha256=sha256_file(SELECTION_PATH),
        selected_checkpoint=selected,
        selected_checkpoint_sha256=selected_sha,
        source_checkpoint_sha256=BASELINE_SOURCE_CHECKPOINT_SHA256,
        baseline=dict(baseline),
    )


def load_target_test_once() -> FinalTestData:
    """Open corrected Plant2 Test CSVs once after all Phase-E gates pass."""

    mapping = EXPERIMENTS["A"]["target"]
    directory = profile_path("A", "target")
    metadata = load_profile_metadata(directory)
    validate_manifest(
        metadata,
        expected_plant=mapping["manifest_plant"],
        expected_profile=mapping["manifest_profile"],
        expected_profile_dir_name=mapping["profile_dir"],
    )
    scalers = validate_scalers(metadata)
    split = load_split(metadata, "test")
    validate_feature_transform_consistency(split, scalers.feature_scaler)
    validate_target_round_trip(split, scalers.target_scaler)
    sequences = build_sequences(split.X_normalized, split.y_normalized, split.timestamps)
    expected = EXPECTED_SEQUENCE_COUNTS[mapping["profile_dir"]]["test"]
    _require(sequences.count == expected == 2607, "Target Test sequence count mismatch")
    return FinalTestData(
        named=NamedSequenceSplit("test", sequences.X_seq, sequences.y_seq),
        split_data=split,
        sequences=sequences,
        target_scaler=scalers.target_scaler,
    )


def predict_authorized_test(
    authorization: SelectedTestAuthorization,
    test: FinalTestData,
) -> np.ndarray:
    _require(isinstance(authorization, SelectedTestAuthorization), "Missing Test authorization")
    _require(test.named.name == "test", "Authorized path accepts Test only")
    return np.asarray(
        authorization.checkpoint_model.model.predict(
            test.named.X_seq,
            batch_size=R25_BATCH_SIZE,
            verbose=0,
        )
    ).reshape(-1)


def build_test_outputs(
    test: FinalTestData,
    prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    true_normalized = np.asarray(test.named.y_seq, dtype=np.float64).reshape(-1)
    _require(len(predicted) == len(true_normalized) == 2607, "Test prediction count mismatch")
    expected_original = test.split_data.y_original[test.sequences.target_row_index]
    true_original = inverse_transform_original(
        true_normalized,
        test.target_scaler,
        expected_original=expected_original,
    )
    predicted_original = inverse_transform_original(predicted, test.target_scaler)
    frame = pd.DataFrame(
        {
            "sample_index": test.sequences.sample_index,
            "target_row_index": test.sequences.target_row_index,
            "target_timestamp": test.sequences.target_timestamp,
            "y_true_normalized": true_normalized,
            "y_pred_normalized": predicted,
            "y_true_original": true_original,
            "y_pred_original": predicted_original,
            "residual_original": true_original - predicted_original,
        }
    )
    return frame, compute_metrics(true_normalized, predicted), compute_metrics(true_original, predicted_original)


def compare_and_classify(
    partial: Mapping[str, float],
    baseline: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    deltas = {f"delta_{name.lower()}": float(partial[name] - baseline[name]) for name in ("MAE", "MSE", "RMSE", "R2")}
    mae_improved = deltas["delta_mae"] < 0
    mse_improved = deltas["delta_mse"] < 0
    rmse_improved = deltas["delta_rmse"] < 0
    r2_improved = deltas["delta_r2"] > 0
    if rmse_improved and mae_improved and r2_improved:
        outcome = "observed_positive"
        wording = "較明確之 Positive Transfer"
    elif rmse_improved and mae_improved and mse_improved and not r2_improved:
        outcome = "observed_partial_positive"
        wording = "誤差層面之部分正向遷移"
    elif not rmse_improved and not mae_improved and not r2_improved:
        outcome = "observed_negative"
        wording = "Negative Transfer"
    else:
        outcome = "observed_mixed"
        wording = "Mixed Result"
    comparison = {
        **deltas,
        "improvement_percent_mae": float((baseline["MAE"] - partial["MAE"]) / baseline["MAE"] * 100),
        "improvement_percent_mse": float((baseline["MSE"] - partial["MSE"]) / baseline["MSE"] * 100),
        "improvement_percent_rmse": float((baseline["RMSE"] - partial["RMSE"]) / baseline["RMSE"] * 100),
        "improvement_percent_r2": float((partial["R2"] - baseline["R2"]) / abs(baseline["R2"]) * 100),
        "mae_direction": "improved" if mae_improved else "deteriorated_or_equal",
        "mse_direction": "improved" if mse_improved else "deteriorated_or_equal",
        "rmse_direction": "improved" if rmse_improved else "deteriorated_or_equal",
        "r2_direction": "improved" if r2_improved else "deteriorated_or_equal",
    }
    classification = {"classification": outcome, "thesis_interpretation": wording, "rules_applied_to": "original_scale_test"}
    return comparison, classification


def _artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_final_target_test() -> PhaseEResult:
    """Perform the first and only final Candidate-B Test evaluation."""

    preflight = validate_phase_e_preflight()
    output_dir = PHASE_D_RUN_ROOT / "final_test"
    _require(not output_dir.exists(), f"Final Test already exists: {output_dir}")
    output_dir.mkdir(parents=False, exist_ok=False)
    manifest_path = output_dir / "final_test_manifest.json"
    manifest: dict[str, Any] = {
        "phase": "R2.5 Phase E",
        "status": "RUNNING",
        "strategy_id": PRIMARY_STRATEGY_ID,
        "phase_d_run_id": PHASE_D_RUN_ID,
        "phase_d_manifest_sha256": preflight.phase_d_manifest_sha256,
        "phase_d_selection_sha256": preflight.selection_sha256,
        "selected_checkpoint_path": str(preflight.selected_checkpoint),
        "selected_checkpoint_sha256": preflight.selected_checkpoint_sha256,
        "source_checkpoint_sha256": preflight.source_checkpoint_sha256,
        "target_test_accessed": False,
        "test_access_count": 0,
        "test_metrics_used_for_training": False,
        "test_metrics_used_for_selection": False,
        "post_test_tuning_allowed": False,
        "positive_transfer_evaluated": False,
        "failure": None,
    }
    _write_json_atomic(manifest_path, manifest)
    stage = "reload_and_authorize"
    try:
        reloaded = reload_best_checkpoint(preflight.selected_checkpoint)
        authorization = authorize_selected_test(
            SELECTION_PATH,
            strategy_id=PRIMARY_STRATEGY_ID,
            checkpoint_model=reloaded,
        )
        stage = "target_test_load"
        test = load_target_test_once()
        manifest.update({"target_test_accessed": True, "test_access_count": 1, "test_sequence_count": len(test.named.X_seq)})
        _write_json_atomic(manifest_path, manifest)

        stage = "authorized_test_evaluation"
        prediction = predict_authorized_test(authorization, test)
        frame, metrics_normalized, metrics_original = build_test_outputs(test, prediction)
        comparison, classification = compare_and_classify(metrics_original, preflight.baseline)

        predictions_path = output_dir / "predictions_test.csv"
        normalized_path = output_dir / "metrics_normalized.json"
        original_path = output_dir / "metrics_original.json"
        original_csv_path = output_dir / "metrics_original.csv"
        comparison_path = output_dir / "comparison_to_without_tl.csv"
        classification_path = output_dir / "transfer_classification.json"
        frame.to_csv(predictions_path, index=False)
        _write_json_atomic(normalized_path, metrics_normalized)
        _write_json_atomic(original_path, metrics_original)
        pd.DataFrame([metrics_original]).to_csv(original_csv_path, index=False)
        pd.DataFrame([{**preflight.baseline, **{f"partial_{key.lower()}": value for key, value in metrics_original.items()}, **comparison}]).to_csv(comparison_path, index=False)
        _write_json_atomic(classification_path, classification)
        artifacts = {
            path.name: _artifact_record(path, output_dir)
            for path in (predictions_path, normalized_path, original_path, original_csv_path, comparison_path, classification_path)
        }
        manifest.update(
            {
                "status": "PASS",
                "positive_transfer_evaluated": True,
                "normalized_metrics": metrics_normalized,
                "original_scale_metrics": metrics_original,
                "baseline_metrics": dict(preflight.baseline),
                "metric_deltas": comparison,
                "transfer_classification": classification,
                "artifacts": artifacts,
            }
        )
        _write_json_atomic(manifest_path, manifest)
        return PhaseEResult(output_dir, manifest_path, classification["classification"], manifest)
    except BaseException as exc:
        manifest.update(
            {
                "status": "FAIL",
                "failure": {"type": type(exc).__name__, "message": str(exc), "stage": stage, "traceback": traceback.format_exc()},
            }
        )
        try:
            _write_json_atomic(manifest_path, manifest)
        except Exception:
            pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, PhaseEError):
            exc.output_dir = output_dir
            raise
        raise PhaseEError(str(exc), output_dir=output_dir) from exc
