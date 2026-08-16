"""Process bootstrap for Solar R2.5 contracts and diagnostic smoke runs."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solar R2.5 Partial FT runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("contract", "smoke", "formal-validation"):
        child = subparsers.add_parser(command)
        child.add_argument("--device", choices=("cpu",), default="cpu")
        child.add_argument("--seed", type=int, default=1234)
        if command == "smoke":
            child.add_argument("--run-id", default=None)
    return parser.parse_args()


def _apply_device_policy(device: str) -> None:
    if device != "cpu":
        raise ValueError("R2.5 supports only the approved CPU device policy")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _validate_hash_seed(seed: int) -> None:
    observed = os.environ.get("PYTHONHASHSEED")
    if observed != str(seed):
        raise ValueError(
            f"PYTHONHASHSEED must be {seed} before process start; observed {observed!r}"
        )


def main() -> int:
    args = _parse_args()
    try:
        _apply_device_policy(args.device)
        _validate_hash_seed(args.seed)
        if args.seed != 1234:
            raise ValueError("R2.5 approved seed is fixed at 1234")
    except ValueError as exc:
        print(f"R2.5 bootstrap failed: {exc}", file=sys.stderr)
        return 2

    if args.command == "contract":
        from r2_config.solar_r25 import R25_OUTPUT_BASE
        from r2_helpers.solar_partial_ft import zero_epoch_contract_summary

        summary = zero_epoch_contract_summary(R25_OUTPUT_BASE)
        print("R2.5_ZERO_EPOCH_CONTRACT=PASS")
        print(json.dumps(dict(summary), ensure_ascii=False, default=str))
        return 0

    if args.command == "smoke":
        from r2_helpers.solar_partial_smoke import PartialSmokeError, run_partial_smoke

        try:
            result = run_partial_smoke(run_id=args.run_id)
        except PartialSmokeError as exc:
            print(f"R2.5 Phase C smoke failed: {exc}", file=sys.stderr)
            if exc.smoke_root is not None:
                print(f"R2.5_SMOKE_FAILURE_ROOT={exc.smoke_root}", file=sys.stderr)
            return 1
        print("R2.5_PHASE_C_SMOKE=PASS")
        print(f"R2.5_SMOKE_ROOT={result.smoke_root}")
        print(f"R2.5_SMOKE_MANIFEST={result.manifest_path}")
        return 0

    if args.command == "formal-validation":
        from r2_helpers.solar_partial_formal import (
            PartialFormalError,
            run_partial_formal_validation,
        )

        try:
            result = run_partial_formal_validation()
        except PartialFormalError as exc:
            print(f"R2.5 Phase D formal validation failed: {exc}", file=sys.stderr)
            if exc.run_root is not None:
                print(f"R2.5_PHASE_D_FAILURE_ROOT={exc.run_root}", file=sys.stderr)
            return 1
        print("R2.5_PHASE_D_FORMAL_VALIDATION=PASS")
        print(f"R2.5_PHASE_D_RUN_ROOT={result.run_root}")
        print(f"R2.5_PHASE_D_MANIFEST={result.manifest_path}")
        print(f"R2.5_PHASE_D_SELECTION={result.selection_path}")
        return 0

    raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
