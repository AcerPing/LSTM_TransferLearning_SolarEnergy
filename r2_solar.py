"""Process bootstrap for the isolated Solar R2 smoke-training gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solar R2 corrected-R1 runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run the disposable R2.3 smoke gate")
    smoke.add_argument("--device", choices=("cpu",), default="cpu")
    smoke.add_argument("--seed", type=int, default=1234)
    smoke.add_argument("--keep-smoke", action="store_true")
    smoke.add_argument("--smoke-root", default=None)
    formal = subparsers.add_parser(
        "formal", help="run the R2.4 Experiment-A formal protocol"
    )
    formal.add_argument("--experiment", default="A")
    formal.add_argument("--device", default="cpu")
    formal.add_argument("--seed", type=int, default=1234)
    formal.add_argument("--run-id", default=None)
    return parser.parse_args()


def _apply_device_policy(device: str) -> None:
    if device != "cpu":
        raise ValueError("R2 supports only the approved CPU device policy")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _validate_hash_seed(seed: int) -> None:
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(seed):
        raise ValueError(
            "PYTHONHASHSEED must be set before process start to "
            f"{seed}; observed {hash_seed!r}"
        )


def main() -> int:
    args = _parse_args()
    try:
        # These gates must run before importing TensorFlow, Keras, or R2 runtime.
        _apply_device_policy(args.device)
        _validate_hash_seed(args.seed)
    except ValueError as exc:
        print(f"R2 bootstrap failed: {exc}.", file=sys.stderr)
        return 2

    if args.command == "smoke":
        from r2_helpers.solar_smoke import SmokeTrainingError, run_smoke

        try:
            result = run_smoke(
                seed=args.seed,
                device=args.device,
                keep_smoke=args.keep_smoke,
                requested_root=args.smoke_root,
            )
        except SmokeTrainingError as exc:
            print(f"R2.3 smoke failed: {exc}", file=sys.stderr)
            print(f"SMOKE_FAILURE_ROOT={exc.smoke_root}", file=sys.stderr)
            traceback.print_exc()
            return 1
        print("R2.3_SMOKE_GATE=PASS")
        print(f"SMOKE_ROOT={result.smoke_root}")
        print("SMOKE_RESULT_JSON=" + json.dumps(result.summary, ensure_ascii=False))
        return 0

    if args.command == "formal":
        if args.experiment != "A":
            print("R2.4 formal runner supports Experiment A only.", file=sys.stderr)
            return 2
        from r2_helpers.solar_formal import FormalRunError, run_formal

        try:
            result = run_formal(
                experiment=args.experiment,
                device=args.device,
                seed=args.seed,
                run_id=args.run_id,
            )
        except FormalRunError as exc:
            print(f"R2.4 formal run failed: {exc}", file=sys.stderr)
            if exc.run_root is not None:
                print(f"FORMAL_FAILURE_ROOT={exc.run_root}", file=sys.stderr)
            traceback.print_exc()
            return 1
        print("R2.4_FORMAL_RUN=PASS")
        print(f"FORMAL_ROOT={result.run_root}")
        return 0

    raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
