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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command != "smoke":
        raise AssertionError(f"Unsupported command: {args.command}")

    # This must happen before importing solar_smoke, TensorFlow, or Keras.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(args.seed):
        print(
            "R2.3 bootstrap failed: PYTHONHASHSEED must be set before process "
            f"start to {args.seed}; observed {hash_seed!r}.",
            file=sys.stderr,
        )
        return 2

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


if __name__ == "__main__":
    raise SystemExit(main())
