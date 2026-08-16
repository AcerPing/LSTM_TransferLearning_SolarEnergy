"""Process bootstrap for Solar R2.5 zero-epoch contract validation."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solar R2.5 Partial FT contract")
    parser.add_argument("command", choices=("contract",))
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
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

    from r2_config.solar_r25 import R25_OUTPUT_BASE
    from r2_helpers.solar_partial_ft import zero_epoch_contract_summary

    summary = zero_epoch_contract_summary(R25_OUTPUT_BASE)
    print("R2.5_ZERO_EPOCH_CONTRACT=PASS")
    print(json.dumps(dict(summary), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

