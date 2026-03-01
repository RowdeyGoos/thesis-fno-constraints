#!/usr/bin/env python3
"""Select top sweep candidates from saved sweep trial logs.

Expected layout:
  <root_dir>/sweeps/<sweep_id>/<trial_name>/logs_best.txt
"""

import argparse
import json
import math
from pathlib import Path


def parse_logs_best(path: Path):
    metrics = {}
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, value = line.split(",", 1)
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics


def finite(x):
    return isinstance(x, (float, int)) and math.isfinite(float(x))


def main():
    parser = argparse.ArgumentParser(description="Rank constraint sweep candidates from local sweep outputs")
    parser.add_argument("--sweep_root", required=True, help="Path like experiments/sweeps/<sweep_id>")
    parser.add_argument("--top_k", type=int, default=5, help="How many top trials to print")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional output JSON path to write ranked candidates",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep root not found: {sweep_root}")

    candidates = []
    for trial_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()]):
        logs_best = trial_dir / "logs_best.txt"
        metrics = parse_logs_best(logs_best)
        if metrics is None:
            continue

        val_err = metrics.get("val_err")
        val_pde = metrics.get("val_pde_residual_norm")
        if not finite(val_err):
            continue

        # Missing PDE residual gets pushed to the back by using +inf.
        pde_key = float(val_pde) if finite(val_pde) else float("inf")

        candidates.append(
            {
                "trial": trial_dir.name,
                "path": str(trial_dir),
                "val_err": float(val_err),
                "val_pde_residual_norm": None if not finite(val_pde) else float(val_pde),
                "val_zero_mode_violation": (
                    None
                    if not finite(metrics.get("val_zero_mode_violation"))
                    else float(metrics.get("val_zero_mode_violation"))
                ),
                "best_val_loss": (
                    None
                    if not finite(metrics.get("best_val_loss"))
                    else float(metrics.get("best_val_loss"))
                ),
                "sort_key": [float(val_err), pde_key],
            }
        )

    candidates.sort(key=lambda c: (c["sort_key"][0], c["sort_key"][1]))

    print(f"Found {len(candidates)} ranked candidates in {sweep_root}")
    print("")

    for idx, cand in enumerate(candidates[: args.top_k], start=1):
        print(
            f"{idx}. {cand['trial']} | val_err={cand['val_err']:.6f} | "
            f"val_pde_residual_norm={cand['val_pde_residual_norm']} | "
            f"val_zero_mode_violation={cand['val_zero_mode_violation']}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = [{k: v for k, v in c.items() if k != "sort_key"} for c in candidates]
        out_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        print("")
        print(f"Wrote ranking JSON to: {out_path}")


if __name__ == "__main__":
    main()
