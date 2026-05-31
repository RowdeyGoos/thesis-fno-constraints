#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""
Evaluate Helmholtz zero-shot transfer with a scales-path ablation.

This script isolates whether zero-shot performance degradation is caused by using
downstream-domain normalization scales during evaluation. It evaluates the same
zero-shot checkpoints twice:
  1) with the default downstream scales from the zero-shot config
  2) with the source pretraining scales

It covers:
  - single-domain Helmholtz pretraining (o in [1,10]) -> downstream o in [1,5]
  - mixed-domain pretraining -> downstream Helmholtz o in [1,5] (mixed format)

Run from the repository root, e.g.:
    python scripts/entrypoints/eval_helmholtz_zeroshot_scales_ablation.py --device cuda:0
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from utils import logging_utils
from utils.YParams import YParams
from utils.inferencer import Inferencer


logging_utils.config_logger()


def _as_float(x):
    if torch.is_tensor(x):
        return float(x.item())
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def _evaluate_with_overrides(
    yaml_config: str,
    config_name: str,
    checkpoint_path: str,
    scales_path: str,
    device: str,
    root_dir: str,
    run_tag: str,
) -> Dict[str, Any]:
    logging.info("\n%s", "=" * 72)
    logging.info("Evaluating %s (%s)", config_name, run_tag)
    logging.info("YAML: %s", yaml_config)
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("scales_path override: %s", scales_path)
    logging.info("%s", "=" * 72)

    class Args:
        def __init__(self):
            self.yaml_config = yaml_config
            self.config = config_name
            self.root_dir = root_dir
            self.run_num = run_tag
            self.sweep = "none"
            self.weights = checkpoint_path

    args = Args()

    params = YParams(os.path.abspath(yaml_config), config_name)
    params["weights"] = checkpoint_path
    params["scales_path"] = scales_path

    # Inferencer decides CPU/GPU internally from torch availability, but we retain
    # the CLI argument for provenance in the saved results.
    params["requested_device"] = device

    inferencer = Inferencer(params, args)
    inferencer.launch()

    metrics = {
        "test_error": _as_float(inferencer.logs.get("test_err", np.nan)),
        "test_loss": _as_float(inferencer.logs.get("test_loss", np.nan)),
        "test_zero_mode_constraint_loss": _as_float(
            inferencer.logs.get("test_zero_mode_constraint_loss", np.nan)
        ),
        "test_pde_residual_norm": _as_float(
            inferencer.logs.get("test_pde_residual_norm", np.nan)
        ),
        "test_zero_mode_violation": _as_float(
            inferencer.logs.get("test_zero_mode_violation", np.nan)
        ),
    }
    return metrics


def _load_config_value(yaml_config: str, config_name: str, key: str) -> Any:
    params = YParams(os.path.abspath(yaml_config), config_name)
    if key not in params:
        raise KeyError(f"{config_name} in {yaml_config} has no '{key}' field")
    return params[key]


def _relative_or_abs(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.is_absolute() else p)


def _safe_ratio(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return float(a / b)


def main():
    parser = argparse.ArgumentParser(
        description="Helmholtz zero-shot scales-path ablation (downstream scales vs pretrain scales)"
    )
    parser.add_argument(
        "--helm_yaml",
        type=str,
        default="config/operators_helmholtz.yaml",
        help="Helmholtz transfer config YAML",
    )
    parser.add_argument(
        "--mixed_yaml",
        type=str,
        default="config/operators_mixed.yaml",
        help="Mixed pretraining config YAML (for mixed pretrain scales_path lookup)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Requested device label (recorded for provenance)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/transfer_learning_helmholtz_o1_5/zeroshot_scales_ablation",
        help="Directory to save JSON results",
    )
    parser.add_argument(
        "--eval_root_dir",
        type=str,
        default="./evaluation_tmp_zeroshot_scales_ablation",
        help="Temporary root dir for inferencer log outputs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "label": "single_domain",
            "display_name": "Helmholtz pretrain (o1_10) -> Helmholtz downstream (o1_5)",
            "eval_yaml": args.helm_yaml,
            "zeroshot_config": "helm-o1_5-zeroshot",
            "pretrain_yaml": args.helm_yaml,
            "pretrain_config": "helm-scale-o1_10",
        },
        {
            "label": "mixed_domain",
            "display_name": "Mixed pretrain -> Helmholtz downstream (o1_5 mixed format)",
            "eval_yaml": args.helm_yaml,
            "zeroshot_config": "helm-o1_5-zeroshot-mixed",
            "pretrain_yaml": args.mixed_yaml,
            "pretrain_config": "mixed-scale-all",
        },
    ]

    results: Dict[str, Any] = {
        "meta": {
            "script": "scripts/entrypoints/eval_helmholtz_zeroshot_scales_ablation.py",
            "helm_yaml": args.helm_yaml,
            "mixed_yaml": args.mixed_yaml,
            "requested_device": args.device,
        },
        "experiments": {},
    }

    for exp in experiments:
        label = exp["label"]
        eval_yaml = exp["eval_yaml"]
        zeroshot_cfg = exp["zeroshot_config"]
        pretrain_yaml = exp["pretrain_yaml"]
        pretrain_cfg = exp["pretrain_config"]

        checkpoint_path = _load_config_value(eval_yaml, zeroshot_cfg, "weights")
        downstream_scales = _load_config_value(eval_yaml, zeroshot_cfg, "scales_path")
        pretrain_scales = _load_config_value(pretrain_yaml, pretrain_cfg, "scales_path")
        test_path = _load_config_value(eval_yaml, zeroshot_cfg, "test_path")

        exp_out = {
            "display_name": exp["display_name"],
            "zeroshot_config": zeroshot_cfg,
            "pretrain_config_for_scales": pretrain_cfg,
            "paths": {
                "checkpoint_path": _relative_or_abs(checkpoint_path),
                "test_path": _relative_or_abs(test_path),
                "downstream_scales_path": _relative_or_abs(downstream_scales),
                "pretrain_scales_path": _relative_or_abs(pretrain_scales),
            },
            "exists": {
                "checkpoint_path": Path(checkpoint_path).exists(),
                "test_path": Path(test_path).exists(),
                "downstream_scales_path": Path(downstream_scales).exists(),
                "pretrain_scales_path": Path(pretrain_scales).exists(),
            },
            "runs": {},
        }

        required_ok = all(exp_out["exists"].values())
        if not required_ok:
            logging.warning("Skipping %s because one or more required files are missing", label)
            results["experiments"][label] = exp_out
            continue

        try:
            downstream_metrics = _evaluate_with_overrides(
                yaml_config=eval_yaml,
                config_name=zeroshot_cfg,
                checkpoint_path=checkpoint_path,
                scales_path=downstream_scales,
                device=args.device,
                root_dir=args.eval_root_dir,
                run_tag=f"{label}-downstream-scales",
            )
            pretrain_metrics = _evaluate_with_overrides(
                yaml_config=eval_yaml,
                config_name=zeroshot_cfg,
                checkpoint_path=checkpoint_path,
                scales_path=pretrain_scales,
                device=args.device,
                root_dir=args.eval_root_dir,
                run_tag=f"{label}-pretrain-scales",
            )

            exp_out["runs"]["downstream_scales"] = downstream_metrics
            exp_out["runs"]["pretrain_scales"] = pretrain_metrics

            d_err = float(downstream_metrics.get("test_error", np.nan))
            p_err = float(pretrain_metrics.get("test_error", np.nan))
            exp_out["comparison"] = {
                "test_error_delta_pretrain_minus_downstream": float(p_err - d_err)
                if not (np.isnan(p_err) or np.isnan(d_err))
                else float("nan"),
                "test_error_ratio_pretrain_over_downstream": _safe_ratio(p_err, d_err),
                "test_error_ratio_downstream_over_pretrain": _safe_ratio(d_err, p_err),
                "improvement_percent_using_pretrain_scales": float(((d_err - p_err) / d_err) * 100.0)
                if d_err and not (np.isnan(p_err) or np.isnan(d_err))
                else float("nan"),
            }

        except Exception as exc:  # keep going so one failure doesn't hide the other
            logging.exception("Failed ablation for %s: %s", label, exc)
            exp_out["error"] = str(exc)

        results["experiments"][label] = exp_out

    output_path = output_dir / "helmholtz_zeroshot_scales_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Saved ablation results to %s", output_path)

    print("\n" + "=" * 96)
    print("HELMHOLTZ ZERO-SHOT SCALES ABLATION SUMMARY")
    print("=" * 96)
    for label, exp_out in results["experiments"].items():
        print(f"\n[{label}] {exp_out.get('display_name', '')}")
        exists = exp_out.get("exists", {})
        if exists and not all(exists.values()):
            print("  Missing files:")
            for k, v in exists.items():
                print(f"    {k}: {'OK' if v else 'MISSING'}")
            continue
        if "error" in exp_out:
            print(f"  Error: {exp_out['error']}")
            continue

        d = exp_out["runs"]["downstream_scales"]["test_error"]
        p = exp_out["runs"]["pretrain_scales"]["test_error"]
        c = exp_out.get("comparison", {})
        print(f"  Zero-shot error (downstream scales): {d:.6e}")
        print(f"  Zero-shot error (pretrain scales):   {p:.6e}")
        print(
            "  Improvement using pretrain scales:  "
            f"{c.get('improvement_percent_using_pretrain_scales', float('nan')):+.2f}%"
        )
        print(
            "  Ratio downstream/pretrain:          "
            f"{c.get('test_error_ratio_downstream_over_pretrain', float('nan')):.3f}x"
        )
    print("\nSaved JSON:", output_path)
    print("=" * 96)


if __name__ == "__main__":
    main()
