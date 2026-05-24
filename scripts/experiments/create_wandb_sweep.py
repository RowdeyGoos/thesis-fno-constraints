#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Create a W&B sweep from a YAML config and print SWEEP_ID=<id>."""

import argparse
import os

import wandb
from ruamel.yaml import YAML


def parse_args():
    parser = argparse.ArgumentParser(description="Create a W&B sweep from YAML")
    parser.add_argument("--sweep_yaml", required=True, help="Path to sweep YAML")
    parser.add_argument("--entity", default=None, help="Override W&B entity")
    parser.add_argument("--project", default=None, help="Override W&B project")
    return parser.parse_args()


def main():
    args = parse_args()

    yaml = YAML(typ="safe")
    with open(args.sweep_yaml, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.load(f)

    entity = args.entity or os.environ.get("WANDB_ENTITY") or sweep_cfg.get("entity")
    project = args.project or os.environ.get("WANDB_PROJECT") or sweep_cfg.get("project")

    sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)
    print(f"SWEEP_ID={sweep_id}")


if __name__ == "__main__":
    main()
