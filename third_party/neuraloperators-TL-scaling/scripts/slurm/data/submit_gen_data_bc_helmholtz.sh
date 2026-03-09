#!/usr/bin/env bash
#SBATCH --job-name=bc-data-helmholtz
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=11:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="helmholtz"
bash "${SCRIPT_DIR}/submit_gen_data_bc.sh"
