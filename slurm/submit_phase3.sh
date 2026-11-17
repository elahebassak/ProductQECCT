#!/bin/bash
# Submit all jobs of this phase.  Run from the project root: bash slurm/submit_phase3.sh
mkdir -p logs results
sbatch slurm/jobs/tr_product_hgp-1600_depolarizing_xz.sh
sbatch slurm/jobs/tr_crossqecct_hgp-1600_depolarizing_xz.sh
sbatch slurm/jobs/cl_bposd_hgp-1600_depolarizing_xz.sh
sbatch slurm/jobs/tr_product_hgp-3600_depolarizing_xz.sh
sbatch slurm/jobs/cl_bposd_hgp-3600_depolarizing_xz.sh
