#!/bin/bash
# Submit all jobs of this phase.  Run from the project root: bash slurm/submit_ablations.sh
mkdir -p logs results
sbatch slurm/jobs/tr_product_toric-L8_depolarizing_xz_abl-nog.sh
sbatch slurm/jobs/tr_product_toric-L8_depolarizing_xz_abl-noshare.sh
sbatch slurm/jobs/tr_product_toric-L8_depolarizing_xz_abl-nooutmix.sh
sbatch slurm/jobs/tr_product_toric-L8_depolarizing_xz_abl-noler.sh
sbatch slurm/jobs/tr_product_toric-L8_depolarizing_xz_abl-N10.sh
sbatch slurm/jobs/tr_product_toric-L8_depolarizing_xz_abl-puniform.sh
sbatch slurm/jobs/tr_product_hgp-625_depolarizing_xz_abl-nog.sh
sbatch slurm/jobs/tr_product_hgp-625_depolarizing_xz_abl-noshare.sh
sbatch slurm/jobs/tr_product_hgp-625_depolarizing_xz_abl-nooutmix.sh
sbatch slurm/jobs/tr_product_hgp-625_depolarizing_xz_abl-noler.sh
sbatch slurm/jobs/tr_product_hgp-625_depolarizing_xz_abl-N10.sh
sbatch slurm/jobs/tr_product_hgp-625_depolarizing_xz_abl-puniform.sh
