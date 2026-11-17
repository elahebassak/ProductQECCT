#!/bin/bash
# Submit all jobs of this phase.  Run from the project root: bash slurm/submit_phase2.sh
mkdir -p logs results
sbatch slurm/jobs/tr_product_hgp-400_independent_x.sh
sbatch slurm/jobs/tr_qecct_hgp-400_independent_x.sh
sbatch slurm/jobs/tr_crossqecct_hgp-400_independent_x.sh
sbatch slurm/jobs/cl_bposd_hgp-400_independent_x.sh
sbatch slurm/jobs/cl_bp_hgp-400_independent_x.sh
sbatch slurm/jobs/tr_product_hgp-400_depolarizing_xz.sh
sbatch slurm/jobs/tr_qecct_hgp-400_depolarizing_xz.sh
sbatch slurm/jobs/tr_crossqecct_hgp-400_depolarizing_xz.sh
sbatch slurm/jobs/cl_bposd_hgp-400_depolarizing_xz.sh
sbatch slurm/jobs/cl_bp_hgp-400_depolarizing_xz.sh
sbatch slurm/jobs/tr_product_hgp-625_independent_x.sh
sbatch slurm/jobs/tr_qecct_hgp-625_independent_x.sh
sbatch slurm/jobs/tr_crossqecct_hgp-625_independent_x.sh
sbatch slurm/jobs/cl_bposd_hgp-625_independent_x.sh
sbatch slurm/jobs/cl_bp_hgp-625_independent_x.sh
sbatch slurm/jobs/tr_product_hgp-625_depolarizing_xz.sh
sbatch slurm/jobs/tr_qecct_hgp-625_depolarizing_xz.sh
sbatch slurm/jobs/tr_crossqecct_hgp-625_depolarizing_xz.sh
sbatch slurm/jobs/cl_bposd_hgp-625_depolarizing_xz.sh
sbatch slurm/jobs/cl_bp_hgp-625_depolarizing_xz.sh
sbatch slurm/jobs/tr_product_hgp-asym_independent_x.sh
sbatch slurm/jobs/tr_qecct_hgp-asym_independent_x.sh
sbatch slurm/jobs/tr_crossqecct_hgp-asym_independent_x.sh
sbatch slurm/jobs/cl_bposd_hgp-asym_independent_x.sh
sbatch slurm/jobs/cl_bp_hgp-asym_independent_x.sh
sbatch slurm/jobs/tr_product_hgp-asym_depolarizing_xz.sh
sbatch slurm/jobs/tr_qecct_hgp-asym_depolarizing_xz.sh
sbatch slurm/jobs/tr_crossqecct_hgp-asym_depolarizing_xz.sh
sbatch slurm/jobs/cl_bposd_hgp-asym_depolarizing_xz.sh
sbatch slurm/jobs/cl_bp_hgp-asym_depolarizing_xz.sh
