#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=cl_bposd_hgp-asym_depolarizing_xz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=<your-email>
#SBATCH --mail-type=END,FAIL

module load python/3.11
source ~/pqecct_env/bin/activate
cd ~/ProductQECCT
mkdir -p logs results
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

python -u eval_classical.py \
    --code hgp-asym --noise depolarizing --err_types xz --decoder bposd \
    --p_test 0.01:0.10:18 --samples 200000 --workers 16 \
    --out results/cl_bposd_hgp-asym_depolarizing_xz.json --osd_order 10
