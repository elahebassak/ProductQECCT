#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=cl_mwpm_toric-L4_independent_x
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
    --code toric-L4 --noise independent --err_types x --decoder mwpm \
    --p_test 0.01:0.2:18 --samples 1000000 --workers 16 \
    --out results/cl_mwpm_toric-L4_independent_x.json 
