#!/bin/bash
#SBATCH --account=def-esousa
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --job-name=tr_product_toric-L10_independent_x
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=e.bassak@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

module load python/3.11
source ~/pqecct_env/bin/activate
cd ~/ProductQECCT
mkdir -p logs results
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

python -u train.py \
    --model product --code toric-L10 --noise independent --err_types x \
    --N_dec 6 --d_model 128 --h 8 \
    --epochs 300 --batches_per_epoch 1000 --batch_size 128 \
    --p_test 0.01:0.2:9 --final_p_test 0.01:0.2:18 \
    --test_every 20 --test_samples 20000 --final_samples 500000 \
    --max_hours 69 --out_root runs 
