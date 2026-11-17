#!/bin/bash
# Re-run the cross-attention models (crossqecct + product) on the toric codes with the fixed block,
# as 24 h jobs.  Run from ~/ProductQECCT.  Old crossqecct runs are moved to runs_old/.
cd ~/ProductQECCT
mkdir -p logs results runs_old
for d in runs/crossqecct_toric-* runs/product_toric-*; do [ -d "$d" ] && mv "$d" runs_old/; done
for m in product crossqecct; do for L in L4 L6 L8 L10; do for s in independent_x depolarizing_xz; do
  j=slurm/jobs/tr_${m}_toric-${L}_${s}.sh
  sed -e 's/--time=72:00:00/--time=24:00:00/' -e 's/--max_hours 69/--max_hours 22/' $j > ${j%.sh}_24h.sh
  sbatch ${j%.sh}_24h.sh
done; done; done
