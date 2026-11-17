#!/bin/bash
# Tuned classical baselines: separate (CSS) BP+OSD and depolarizing-aware joint BP+OSD ("bposd_corr"),
# each over a small BP grid (min-sum scaling {0.625, 1.0} x schedule {parallel, serial}), OSD-CS order 10.
# One CPU job per (code, noise) runs all variants sequentially; every variant is saved as its own model
# name, e.g. "bposd[ms1.0-serial]", so report.py lists them side by side.
cd ~/ProductQECCT
mkdir -p logs results slurm/jobs_classical
ACCOUNT=def-esousa
CODES=${CODES:-"toric-L4 toric-L6 toric-L8 toric-L10 hgp-400 hgp-asym hgp-625"}
for code in $CODES; do
  case $code in toric*) PD="0.05:0.2:18"; PI="0.01:0.2:18";; *) PD="0.01:0.10:18"; PI="0.01:0.10:18";; esac
  for noise in depolarizing independent; do
    if [ $noise = depolarizing ]; then et=xz; P=$PD; decs="bposd bposd_corr"; else et=x; P=$PI; decs="bposd"; fi
    name=clt_${code}_${noise}
    o=slurm/jobs_classical/$name.sh
    {
      echo "#!/bin/bash"
      echo "#SBATCH --account=$ACCOUNT"
      echo "#SBATCH --cpus-per-task=16"
      echo "#SBATCH --mem=32G"
      echo "#SBATCH --time=24:00:00"
      echo "#SBATCH --job-name=$name"
      echo "#SBATCH --output=logs/%x_%j.out"
      echo "#SBATCH --error=logs/%x_%j.err"
      echo "module load python/3.11"
      echo "source ~/pqecct_env/bin/activate"
      echo "cd ~/ProductQECCT"
      for d in $decs; do for ms in 0.625 1.0; do for sch in parallel serial; do
        echo "python -u eval_classical.py --code $code --noise $noise --err_types $et --decoder $d --osd_order 10 --ms_scaling $ms --schedule $sch --p_test $P --samples 200000 --workers 16 --tag '[ms$ms-$sch]' --out results/clt_${d}_ms${ms}_${sch}_${code}_${noise}_${et}.json"
      done; done; done
    } > $o
    echo "$name -> $(sbatch --parsable $o)"
  done
done
