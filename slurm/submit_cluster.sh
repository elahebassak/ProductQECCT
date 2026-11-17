#!/bin/bash
# Cluster-agnostic chained submission (Narval / Fir / Nibi).  Each run is a chain of N jobs (HOURS h each,
# default 24) that auto-resume from runs/<run>/last.pt.  Every chunk trains for up to HOURS-2 h; the final
# evaluation (5e5 shots) runs in whichever chunk reaches the last epoch.  If the last chunk runs out of time
# before the last epoch it evaluates anyway but writes results_partial_epoch{N}.json, never results_final.json,
# so an unfinished run can never be mistaken for a finished one (check with tools/audit_runs.py and resubmit).
# A chunk that finds runs/<run>/results_final.json exits immediately, so N can be generous.
#
#   CLUSTER=narval bash slurm/submit_cluster.sh "tr_product_hgp-625_* tr_product_hgp-asym_*" 3
#   CLUSTER=fir    bash slurm/submit_cluster.sh "tr_qecct_hgp-625_* tr_crossqecct_hgp-625_*" 4
#
# Environment overrides:  HOURS (chunk length), GRES (GPU request), MEM, CPUS, EXTRA (extra train.py args), INCLUDE_ABL=1, DRY=1
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs results slurm/jobs_chained
PAT=${1:?"pattern of slurm/jobs scripts, e.g. 'tr_qecct_hgp-625_*'"}
N=${2:-4}
HOURS=${HOURS:-24}
CLUSTER=${CLUSTER:-$(hostname -f | grep -oE 'narval|fir|nibi' | head -1)}
case "$CLUSTER" in
  narval) GRES=${GRES:-gpu:a100:1};  MEM=${MEM:-40G}; CPUS=${CPUS:-4} ;;   # A100-40GB (whole GPU)
  fir)    GRES=${GRES:-gpu:h100:1};  MEM=${MEM:-48G}; CPUS=${CPUS:-4} ;;   # H100-80GB (whole GPU)
  nibi)   GRES=${GRES:-gpu:h100_3g.40gb:1}; MEM=${MEM:-32G}; CPUS=${CPUS:-4} ;;
  *) echo "set CLUSTER=narval|fir|nibi"; exit 1 ;;
esac
echo "cluster=$CLUSTER gres=$GRES mem=$MEM cpus=$CPUS chunks=$N extra='$EXTRA'"
for j in $(cd slurm/jobs && ls $PAT 2>/dev/null | grep -v -E "_24h|_c[0-9]+\.sh" | { if [ "${INCLUDE_ABL:-0}" = 1 ]; then cat; else grep -v "_abl-"; fi; }); do
  base=${j%.sh}
  run=$(grep -oE -- "--run_name [^ ]+" slurm/jobs/$j | awk '{print $2}')
  [ -z "$run" ] && run=$(echo "$base" | sed -E 's/^tr_//')_N6_d128_s42
  if [ -f "runs/$run/results_final.json" ]; then echo "$base: already finished (runs/$run/results_final.json) - skipped"; continue; fi
  prev=""
  for c in $(seq 1 $N); do
    mh=$(( HOURS - 2 )); if [ $c -lt $N ]; then fin=0; else fin=1; fi
    o=slurm/jobs_chained/${base}_${CLUSTER}_c${c}.sh
    sed -e "s/--time=[0-9:]*/--time=${HOURS}:00:00/" \
        -e "s/--max_hours [0-9.]*/--max_hours $mh --final_on_timeout $fin/" \
        -e 's/--batch_size 64 /--batch_size 128 /' \
        -e "s/^#SBATCH --gres=.*/#SBATCH --gres=$GRES/" \
        -e "s/^#SBATCH --mem=.*/#SBATCH --mem=$MEM/" \
        -e "s/^#SBATCH --cpus-per-task=.*/#SBATCH --cpus-per-task=$CPUS/" \
        -e "s/^#SBATCH --job-name=\(.*\)$/#SBATCH --job-name=\1_c$c/" \
        -e 's/^module load python\/3.11$/module load StdEnv\/2023 python\/3.11 2>\/dev\/null || module load python\/3.11/' slurm/jobs/$j > $o
    [ -n "$EXTRA" ] && sed -i "s/--out_root runs/--out_root runs $EXTRA/" $o
    if [ "${DRY:-0}" = 1 ]; then echo "  would submit $o"; continue; fi
    if [ -z "$prev" ]; then id=$(sbatch --parsable $o); else id=$(sbatch --parsable --dependency=afterany:$prev $o); fi
    echo "$base  chunk $c/$N -> job $id"
    prev=$id
  done
done
