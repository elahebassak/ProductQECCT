#!/bin/bash
# Submit training runs as chains of 24 h jobs that auto-resume from last.pt (24 h jobs are scheduled
# much faster on Nibi than 72 h ones).  Chunks 1..N-1 train for up to 22 h and defer the final
# evaluation; the last chunk trains for up to 10 h and leaves >= 12 h for the final evaluation.
# A chunk that finds the run already finished exits immediately.
#
#   bash slurm/submit_chained.sh "tr_*_hgp-400_* tr_*_hgp-625_* tr_*_hgp-asym_*" 4
#
cd ~/ProductQECCT
mkdir -p logs results slurm/jobs_chained
PAT=${1:-"tr_*_hgp-400_* tr_*_hgp-625_* tr_*_hgp-asym_*"}
N=${2:-4}
GRES=${GRES:-}            # optional override, e.g. GRES=gpu:h100_3g.40gb:1 bash slurm/submit_chained.sh ...
INCLUDE_ABL=${INCLUDE_ABL:-0}   # ablation scripts (*_abl-*) are skipped unless INCLUDE_ABL=1
EXTRA=${EXTRA:-}          # extra train.py arguments appended to every chunk, e.g. EXTRA="--accum_steps 2"
for j in $(cd slurm/jobs && ls $PAT 2>/dev/null | grep -v -E "_24h|_c[0-9]+\.sh" | { if [ "$INCLUDE_ABL" = 1 ]; then cat; else grep -v "_abl-"; fi; }); do
  base=${j%.sh}; prev=""
  for c in $(seq 1 $N); do
    if [ $c -lt $N ]; then mh=22; fin=0; else mh=10; fin=1; fi
    o=slurm/jobs_chained/${base}_c${c}.sh
    sed -e 's/--time=72:00:00/--time=24:00:00/' \
        -e "s/--max_hours [0-9.]*/--max_hours $mh --final_on_timeout $fin/" \
        -e 's/--batch_size 64 /--batch_size 128 /' \
        -e "s/^#SBATCH --job-name=\(.*\)$/#SBATCH --job-name=\1_c$c/" slurm/jobs/$j > $o
    [ -n "$GRES" ] && sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=$GRES/" $o
    [ -n "$EXTRA" ] && sed -i "s/--out_root runs/--out_root runs $EXTRA/" $o
    if [ -z "$prev" ]; then id=$(sbatch --parsable $o); else id=$(sbatch --parsable --dependency=afterany:$prev $o); fi
    echo "$base  chunk $c/$N -> job $id"
    prev=$id
  done
done
