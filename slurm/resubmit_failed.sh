#!/bin/bash
# Re-submit only the jobs that FAILED (run from ~/ProductQECCT after unzipping the new code)
cd ~/ProductQECCT
mkdir -p logs results
for name in $(sacct -u elabass -S today -n --format=JobName%60,State | awk '$2=="FAILED"{print $1}' | sort -u); do
  if [ -f "slurm/jobs/$name.sh" ]; then echo "resubmitting $name"; sbatch "slurm/jobs/$name.sh"; fi
done
