#!/bin/bash
# Run on the login node of the cluster that will CONTINUE the runs, from ~/ProductQECCT.  Pulls the run directories (checkpoints last.pt/best.pt, per-epoch log, periodic test JSONs) and all result files
# from Nibi, so that partially trained runs resume instead of restarting.  Nibi's runs/ may be a symlink into
# /project (--copy-links follows it).  You will be asked for your Nibi password / Duo MFA once per rsync.
#
#   bash slurm/pull_runs.sh "product_hgp-625_* product_hgp-asym_* qecct_hgp-400_* crossqecct_hgp-400_* qecct_toric-L10_depolarizing_*"
#
# Set NIBI=user@host if your Nibi user name differs or the alliancecan.ca alias does not resolve
# (alternatives: <user>@nibi.sharcnet.ca).
cd "$(dirname "$0")/.." || exit 1
set -f   # no shell globbing: patterns are rsync patterns, not local file names
NIBI=${NIBI:-<user>@nibi.alliancecan.ca}   # source cluster (user@host); set NIBI=user@narval.alliancecan.ca etc.
PAT=${1:?"patterns of run names under runs/, e.g. 'product_hgp-625_* qecct_hgp-400_*'"}
mkdir -p runs results
# result files of finished runs (small; needed by report.py on this cluster)
rsync -rltvzP --copy-links "$NIBI:ProductQECCT/results/" results/
# the run directories to continue
INC=""
for p in $PAT; do INC="$INC --include=/$p/ --include=/$p/**"; done
rsync -rltvzP --copy-links --prune-empty-dirs $INC --exclude='*' "$NIBI:ProductQECCT/runs/" runs/
echo
echo "pulled run directories:"; ls -d runs/*/ 2>/dev/null
echo "resume points (epoch in last.pt):"
for d in runs/*/; do
  [ -f "$d/last.pt" ] && python - "$d" <<'EOF'
import sys, torch
d = sys.argv[1]
ck = torch.load(d + "/last.pt", map_location="cpu", weights_only=False)
print(f"  {d}: epoch {ck.get('epoch', '?')}" + ("  [FINISHED]" if __import__('os').path.exists(d + '/results_final.json') else ""))
EOF
done
