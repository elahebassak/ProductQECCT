# Moving the remaining runs from Nibi to Narval + Fir

State on Nibi (27 Oct): ~180 queued jobs, **none running** — every chain head is `(Priority)`-blocked.
Everything below moves the 19 remaining runs to the two empty clusters and resumes the partially
trained ones from their Nibi checkpoints.

## Split

| cluster | GPU | runs | chunks (24 h each) |
|---|---|---|---|
| **Fir** | H100-80GB | QECCT hgp-625 ×2, CrossQECCT hgp-625 ×2, QECCT hgp-asym ×2, CrossQECCT hgp-asym ×2 | 4 |
| **Fir** | H100-80GB | ProductQECCT hgp-1600, hgp-3600 (Phase 3) | 6 |
| **Narval** | A100-40GB | ProductQECCT hgp-625 ×2 (depol. resumes at epoch 253), hgp-asym ×2 | 3 |
| **Narval** | A100-40GB | QECCT hgp-400 ×2, CrossQECCT hgp-400 ×2 (resume at epochs 60–280) | 4 |
| **Narval** | A100-40GB | QECCT toric-L10 depolarizing (training done; final evaluation only) | 1 |
| **Narval** | A100-40GB | ablations on toric-L8 and hgp-625 (12 runs), once the above are in | 2 |

Rationale: the hgp-625/asym baselines are the slow, memory-hungry ones (QECCT hgp-625 needed ~1 190 s/epoch
on a 40 GB MIG slice; a whole H100 should do ~500 s/epoch, i.e. ~2 days for 300 epochs); CrossQECCT hgp-625
needed 28.5 GiB with gradient accumulation and fits an H100 without it. Everything else is light.

## 0. On Nibi — cancel the queued jobs (nothing is running, so the files are static)

```bash
scancel -u <user>
squeue -u <user>          # should be empty
```

## 1. On each of Narval and Fir — code + environment (once per cluster)

Upload the package zip from your laptop, then:

```bash
ssh <user>@narval.alliancecan.ca      # and, separately, <user>@fir.alliancecan.ca
cd ~ && rm -rf ProductQECCT && unzip -q ProductQECCT_oct27.zip && cd ProductQECCT
bash slurm/setup_env.sh                # ~10 min; builds ldpc from GitHub, needs the login node's internet
python tests/test_all.py               # 7 tests
```

Narval's home quota is 50 GB and Fir's similar; the runs are small (< 5 GB in total), so home is fine.
If you prefer `/project/<your-allocation>/<user>`, put `ProductQECCT` there and symlink it into `~`.

## 2. Pull checkpoints and results from Nibi (per cluster, only the runs that cluster continues)

On **Narval**:
```bash
cd ~/ProductQECCT
bash slurm/pull_runs.sh "product_hgp-625_* product_hgp-asym_* qecct_hgp-400_* crossqecct_hgp-400_* qecct_toric-L10_depolarizing_*"
```
On **Fir**:
```bash
cd ~/ProductQECCT
bash slurm/pull_runs.sh "qecct_hgp-625_* crossqecct_hgp-625_* qecct_hgp-asym_* crossqecct_hgp-asym_* product_hgp-1600_* product_hgp-3600_*"
```
Each rsync asks for your Nibi password and a Duo push. The script ends by printing the epoch stored in each
`last.pt`; check that these match what Nibi had (e.g. product hgp-625 depolarizing: 253; QECCT hgp-400
independent: ~220; CrossQECCT hgp-400 independent: ~280). If the `nibi.alliancecan.ca` alias does not
resolve, run with `NIBI=<user>@nibi.sharcnet.ca`.

If `rsync` from the login node is refused, use Globus (nibi → narval / fir collections, folder
`ProductQECCT/runs` and `ProductQECCT/results`) and skip this step.

## 3. Submit

On **Fir**:
```bash
cd ~/ProductQECCT
sinfo -o "%G" | sort -u                                   # confirm the GRES name is gpu:h100 (else set GRES=...)
CLUSTER=fir bash slurm/submit_cluster.sh "tr_qecct_hgp-625_* tr_crossqecct_hgp-625_* tr_qecct_hgp-asym_* tr_crossqecct_hgp-asym_*" 4
CLUSTER=fir bash slurm/submit_cluster.sh "tr_product_hgp-1600_* tr_product_hgp-3600_*" 6
squeue -u <user>
```
On **Narval**:
```bash
cd ~/ProductQECCT
sinfo -o "%G" | sort -u                                   # gpu:a100:1 is the whole-GPU request
CLUSTER=narval bash slurm/submit_cluster.sh "tr_product_hgp-625_* tr_product_hgp-asym_*" 3
CLUSTER=narval bash slurm/submit_cluster.sh "tr_qecct_hgp-400_* tr_crossqecct_hgp-400_*" 4
CLUSTER=narval bash slurm/submit_cluster.sh "tr_qecct_toric-L10_depolarizing_*" 1
# ablations, after the above have started (12 runs, product model only)
INCLUDE_ABL=1 CLUSTER=narval bash slurm/submit_cluster.sh "tr_product_toric-L8_depolarizing_xz_abl-* tr_product_hgp-625_depolarizing_xz_abl-*" 2
squeue -u <user>
```
Notes
- `submit_cluster.sh` skips any run that already has `runs/<run>/results_final.json`, rewrites the GPU/memory
  request for the cluster, and chains N 24 h jobs; a chunk that finds the run finished exits at once.
- The `tr_*` scripts in `slurm/jobs/` are the originals (72 h, batch 64 for baselines); the chained copies written
  to `slurm/jobs_chained/` have batch 128, 24 h, and the cluster's GRES. Nothing in `slurm/jobs/` is modified.
- CrossQECCT hgp-625 on Fir needs no `--accum_steps` (80 GB); on Narval it would need `EXTRA="--accum_steps 2"`.
- Phase 3 (hgp-1600/3600) jobs already carry `--amp --ler_warmup_epochs 20` and batch 64/32.

## 4. Watch

```bash
tail -n 3 logs/tr_*_c1_*.out                     # first epochs; s/epoch and peak_mem are in each epoch line
ls runs/*/test_epoch*.json | sort -V | tail      # periodic evaluations
python report.py                                 # tables from whatever has finished
```
When a run finishes, `runs/<run>/results_final.json` appears and `python collect_results.py` adds it to `results/`.
Send me the `report.py` output (and the epoch lines with `s/epoch` / `peak_mem`) from each cluster and I will fill
the manuscript.

## 5. Afterwards

Copy `results/*.json` and the `runs/*/results_final.json` + `test_epoch*.json` files from both clusters back to one
place (your laptop or Nibi) so the paper's numbers have a single source; the checkpoints (`best.pt`) are what the
code release should include.
