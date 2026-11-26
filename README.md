# ProductQECCT

**Product-structured transformer decoding of hypergraph-product quantum codes.**
Code, trained decoders and complete results for

> E. Bassak and E. S. Sousa, *ProductQECCT: Product-Structured Transformer Decoding of Hypergraph-Product Quantum Codes*, manuscript under review, 2026.

Hypergraph-product (HGP) codes — the family that contains the toric and surface codes and the prototype of
product quantum LDPC codes — place their qubits and stabilizers on a grid whose 2×2 block layout
`[Q1 S_Z; S_X Q2]` is exactly the composite matrix that ProductMPT uses for classical product codes.
ProductQECCT exploits this: instead of global attention over every qubit and stabilizer (QECCT), it runs four
masked cross-attention processes along the rows and columns of the grid. The result is a decoder with
**1.2 M parameters independent of the code size** that has the **lowest logical error rate of all neural decoders
in every setting we tested** (toric codes L = 4–10, HGP codes with 400–3600 qubits), beats a correlation-aware
BP+OSD on mid-size HGP codes under depolarizing noise, and trains at 1600 and 3600 qubits in 22–27 A100-hours, where
QECCT would need 349 M / 1.76 B parameters and 88 / 236 A100-hours and CrossQECCT does not fit in 40 GB
(`results/benchmark_training.csv`). Inference latency, throughput and memory of every decoder on every code, measured
with one protocol, are in `results/benchmark_inference.csv`.

Trained checkpoints (all 56 runs): **Zenodo DOI [10.5281/zenodo.24324614](https://doi.org/10.5281/zenodo.24324614)** · complete results: [`results/`](results/)

---

## Contents

| Path | What it is |
|---|---|
| `pqecct/models/product_qecct.py` | **ProductQECCT** — grid embedding, factorized noise estimator, four row/column cross-attention processes, factorized output head |
| `pqecct/models/qecct.py` | QECCT baseline (Deep Quantum Error Correction, Choukroun & Wolf, AAAI 2024) re-implemented in the common interface |
| `pqecct/models/cross_qecct.py` | CrossQECCT baseline — CrossMPT adapted to QEC the same way as QECCT |
| `pqecct/models/common.py` | shared attention block (with the context LayerNorm that fixes the cross-attention collapse), soft logical parity, loss, metrics |
| `pqecct/codes.py`, `pqecct/gf2.py` | toric and HGP code construction, exact GF(2) linear algebra for logical operators |
| `pqecct/noise.py` | independent and depolarizing noise, syndromes, logical flips (GPU) |
| `train.py` | train any of the three decoders from scratch — auto-resume, periodic and final evaluation |
| `evaluate.py` | re-evaluate a checkpoint (more shots, finer grid, latency) |
| `eval_classical.py` | MWPM, BP, BP+OSD (separate CSS) and joint X/Y/Z BP+OSD on the same noise and metrics |
| `collect_results.py`, `report.py` | gather result files, print the paper's tables |
| `tools/` | `audit_runs.py` (release gate), `aggregate_results.py`, `export_run_metadata.py`, `export_checkpoints.py`, `zenodo_upload.py`, `make_results_xlsx.py`, `measure_latency.py`, `benchmark_inference.py` (latency, throughput and memory of every decoder on every code, one protocol), `benchmark_training_step.py` (training step time of every neural decoder on every code on one GPU) |
| `slurm/` | cluster scripts: environment setup, job generation, chained submission, cross-cluster resume |
| `tests/test_all.py` | correctness tests (CSS commutation, logical operators, grid adjacency ⇔ parity-check matrices, soft parity, metrics) |
| `results/` | every final result: `all_results.csv` / `.json`, `report_final.txt`, the Excel workbook, per-run metadata |
| `docs/make_paper_figures.py` | regenerates the figures of the paper from `results/` |

## Installation

Python ≥ 3.10 and PyTorch ≥ 2.1 (CUDA recommended).

```bash
git clone https://github.com/elahebassak/ProductQECCT.git && cd ProductQECCT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# if the ldpc wheel fails to build on your platform:
pip install "git+https://github.com/quantumgizmos/ldpc.git"
python tests/test_all.py            # 7 tests, ~1 min on CPU
```

On a Digital Research Alliance of Canada cluster use `bash slurm/setup_env.sh` instead (wheels from the Alliance
wheelhouse; works on Nibi, Narval and Fir).

## Quick start

```bash
# list the codes and their parameters
python -m pqecct.codes toric-L6 hgp-400 hgp-625 hgp-1600

# train ProductQECCT on the [[400,16]] HGP code under depolarizing noise (full protocol: 300 epochs x 1000 batches)
python train.py --model product --code hgp-400 --noise depolarizing --err_types xz \
    --epochs 300 --batches_per_epoch 1000 --batch_size 128 --p_test 0.01:0.10:9 --final_p_test 0.01:0.10:18 \
    --final_samples 500000 --out_root runs

# a 10-minute smoke test on a toric code
python train.py --model product --code toric-L4 --noise depolarizing --err_types xz --epochs 3 --batches_per_epoch 200 \
    --p_test 0.05:0.2:5 --final_samples 20000 --out_root runs_smoke

# the two baselines use the same command with --model qecct / --model crossqecct
# re-evaluate a checkpoint with more shots
python evaluate.py --ckpt runs/product_hgp-400_depolarizing_xz_N6_d128_s42/best.pt --p_test 0.01:0.10:18 --samples 500000

# classical decoders on identical noise and metrics
python eval_classical.py --code hgp-400 --noise depolarizing --err_types xz --decoder bposd      --osd_order 10 --p_test 0.01:0.10:18 --samples 200000
python eval_classical.py --code hgp-400 --noise depolarizing --err_types xz --decoder bposd_corr --osd_order 10 --p_test 0.01:0.10:18 --samples 200000 --schedule serial
python eval_classical.py --code toric-L8 --noise independent  --err_types x  --decoder mwpm --p_test 0.01:0.2:18 --samples 1000000
```

Codes: `toric-L{L}`; `hgp-400` = [[400,16]], `hgp-asym` = [[500,20]], `hgp-625` = [[625,25]], `hgp-1600` = [[1600,64]],
`hgp-3600` = [[3600,144]] (HGP of random (3,4)-regular LDPC codes with fixed seeds); any `hgp:<spec>:<spec>` or `.alist`
component codes. Noise: `--noise independent --err_types x` (X errors from Z-checks, DQEC's protocol) or
`--noise depolarizing --err_types xz` (joint X/Z decoding).

### Using a released checkpoint

```python
import torch
from pqecct.codes import get_code
from pqecct.models import build_model
import argparse
ck = torch.load("product_hgp-625_depolarizing_xz_N6_d128_s42.pt", map_location="cpu", weights_only=False)   # from Zenodo
args = argparse.Namespace(**ck["args"]); code = get_code(args.code)
model = build_model(args, code).eval(); model.load_state_dict(ck["model"])
# model(s_x, s_z) -> logits for (e_x, e_z); see evaluate.py for the sampling / metric loop
```

## Reproducing the paper

Every number in the paper is in `results/all_results.csv` (one row per decoder × code × noise × p) and in
`results/ProductQECCT_all_results.xlsx` (sheet *Paper tables* reproduces Tables E1–E4 with ratio formulas).
`python report.py` prints the tables from `results/`; `python tools/make_results_xlsx.py results/all_results.csv out.xlsx`
rebuilds the workbook; `python docs/make_paper_figures.py` and `python docs/make_benchmark_figures.py` regenerate the figures and the cost tables.

To re-run the experiments on a SLURM cluster:

```bash
python slurm/gen_jobs.py                      # writes one job script per run into slurm/jobs/ (edit CONFIG for account / GPU names)
CLUSTER=narval bash slurm/submit_cluster.sh "tr_product_toric-*" 3        # chained resumable jobs, 3 x 24 h
CLUSTER=fir    bash slurm/submit_cluster.sh "tr_qecct_hgp-625_*" 4
INCLUDE_ABL=1 CLUSTER=nibi bash slurm/submit_cluster.sh "tr_product_*_abl-*" 2
bash slurm/submit_classical_tuned.sh          # BP+OSD in four BP settings, joint BP+OSD, MWPM, BP
python tools/audit_runs.py                    # every run: reached epoch 300? which checkpoint was evaluated? shots, grid
python tools/aggregate_results.py && python report.py
```

`slurm/MIGRATION.md` documents moving partially trained runs between clusters (`slurm/pull_runs.sh`).

### Protocol (identical for all three neural decoders)

300 epochs × 1000 batches of 128 (64 / 32 for hgp-1600 / hgp-3600), Adam 5·10⁻⁴ with a 1000-step linear warm-up and
cosine decay to 10⁻⁶, gradient clipping 1.0, N = 6 layers, d = 128, 8 heads, seed 42, p drawn per sample from the
9-point training grid, loss weights (λ_BER, λ_LER, λ_g) = (0.5, 1, 0.5). The checkpoint with the lowest training
loss (re-weighted with the final loss weights, so that it is comparable across epochs when `--ler_warmup_epochs > 0`)
is evaluated on 5·10⁵ shots per error rate on an 18-point grid. Two metrics are reported: **LER** (DQEC-style
logical-class error rate) and **LER_strict** (logical flip *or* syndrome-inconsistent correction).

### Two lessons that are now enforced by the code

* **Checkpoint selection under loss warm-up.** With `--ler_warmup_epochs > 0` the raw training loss is lowest in the
  first epochs (the logical-loss term is not yet switched on), so "lowest training loss" selected an untrained
  checkpoint in our first large-code runs. `train.py` now selects on the loss re-weighted with the final weights.
* **An inert head in the reported runs.** The factorized output mixing `G + A G B^T` of ProductQECCT was initialized
  with *both* factors at zero in the paper's runs, which gives both a zero gradient, so the mixing never trained and the
  head was the plain per-cell linear readout in all 56 runs (the released checkpoints contain `A = B = 0`). The default
  (`--out_mix_init zero`) keeps that configuration so that every command in this README reproduces the paper's runs;
  `--out_mix_init identity` makes the mixing trainable (not evaluated in the paper).
* **No silent partial results.** A chained job that runs out of time before the last epoch writes
  `results_partial_epoch{N}.json`, never `results_final.json`, and `tools/audit_runs.py` flags it. Run the audit before
  trusting a results directory.

## Results at a glance (logical error rate, lower is better)

| setting | ProductQECCT | CrossQECCT | QECCT | best classical |
|---|---|---|---|---|
| toric L=10, depolarizing, p=0.10 | **4.63e-2** | 8.01e-2 | 8.87e-2 | 2.54e-2 (joint BP+OSD) |
| toric L=10, independent, p=0.05 | **1.87e-2** | 3.49e-2 | 3.39e-2 | 1.42e-2 (BP+OSD) |
| [[625,25]] HGP, depolarizing, p=0.02 | **8.3e-4** | 3.38e-3 | 4.01e-3 | 2.52e-3 (joint BP+OSD) |
| [[625,25]] HGP, independent, p=0.04 | **1.14e-1** | 1.83e-1 | 2.26e-1 | 7.18e-2 (BP+OSD) |
| [[1600,64]] HGP, depolarizing, p=0.02 | **4.1e-4** | — | — | 1.56e-3 (BP+OSD) |
| [[3600,144]] HGP, depolarizing, p=0.02 | **1.4e-4** | — | — | 2.8e-4 (BP+OSD) |

Parameters: ProductQECCT 1.21–1.26 M for every code; QECCT / CrossQECCT 1.2–54 M. Full tables: `results/`.

## Citation

```bibtex
@unpublished{bassak2026productqecct,
  title   = {ProductQECCT: Product-Structured Transformer Decoding of Hypergraph-Product Quantum Codes},
  author  = {Bassak, Elaheh and Sousa, Elvino S.},
  note    = {manuscript under review},
  year    = {2026}
}
```

## Acknowledgements

This research was enabled in part by support provided by the Digital Research Alliance of Canada
(clusters Nibi, Narval and Fir). The classical baselines use [`ldpc`](https://github.com/quantumgizmos/ldpc) and
[`PyMatching`](https://github.com/oscarhiggott/PyMatching). The architecture builds on
[DQEC](https://github.com/yoniLc/DQEC) (QECCT), [CrossMPT](https://github.com/seongjoon-park/CrossMPT) and ProductMPT.

## License

MIT — see [LICENSE](LICENSE).
