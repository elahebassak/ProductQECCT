#!/usr/bin/env python3
"""
Generate the SLURM job scripts for the full experimental matrix of the proposal.

    python slurm/gen_jobs.py            # writes slurm/jobs/*.sh and slurm/submit_phase{1,2,3}.sh

Edit the CONFIG block (account, e-mail, paths, GPU gres names) for your allocation.
Every training job trains *from scratch*, auto-resumes from <run>/last.pt if requeued, and stops
gracefully (then runs the final evaluation) `MAX_HOURS` before the SLURM time limit.
"""
import os
import textwrap

# ----------------------------------------------------------------------------- CONFIG
ACCOUNT = "<your-allocation>"
MAIL = "<your-email>"
PROJECT_DIR = "~/ProductQECCT"
VENV = "~/pqecct_env"
GRES_SMALL = "gpu:h100_3g.40gb:1"     # MIG slice: toric codes, hgp-400/625/asym for the product model
GRES_LARGE = "gpu:h100:1"             # full H100: hgp-1600/3600, and all QECCT/CrossQECCT HGP jobs
JOB_HOURS = 72
# ----------------------------------------------------------------------------- end CONFIG

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")
os.makedirs(JOBS, exist_ok=True)

HEADER = """\
#!/bin/bash
#SBATCH --account={account}
{gres_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hours}:00:00
#SBATCH --job-name={name}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user={mail}
#SBATCH --mail-type=END,FAIL

module load python/3.11
source {venv}/bin/activate
cd {project}
mkdir -p logs results
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
"""

# p grids (DQEC settings for toric codes; lower range for HGP-LDPC codes, around the BP+OSD pseudo-threshold)
P_GRID = {
    ("toric", "independent"): "0.01:0.2:9",
    ("toric", "depolarizing"): "0.05:0.2:9",
    ("hgp", "independent"): "0.01:0.10:9",
    ("hgp", "depolarizing"): "0.01:0.10:9",
}
P_FINAL = {
    ("toric", "independent"): "0.01:0.2:18",
    ("toric", "depolarizing"): "0.05:0.2:18",
    ("hgp", "independent"): "0.01:0.10:18",
    ("hgp", "depolarizing"): "0.01:0.10:18",
}


def fam(code):
    return "toric" if code.startswith("toric") else "hgp"


def train_job(model, code, noise, err_types, N_dec=6, d_model=128, h=8, epochs=300, bpe=1000, batch=128,
              large=False, extra="", suffix=""):
    name = f"tr_{model}_{code}_{noise}_{err_types}{suffix}"
    if suffix:
        extra = extra + f" --run_name {model}_{code}_{noise}_{err_types}_N{N_dec}_d{d_model}_s42{suffix}"
    gres = GRES_LARGE if large else GRES_SMALL
    body = HEADER.format(account=ACCOUNT, gres_line=f"#SBATCH --gres={gres}\n", cpus=4, mem="48G" if large else "32G",
                         hours=JOB_HOURS, name=name, mail=MAIL, venv=VENV, project=PROJECT_DIR)
    body += textwrap.dedent(f"""
    python -u train.py \\
        --model {model} --code {code} --noise {noise} --err_types {err_types} \\
        --N_dec {N_dec} --d_model {d_model} --h {h} \\
        --epochs {epochs} --batches_per_epoch {bpe} --batch_size {batch} \\
        --p_test {P_GRID[(fam(code), noise)]} --final_p_test {P_FINAL[(fam(code), noise)]} \\
        --test_every 20 --test_samples 20000 --final_samples 500000 \\
        --max_hours {JOB_HOURS - 3} --out_root runs {extra}
    """)
    path = os.path.join(JOBS, name + ".sh")
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, 0o755)
    return path


def classical_job(decoder, code, noise, err_types, samples=200000, workers=16, extra=""):
    name = f"cl_{decoder}_{code}_{noise}_{err_types}"
    body = HEADER.format(account=ACCOUNT, gres_line="", cpus=workers, mem="32G", hours=24, name=name, mail=MAIL,
                         venv=VENV, project=PROJECT_DIR)
    body += textwrap.dedent(f"""
    python -u eval_classical.py \\
        --code {code} --noise {noise} --err_types {err_types} --decoder {decoder} \\
        --p_test {P_FINAL[(fam(code), noise)]} --samples {samples} --workers {workers} \\
        --out results/{name}.json {extra}
    """)
    path = os.path.join(JOBS, name + ".sh")
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, 0o755)
    return path


def eval_job(model, code, noise, err_types, samples=1000000, large=False):
    """Re-evaluate a trained checkpoint with more samples (optional; train.py already does a 5e5-sample final eval)."""
    name = f"ev_{model}_{code}_{noise}_{err_types}"
    run = f"{model}_{code}_{noise}_{err_types}_N6_d128_s42"
    gres = GRES_LARGE if large else GRES_SMALL
    body = HEADER.format(account=ACCOUNT, gres_line=f"#SBATCH --gres={gres}\n", cpus=4, mem="32G", hours=12, name=name,
                         mail=MAIL, venv=VENV, project=PROJECT_DIR)
    body += textwrap.dedent(f"""
    python -u evaluate.py --ckpt runs/{run}/best.pt \\
        --p_test {P_FINAL[(fam(code), noise)]} --samples {samples} --batch_size 1024 \\
        --out results/{name}.json
    """)
    path = os.path.join(JOBS, name + ".sh")
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, 0o755)
    return path


def write_submit(name, paths):
    p = os.path.join(HERE, name)
    with open(p, "w") as f:
        f.write("#!/bin/bash\n# Submit all jobs of this phase.  Run from the project root: bash slurm/%s\n" % name)
        f.write("mkdir -p logs results\n")
        for x in paths:
            f.write(f"sbatch {os.path.relpath(x, os.path.dirname(HERE))}\n")
    os.chmod(p, 0o755)
    print(f"{name}: {len(paths)} jobs")


# ============================================================================ Phase 1: toric codes (DQEC parity)
phase1 = []
for L in (4, 6, 8, 10):
    code = f"toric-L{L}"
    for noise, et in (("independent", "x"), ("depolarizing", "xz")):
        for model in ("product", "qecct", "crossqecct"):
            phase1.append(train_job(model, code, noise, et))
        phase1.append(classical_job("mwpm", code, noise, et, samples=1000000))
        phase1.append(classical_job("bposd", code, noise, et, samples=200000, extra="--osd_order 10"))
write_submit("submit_phase1.sh", phase1)

# ============================================================================ Phase 2: mid-size HGP codes
phase2 = []
for code in ("hgp-400", "hgp-625", "hgp-asym"):
    for noise, et in (("independent", "x"), ("depolarizing", "xz")):
        phase2.append(train_job("product", code, noise, et))
        phase2.append(train_job("qecct", code, noise, et, large=True))
        phase2.append(train_job("crossqecct", code, noise, et, large=True))
        phase2.append(classical_job("bposd", code, noise, et, samples=200000, extra="--osd_order 10"))
        phase2.append(classical_job("bp", code, noise, et, samples=200000))
write_submit("submit_phase2.sh", phase2)

# ============================================================================ Phase 3: large HGP codes (scalability)
phase3 = []
for code in ("hgp-1600", "hgp-3600"):
    for noise, et in (("depolarizing", "xz"),):
        phase3.append(train_job("product", code, noise, et, large=True, batch=64 if code == "hgp-1600" else 32,
                                extra="--amp --ler_warmup_epochs 20"))
        if code == "hgp-1600":
            # global-attention baselines are expected to be at (or beyond) the memory limit here
            phase3.append(train_job("crossqecct", code, noise, et, large=True, batch=16, extra="--amp --g_hidden_mult 2"))
        phase3.append(classical_job("bposd", code, noise, et, samples=100000, extra="--osd_order 10"))
write_submit("submit_phase3.sh", phase3)

# ============================================================================ Ablations (toric-L8 & hgp-625, depolarizing)
abl = []
for code in ("toric-L8", "hgp-625"):
    abl.append(train_job("product", code, "depolarizing", "xz", extra="--no_g 1", suffix="_abl-nog"))
    abl.append(train_job("product", code, "depolarizing", "xz", extra="--no_share", suffix="_abl-noshare"))
    abl.append(train_job("product", code, "depolarizing", "xz", extra="--no_out_mix", suffix="_abl-nooutmix"))
    abl.append(train_job("product", code, "depolarizing", "xz", extra="--lam_ler 0", suffix="_abl-noler"))
    abl.append(train_job("product", code, "depolarizing", "xz", N_dec=10, suffix="_abl-N10"))
    abl.append(train_job("product", code, "depolarizing", "xz", extra="--p_sampling uniform", suffix="_abl-puniform"))
write_submit("submit_ablations.sh", abl)
print("job scripts written to", JOBS)
