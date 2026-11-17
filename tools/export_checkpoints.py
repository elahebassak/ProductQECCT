"""Export release checkpoints: model weights + args + code name only (no optimizer state), fp32, one file per run,
plus a manifest with sizes and sha256.  Run this on the machine that holds runs/, then upload the directory
to Zenodo with tools/zenodo_upload.py.

    python tools/export_checkpoints.py [--runs runs] [--out release_checkpoints] [--which best|last]

For the two large-code runs (hgp-1600, hgp-3600) the epoch-300 checkpoint (last.pt) is exported, matching the
checkpoint their reported results were evaluated with; for every other run best.pt is exported."""
import argparse, glob, hashlib, json, os
import torch

ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="release_checkpoints")
ap.add_argument("--which", default="auto", choices=["auto", "best", "last"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
manifest = []
for d in sorted(glob.glob(f"{a.runs}/*/")):
    r = d.rstrip("/").split("/")[-1]
    which = a.which
    if which == "auto":
        which = "last" if ("hgp-1600" in r or "hgp-3600" in r) else "best"
    src = d + f"{which}.pt"
    if not os.path.exists(src):
        print("skip (no checkpoint):", r); continue
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = dict(model=ck["model"], args=ck["args"], code=ck.get("code"), epoch=ck["epoch"], source=f"{which}.pt")
    path = f"{a.out}/{r}.pt"
    torch.save(out, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest.append(dict(run=r, file=os.path.basename(path), epoch=ck["epoch"], from_checkpoint=which,
                         size_mb=round(os.path.getsize(path) / 2**20, 1), sha256=h))
    print(f"{r}: epoch {ck['epoch']} ({which}.pt) -> {manifest[-1]['size_mb']} MB")
json.dump(manifest, open(f"{a.out}/MANIFEST.json", "w"), indent=1)
print(f"\n{len(manifest)} checkpoints, {sum(m['size_mb'] for m in manifest)/1024:.2f} GB -> {a.out}/ (MANIFEST.json written)")
