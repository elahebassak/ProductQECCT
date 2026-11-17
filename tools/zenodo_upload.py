"""Upload the exported checkpoints (and the results files) to a Zenodo deposition and print the DOI.

    export ZENODO_TOKEN=...            # personal access token with deposit:write + deposit:actions
    python tools/zenodo_upload.py --dir release_checkpoints --extra results/all_results.csv results/all_results.json \
        --title "ProductQECCT: trained decoders and results" [--sandbox] [--publish]

Without --publish the deposition is left in draft state so that you can review it in the browser and press
Publish yourself (recommended). The DOI is reserved at creation and printed either way.
Large uploads use the deposition's file bucket (streaming PUT), so multi-GB files are fine."""
import argparse, json, os, sys, glob, requests

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--title", default="ProductQECCT: trained decoders and results")
ap.add_argument("--sandbox", action="store_true"); ap.add_argument("--publish", action="store_true")
ap.add_argument("--creators", nargs="*", default=["Bassak, Elaheh", "Sousa, Elvino S."])
ap.add_argument("--affiliation", default="University of Toronto")
ap.add_argument("--github", default="")
a = ap.parse_args()
tok = os.environ.get("ZENODO_TOKEN") or sys.exit("set ZENODO_TOKEN")
base = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
H = {"Authorization": f"Bearer {tok}"}
r = requests.post(f"{base}/deposit/depositions", json={}, headers=H); r.raise_for_status()
dep = r.json(); dep_id = dep["id"]; bucket = dep["links"]["bucket"]
doi = dep["metadata"].get("prereserve_doi", {}).get("doi", "?")
print(f"deposition {dep_id} created; reserved DOI: {doi}")
desc = ("Trained decoder checkpoints (model weights, fp32) and complete result files for the paper "
        "'ProductQECCT: Product-Structured Transformer Decoding of Hypergraph-Product Quantum Codes'. "
        "One checkpoint per run: ProductQECCT, QECCT and CrossQECCT on toric codes (L=4..10) and hypergraph-product "
        "codes ([[400,16]], [[500,20]], [[625,25]], [[1600,64]], [[3600,144]]) under independent and depolarizing noise, "
        "plus the ablation runs. Load with the accompanying code" + (f" ({a.github})" if a.github else "") + ".")
meta = {"metadata": {"title": a.title, "upload_type": "dataset", "description": desc,
                     "creators": [{"name": c, "affiliation": a.affiliation} for c in a.creators],
                     "keywords": ["quantum error correction", "hypergraph product codes", "neural decoder", "transformer", "qLDPC"],
                     "license": "cc-by-4.0", "access_right": "open"}}
if a.github:
    meta["metadata"]["related_identifiers"] = [{"identifier": a.github, "relation": "isSupplementTo", "scheme": "url"}]
r = requests.put(f"{base}/deposit/depositions/{dep_id}", json=meta, headers=H); r.raise_for_status()
files = sorted(glob.glob(os.path.join(a.dir, "*"))) + list(a.extra)
for f in files:
    name = os.path.basename(f); size = os.path.getsize(f) / 2**20
    with open(f, "rb") as fh:
        r = requests.put(f"{bucket}/{name}", data=fh, headers=H)
    r.raise_for_status(); print(f"uploaded {name} ({size:.1f} MB)")
if a.publish:
    r = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish", headers=H); r.raise_for_status()
    print("published:", r.json()["doi_url"])
else:
    print(f"draft left unpublished: review at {'https://sandbox.zenodo.org' if a.sandbox else 'https://zenodo.org'}/deposit/{dep_id} and press Publish")
print("DOI:", doi)
