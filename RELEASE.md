# Release runbook (GitHub repository + Zenodo checkpoints)

Everything below runs on the **Narval login node**, where `~/ProductQECCT` is the master copy (code, `runs/` with all
54 finished runs, `results/`). Replace `elahebassak` with your GitHub user name throughout.

## 0. Overlay the release files onto the master copy

```bash
cd ~ && unzip -q -o ProductQECCT_release.zip -d release_tmp && rsync -a --exclude runs --exclude logs --exclude results release_tmp/ProductQECCT/ ~/ProductQECCT/ && rm -rf release_tmp
cd ~/ProductQECCT && source ~/pqecct_env/bin/activate
python tests/test_all.py                              # 7 tests must pass
```

## 1. Gate: audit every run, rebuild the results files

```bash
python tools/audit_runs.py                            # must end with "all runs clean" (56 runs)
python collect_results.py >/dev/null && python report.py > results/report_final.txt
python tools/aggregate_results.py                     # results/all_results.{csv,json}  (expect 170 entries)
python tools/export_run_metadata.py                   # results/runs/<run>/ (args, history, test_epoch*, results_final)
python tools/make_results_xlsx.py results/all_results.csv results/ProductQECCT_all_results.xlsx
du -sh results                                        # a few tens of MB; fine for git
```

## 2. GitHub

Create an empty repository `ProductQECCT` (public, no README/license — we have them) at https://github.com/new.
Authenticate once on Narval with SSH:

```bash
ssh-keygen -t ed25519 -C "narval" -f ~/.ssh/id_ed25519_github -N ""      # if you don't already have a key
cat ~/.ssh/id_ed25519_github.pub                                         # add at https://github.com/settings/keys
printf "Host github.com\n  IdentityFile ~/.ssh/id_ed25519_github\n" >> ~/.ssh/config
ssh -T git@github.com                                                    # "Hi elahebassak! ..."
```

Then:

```bash
cd ~/ProductQECCT
sed -i 's/elahebassak/<your user name>/g' README.md CITATION.cff
git init -b main
git add .                                             # .gitignore excludes runs/, logs/, *.pt, release_checkpoints/
git status | head -20                                 # sanity check: no checkpoints, no logs
git -c user.name="Elaheh Bassak" -c user.email="e.bassak@mail.utoronto.ca" commit -m "ProductQECCT: code, results and paper sources"
git remote add origin git@github.com:elahebassak/ProductQECCT.git
git push -u origin main
```

(Set `git config --global user.name/user.email` once instead of the `-c` flags if you prefer.)

## 3. Zenodo (checkpoints)

```bash
python tools/export_checkpoints.py                    # release_checkpoints/<run>.pt, weights only (~1.5 GB total) + MANIFEST.json
```

Create a Zenodo account (https://zenodo.org, "Log in with GitHub" works) and a personal access token at
https://zenodo.org/account/settings/applications/ with scopes `deposit:write` and `deposit:actions`. Then:

```bash
export ZENODO_TOKEN=<token>
python tools/zenodo_upload.py --dir release_checkpoints --extra results/all_results.csv results/all_results.json results/ProductQECCT_all_results.xlsx \
    --github https://github.com/elahebassak/ProductQECCT
```

The script creates the deposition with the metadata (title, authors, description, CC-BY-4.0, link to the repository),
uploads every file, and prints the **reserved DOI** (`10.5281/zenodo.NNNNNNN`). It leaves the record as a draft:
open the printed URL, check the file list, press **Publish**. (Test first with `--sandbox` and a sandbox token from
https://sandbox.zenodo.org if you want a dry run.)

## 4. Put the DOI into the repository and the paper

```bash
cd ~/ProductQECCT
sed -i 's|ZENODO_DOI|10.5281/zenodo.NNNNNNN|' README.md
git commit -am "add Zenodo DOI" && git push
```

Then send the repository URL and the DOI back to the chat; they go into the two `\todo{URL}` placeholders of
the manuscript (title footnote and the reproducibility section; the manuscript is kept outside the public repository until publication).

## 5. Optional polish

* Add the GitHub topics `quantum-error-correction`, `qldpc`, `transformer`, `neural-decoder`.
* Enable "Zenodo ↔ GitHub" integration so that each GitHub release gets its own DOI for the *code* as well
  (https://zenodo.org/account/settings/github/), then create release `v1.0` on GitHub.
* Pin the exact package versions used (`pip freeze > requirements-lock.txt`) and commit it.
