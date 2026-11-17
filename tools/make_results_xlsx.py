"""Build results/ProductQECCT_all_results.xlsx (paper tables with ratio formulas, full grids per code, raw rows).

    python tools/make_results_xlsx.py results/all_results.csv results/ProductQECCT_all_results.xlsx
"""
import csv, sys, collections
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

SRC, OUT = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(SRC)))
for r in rows:
    for k in ("p", "ler", "ler_strict", "ber"):
        r[k] = float(r[k]) if r[k] not in ("", None) else None
    r["samples"] = int(r["samples"]) if r["samples"] else None
    r["params"] = int(r["params"]) if r["params"] else None

F = "Arial"
def font(bold=False, size=10, color=None, italic=False):
    return Font(name=F, bold=bold, size=size, color=color, italic=italic)
HDR_FILL = PatternFill("solid", fgColor="1F3864"); HDR_FONT = Font(name=F, bold=True, color="FFFFFF", size=10)
SUB_FILL = PatternFill("solid", fgColor="D9E1F2"); OURS_FILL = PatternFill("solid", fgColor="E2EFDA")
thin = Side(style="thin", color="BFBFBF"); BORDER = Border(bottom=thin)
SCI = "0.000E+00"

NAMES = {"product": "ProductQECCT (ours)", "qecct": "QECCT (DQEC baseline)", "crossqecct": "CrossQECCT (CrossMPT-based baseline)",
         "mwpm": "MWPM (PyMatching)", "bp": "BP (min-sum, 100 it.)", "bposd": "BP+OSD-10, separate CSS (default setting: min-sum 0.625, parallel)",
         "bposd_corr": "BP+OSD-10, joint X/Y/Z (correlation-aware)"}
def nice(d):
    base, _, tag = d.partition("[")
    s = NAMES.get(base, base)
    if tag: s += " [" + tag
    return s

CODE_ORDER = ["toric-L4", "toric-L6", "toric-L8", "toric-L10", "hgp-400", "hgp-asym", "hgp-625", "hgp-1600", "hgp-3600"]
DEC_ORDER = ["mwpm", "bp", "bposd", "bposd[ms0.625-parallel]", "bposd[ms0.625-serial]", "bposd[ms1.0-parallel]", "bposd[ms1.0-serial]",
             "bposd_corr[ms0.625-parallel]", "bposd_corr[ms0.625-serial]", "bposd_corr[ms1.0-parallel]", "bposd_corr[ms1.0-serial]",
             "qecct", "crossqecct", "product", "product[abl-N10]", "product[abl-nog]", "product[abl-noler]", "product[abl-nooutmix]",
             "product[abl-noshare]", "product[abl-puniform]"]
CODE_INFO = {"toric-L4": "[[32,2,4]]", "toric-L6": "[[72,2,6]]", "toric-L8": "[[128,2,8]]", "toric-L10": "[[200,2,10]]",
             "hgp-400": "[[400,16]] HGP of (3,4)-regular 12x16 LDPC", "hgp-asym": "[[500,20]] HGP of 12x16 and 15x20 LDPC",
             "hgp-625": "[[625,25]] HGP of 15x20 LDPC", "hgp-1600": "[[1600,64]] HGP of 24x32 LDPC", "hgp-3600": "[[3600,144]] HGP of 36x48 LDPC"}

by = collections.defaultdict(dict)   # (decoder, code, noise) -> {p: row}
for r in rows:
    by[(r["decoder"], r["code"], r["noise"])][r["p"]] = r

def nearest(d, p0, tol=0.006):
    if not d: return None
    p = min(d, key=lambda x: abs(x - p0))
    return d[p] if abs(p - p0) <= tol else None

wb = Workbook()

# ---------------------------------------------------------------- README
ws = wb.active; ws.title = "README"
lines = [
    ("ProductQECCT — complete experimental results", True, 14),
    ("Product-structured message-passing transformers for decoding hypergraph-product quantum codes (E. Bassak, E. S. Sousa).", False, 10),
    ("", False, 10),
    ("Contents", True, 11),
    ("Paper tables  — the tables of the manuscript (toric codes, HGP codes, large codes, ablations) with ratio columns computed by formula.", False, 10),
    ("One sheet per code — full error-rate grid for every decoder: LER (DQEC-style logical error rate) and LER_strict (logical flip OR syndrome-inconsistent correction).", False, 10),
    ("Raw  — every (decoder, code, noise, p) record: LER, LER_strict, BER, shots, parameter count, source file.", False, 10),
    ("", False, 10),
    ("Protocol", True, 11),
    ("Neural decoders: 300 epochs x 1000 batches of 128 (batch 64 / 32 for hgp-1600 / hgp-3600), Adam 5e-4 with 1000-step warm-up and cosine decay, seed 42; final evaluation 500 000 shots per p on an 18-point grid.", False, 10),
    ("Evaluated checkpoint: lowest training loss (DQEC rule); for hgp-1600 and hgp-3600 (20-epoch LER-loss warm-up) the epoch-300 checkpoint. All runs audited for epoch 300, checkpoint epoch, shots and grid.", False, 10),
    ("Classical decoders: MWPM (PyMatching); BP min-sum 100 iterations; BP+OSD-CS order 10 (ldpc) in four BP settings [min-sum scaling x schedule] — 'bposd' without a tag = the default setting (min-sum scaling 0.625, parallel schedule); the Paper tables use the best of the four settings at each p, as in the manuscript; 200 000 shots (MWPM 1e6; hgp-1600/3600 1e5).", False, 10),
    ("Noise: independent = X errors decoded from Z-checks with flip probability p; depolarizing = X,Y,Z each with probability p/3, decoded jointly.", False, 10),
    ("", False, 10),
    ("Decoder names", True, 11),
] + [(f"{k}: {v}", False, 10) for k, v in NAMES.items()] + [
    ("product[abl-*]: ablations of ProductQECCT — N10 = 10 layers; nog = no initial estimator g; noler = no logical-error loss; nooutmix = no factorized output mixing; noshare = unshared A–D blocks; puniform = continuous p sampling.", False, 10),
    ("", False, 10),
    ("Codes", True, 11),
] + [(f"{k}: {v}", False, 10) for k, v in CODE_INFO.items()] + [
    ("", False, 10),
    (f"Generated from results/all_results.csv ({len(rows)} rows, {len(by)} decoder x code x noise entries).", False, 9),
]
for i, (t, b, sz) in enumerate(lines, 1):
    c = ws.cell(row=i, column=1, value=t); c.font = font(bold=b, size=sz); c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 150

# ---------------------------------------------------------------- Paper tables
pt = wb.create_sheet("Paper tables")
row = 1
def header(ws, r, cols, widths=None):
    for j, h in enumerate(cols, 1):
        c = ws.cell(row=r, column=j, value=h); c.font = HDR_FONT; c.fill = HDR_FILL; c.alignment = Alignment(horizontal="center", wrap_text=True)
def title(ws, r, t, note=None):
    ws.cell(row=r, column=1, value=t).font = font(bold=True, size=12)
    if note: ws.cell(row=r + 1, column=1, value=note).font = font(italic=True, size=9, color="595959")

def paper_block(r0, heading, codes, noise, ps, decoders, note):
    """rows: code x decoder; columns: LER at each p, LER_strict at each p, params, ratio-to-ours formulas."""
    title(pt, r0, heading, note); r = r0 + 2
    cols = ["code", "decoder"] + [f"LER @ p={p:g}" for p in ps] + [f"LER_strict @ p={p:g}" for p in ps] + ["params"] + [f"ratio to ours @ p={p:g}" for p in ps]
    header(pt, r, cols); r += 1
    for code in codes:
        ours_row = None; block_rows = []
        for d in decoders:
            grid = by.get((d, code, noise))
            if not grid: continue
            vals = [nearest(grid, p) for p in ps]
            if all(v is None for v in vals): continue
            pt.cell(row=r, column=1, value=code).font = font()
            pt.cell(row=r, column=2, value=nice(d)).font = font(bold=(d == "product"))
            for j, v in enumerate(vals):
                c = pt.cell(row=r, column=3 + j, value=(v["ler"] if v else None)); c.number_format = SCI; c.font = font()
                c2 = pt.cell(row=r, column=3 + len(ps) + j, value=(v["ler_strict"] if v else None)); c2.number_format = "0.000"; c2.font = font()
            pc = pt.cell(row=r, column=3 + 2 * len(ps), value=(vals[0] or {}).get("params") if any(vals) else None); pc.number_format = "#,##0"; pc.font = font()
            if d == "product":
                ours_row = r
                for j in range(1, len(cols) + 1): pt.cell(row=r, column=j).fill = OURS_FILL
            block_rows.append(r); r += 1
        if ours_row:
            for rr in block_rows:
                if rr == ours_row: continue
                for j in range(len(ps)):
                    col = get_column_letter(3 + j); oc = get_column_letter(3 + j)
                    c = pt.cell(row=rr, column=4 + 2 * len(ps) + j, value=f'=IF(AND(ISNUMBER({col}{rr}),ISNUMBER({oc}{ours_row}),{oc}{ours_row}>0),{col}{rr}/{oc}{ours_row},"")')
                    c.number_format = "0.00\"x\""; c.font = font(color="1F3864")
        for j in range(1, len(cols) + 1): pt.cell(row=r - 1, column=j).border = BORDER
    return r + 2

TORIC = ["toric-L4", "toric-L6", "toric-L8", "toric-L10"]; MID = ["hgp-400", "hgp-asym", "hgp-625"]; LARGE = ["hgp-1600", "hgp-3600"]
abl_dec = ["product", "product[abl-N10]", "product[abl-nog]", "product[abl-noler]", "product[abl-nooutmix]", "product[abl-noshare]", "product[abl-puniform]"]
# best-of-four BP settings for the paper tables (the plain 'bposd' entry in the CSV is the default setting)
def synth_best(prefix, key):
    """per-p minimum of LER over the four BP settings (as in the manuscript's BP+OSD rows)"""
    for code in CODE_ORDER:
        for noise in ("independent", "depolarizing"):
            cands = [d for d in DEC_ORDER if d.startswith(prefix + "[") and (d, code, noise) in by]
            if not cands: continue
            merged = {}
            for p in sorted(set(p for d in cands for p in by[(d, code, noise)])):
                vals = [by[(d, code, noise)][p] for d in cands if p in by[(d, code, noise)]]
                merged[p] = min(vals, key=lambda v: v["ler"])
            by[(key, code, noise)] = merged
    NAMES[key] = {"bposd_best": "BP+OSD-10, separate CSS — best of 4 BP settings per p", "bposd_corr_best": "BP+OSD-10, joint X/Y/Z (correlation-aware) — best of 4 BP settings per p"}[key]
synth_best("bposd", "bposd_best"); synth_best("bposd_corr", "bposd_corr_best")
main_dec = ["mwpm", "bp", "bposd_best", "bposd_corr_best", "qecct", "crossqecct", "product"]
row = paper_block(row, "Table E1a — toric codes, independent noise (X errors from Z-checks)", TORIC, "independent", [0.03, 0.05, 0.10], main_dec,
                  "LER = DQEC-style logical error rate; LER_strict = logical flip or syndrome-inconsistent correction. 'ratio to ours' = decoder LER / ProductQECCT LER (>1 means ProductQECCT is better). Values at the grid point nearest to the stated p.")
row = paper_block(row, "Table E1b — toric codes, depolarizing noise (joint X/Z decoding)", TORIC, "depolarizing", [0.08, 0.10, 0.15], main_dec,
                  "bposd_corr = correlation-aware joint BP+OSD (best of four BP settings).")
row = paper_block(row, "Table E2a — HGP codes, independent noise", MID, "independent", [0.02, 0.04, 0.06, 0.08], main_dec, None)
row = paper_block(row, "Table E2b — HGP codes, depolarizing noise", MID, "depolarizing", [0.02, 0.04, 0.06, 0.08], main_dec, None)
row = paper_block(row, "Table E3 — large HGP codes, depolarizing noise (ProductQECCT: epoch-300 checkpoint; batch 64 / 32)", LARGE, "depolarizing", [0.02, 0.04, 0.06, 0.08], ["bposd_best", "product"],
                  "Classical reference is the separate BP+OSD (1e5 shots); the joint decoder is impractical at this size.")
row = paper_block(row, "Table E4 — ablations of ProductQECCT, depolarizing noise", ["toric-L8", "hgp-625"], "depolarizing", [0.02, 0.04, 0.06, 0.08, 0.10, 0.15], abl_dec,
                  "toric-L8 uses p = 0.08/0.10/0.15, hgp-625 uses p = 0.02–0.08; cells outside a code's grid are blank. Ratio > 1 = variant worse than the full model.")
for j, w in enumerate([11, 40] + [13] * 30, 1): pt.column_dimensions[get_column_letter(j)].width = w
pt.freeze_panes = "C1"

# ---------------------------------------------------------------- one sheet per code: full grids
for code in CODE_ORDER:
    ws = wb.create_sheet(code)
    r = 1
    ws.cell(row=r, column=1, value=f"{code}  {CODE_INFO[code]}").font = font(bold=True, size=12); r += 2
    for noise in ("independent", "depolarizing"):
        decs = [d for d in DEC_ORDER if (d, code, noise) in by]
        if not decs: continue
        for metric, fmt in (("ler", SCI), ("ler_strict", "0.0000")):
            ws.cell(row=r, column=1, value=f"{noise} noise — {metric.upper()}").font = font(bold=True, size=11); r += 1
            header(ws, r, ["p"] + [nice(d) for d in decs]); r += 1
            ps = sorted(set(p for d in decs for p in by[(d, code, noise)]))
            for p in ps:
                c = ws.cell(row=r, column=1, value=p); c.number_format = "0.0000"; c.font = font()
                for j, d in enumerate(decs, 2):
                    v = by[(d, code, noise)].get(p)
                    if v is None:
                        v = nearest(by[(d, code, noise)], p, tol=1e-4)
                    cc = ws.cell(row=r, column=j, value=(v[metric] if v else None)); cc.number_format = fmt; cc.font = font()
                    if d == "product": cc.fill = OURS_FILL
                r += 1
            ws.cell(row=r, column=1, value="params").font = font(italic=True, size=9)
            for j, d in enumerate(decs, 2):
                anyv = next(iter(by[(d, code, noise)].values()))
                c = ws.cell(row=r, column=j, value=anyv.get("params")); c.number_format = "#,##0"; c.font = font(italic=True, size=9)
            r += 1
            ws.cell(row=r, column=1, value="shots").font = font(italic=True, size=9)
            for j, d in enumerate(decs, 2):
                anyv = next(iter(by[(d, code, noise)].values()))
                c = ws.cell(row=r, column=j, value=anyv.get("samples")); c.number_format = "#,##0"; c.font = font(italic=True, size=9)
            r += 2
    ws.column_dimensions["A"].width = 10
    for j in range(2, 25): ws.column_dimensions[get_column_letter(j)].width = 24
    ws.freeze_panes = "B1"

# ---------------------------------------------------------------- Raw
raw = wb.create_sheet("Raw")
cols = ["decoder", "code", "noise", "err_types", "params", "p", "ler", "ler_strict", "ber", "samples", "source"]
header(raw, 1, cols)
for i, rr in enumerate(sorted(rows, key=lambda x: (CODE_ORDER.index(x["code"]), x["noise"], DEC_ORDER.index(x["decoder"]) if x["decoder"] in DEC_ORDER else 99, x["p"])), 2):
    for j, k in enumerate(cols, 1):
        c = raw.cell(row=i, column=j, value=rr[k]); c.font = font(size=9)
        if k in ("ler", "ber"): c.number_format = SCI
        if k == "ler_strict": c.number_format = "0.0000"
        if k == "p": c.number_format = "0.0000"
        if k in ("params", "samples"): c.number_format = "#,##0"
tab = Table(displayName="AllResults", ref=f"A1:{get_column_letter(len(cols))}{len(rows) + 1}")
tab.tableStyleInfo = TableStyleInfo(name="TableStyleLight9", showRowStripes=True)
raw.add_table(tab); raw.freeze_panes = "A2"
for j, w in enumerate([28, 11, 13, 9, 12, 9, 12, 11, 12, 10, 70], 1): raw.column_dimensions[get_column_letter(j)].width = w

wb.save(OUT)
print("saved", OUT)
