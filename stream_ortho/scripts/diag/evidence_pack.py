"""Collect every headline number of the PIVOT-B study into one machine-readable pack.

Reads the JSONs produced by T3..T11 and writes ``results/week10-pivotb/evidence/``:

    tables.json   all tables, machine readable
    tables.md     the same tables as markdown (paste into the paper)
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _load(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _row(name, m, keys):
    return {"name": name, **{k: (float(m[k]) if isinstance(m.get(k), (int, float)) else None)
                             for k in keys}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/week10-pivotb")
    ap.add_argument("--out", default="results/week10-pivotb/evidence")
    a = ap.parse_args(argv)

    root = pathlib.Path(a.root)
    keys = ("lcr_fixed", "n_jump", "ra", "sei", "z_iqr_ratio", "pc",
            "layer_index_preserved", "owner_preserved", "z_shift_p95_abs",
            "height_median")
    pack = {"main": [], "e6": [], "e6_folds": [], "ablation": [], "e4": {}, "e2": None}

    # ---- main table: sequence x method ----
    for seq in ("ieu", "npu", "head"):
        for tag, label in ((f"{seq}/p1/pivotb_t3.json", "prod"),
                           (f"{seq}/p1/pivotb_hybrid.json", "hybrid0.1")):
            j = _load(root / tag)
            if not j:
                continue
            pack["main"].append(_row(f"{seq}/{label}/raw", j["raw"], keys))
            for mode, m in j["modes"].items():
                merged = {**m["final"], "pc": m.get("pc")}
                pack["main"].append(_row(f"{seq}/{label}/{mode}", merged, keys))

    # ---- DOF ladder ----
    for model in ("r1", "r4", "f4"):
        j = _load(root / f"ieu/p1/pivotb_{model}.json")
        if j:
            for mode, m in j["modes"].items():
                pack["ablation"].append(_row(f"ieu/{model}/{mode}", m["final"], keys))

    # ---- freeze ablation (owner_only vs unfrozen vs frozen) ----
    j = _load(root / "ieu/p1/pivotb_t3.json")
    if j:
        for mode, m in j["modes"].items():
            pack["ablation"].append(_row(f"ieu/freeze/{mode}", m["final"], keys))

    # ---- E6 ----
    ho = _load(root / "ieu/e6/heldout.json")
    if ho:
        for name, v in ho["even_odd"].items():
            for tag in ("raw", "frozen"):
                pack["e6"].append(_row(f"{name}/{tag}", v[tag], ("lcr_fixed", "ra", "sei",
                                                                 "z_iqr_ratio", "pc")))
        for f in ho.get("folds", []):
            for tag in ("raw", "frozen"):
                pack["e6_folds"].append(_row(f"fold{f['fold']}/{tag}", f[tag],
                                             ("lcr_fixed", "ra", "sei", "z_iqr_ratio", "pc")))
    nf = _load(root / "ieu/e6/e6_neighbor_fit.json")
    if nf:
        for name, v in nf["directions"].items():
            for cfg in ("prod", "nbr", "hybrid0.1", "hybrid1", "nbr_top"):
                if cfg in v:
                    m = v[cfg]
                    pack["e6"].append({"name": f"{name}/{cfg}",
                                       "pc_rel_median": m["pc_rel_median"],
                                       "pct_improved": m["pct_improved"],
                                       "min_pc_rel": m["min_pc_rel"],
                                       "lcr_fixed": m["lcr_fixed"],
                                       "ra": m["ra"], "sei": m["sei"]})
    sel = _load(root / "ieu/e6/e6_selector.json")
    if sel:
        for name, v in sel["directions"].items():
            pack["e6"].append({"name": f"{name}/selector",
                               "pc_rel_median": v["pc_rel_median"],
                               "pct_improved": v["pct_improved"],
                               "min_pc_rel": v["min_pc_rel"],
                               "fallback_frac": v["fallback_frac"],
                               "lcr_fixed": v["lcr_fixed"]})
    ref = _load(root / "ieu/e6/reference.json")
    if ref:
        pack["reference"] = ref.get("reference")

    # ---- E4 ----
    e4 = _load(root / "ieu/e4/e4_window.json")
    if e4:
        pack["e4"] = {"slope_ms_per_frame": e4["slope_ms_per_frame"],
                      "windows": e4["windows"], "gate": e4["gate"],
                      "assign_mb_int32": e4["assign_mb_int32"]}

    # ---- T13 product audit ----
    aud = _load(root / "ieu/audit/boundary_bias.json")
    if aud:
        pack["audit"] = {"tail": aud["tail"], "boundary_bias": aud["boundary_bias"]}

    # ---- E2 pilot ----
    e2 = _load("results/week10-e2/ieu/pilot.json")
    if e2:
        pack["e2"] = {"passed": e2["passed"], "failures": e2["failures"]}

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False))

    # ---- markdown ----
    md = ["# PIVOT-B evidence pack", ""]
    md.append("## Main table\n")
    def _f(v, spec):
        return ("{: %s}" % spec).format(v).strip() if isinstance(v, (int, float)) else "—"

    md.append("| seq/method/config | LCR_fixed | n_jump | RA | SEI | z_IQR | PC | "
              "layer_pres | owner_pres | |dz| p95 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in pack["main"]:
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("name", "?"), _f(r.get("lcr_fixed"), ".4f"), _f(r.get("n_jump"), ".0f"),
            _f(r.get("ra"), ".3f"), _f(r.get("sei"), ".2f"), _f(r.get("z_iqr_ratio"), ".3f"),
            _f(r.get("pc"), ".3f"), _f(r.get("layer_index_preserved"), ".4f"),
            _f(r.get("owner_preserved"), ".4f"), _f(r.get("z_shift_p95_abs"), ".3f")))
    md.append("\n## Ablation (DOF ladder + freeze)\n")
    md.append("| variant | LCR_fixed | n_jump | RA | SEI | z_IQR |")
    md.append("|---|---|---|---|---|---|")
    for r in pack["ablation"]:
        md.append("| {name} | {lcr_fixed:.4f} | {n_jump:.0f} | {ra:.3f} | {sei:.2f} | "
                  "{z_iqr_ratio:.3f} |".format(**r))
    def _fmt(v, spec, suffix=""):
        return ("{:%s}%s" % (spec, suffix)).format(v) if isinstance(v, (int, float)) else "—"

    md.append("\n## E6 held-out\n")
    md.append("| variant | PC rel median | % improved | worst | LCR_fixed | RA | SEI |")
    md.append("|---|---|---|---|---|---|---|")
    for r in pack["e6"]:
        md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("name", "?"), _fmt(r.get("pc_rel_median"), "+.1f", "%"),
            _fmt(r.get("pct_improved"), ".1f", "%"), _fmt(r.get("min_pc_rel"), "+.1f", "%"),
            _fmt(r.get("lcr_fixed"), ".4f"), _fmt(r.get("ra"), ".3f"),
            _fmt(r.get("sei"), ".2f")))
    md.append("\n## T13 product audit (frame-pair bias at owner boundaries)\n")
    if pack.get("audit"):
        b = pack["audit"]["boundary_bias"]
        t = pack["audit"]["tail"]
        md.append(f"tail |dz|>1 m: {100*t['tail_frac']:.1f}% of live cells "
                  f"(roof {100*t['tail_on_roof_frac']:.1f}% / veg {100*t['tail_on_veg_frac']:.1f}% "
                  f"/ ground {100*t['tail_on_ground_frac']:.1f}%); "
                  f"{100*t['tail_within_2cells_of_boundary_frac']:.1f}% within 2 cells of a boundary\n")
        md.append("| metric | raw | hybrid | reduction |")
        md.append("|---|---|---|---|")
        for k in ("boundary_bias_p95", "double_wall_rate", "frame_bias_std"):
            r_, h_ = b.get(k + "_raw"), b.get(k + "_hybrid")
            if r_ is not None and h_ is not None:
                md.append(f"| {k} | {r_:.4f} | {h_:.4f} | {100*(1-h_/r_):+.1f}% |")
        md.append(f"| median bias reduction | — | — | "
                  f"{b['median_bias_reduction_pct']:+.1f}% |")
        md.append(f"| robust weighted reduction | — | — | "
                  f"{b['weighted_bias_reduction_pct_robust']:+.1f}% |")

    md.append("\n## E4 streaming\n")
    if pack["e4"]:
        md.append(f"slope {pack['e4']['slope_ms_per_frame']:.2f} ms/frame; "
                  f"assign {pack['e4']['assign_mb_int32']:.0f} MB int32; "
                  f"gate {pack['e4']['gate']}\n")
        md.append("| N built | ms p50 | ms p95 | merge ms |")
        md.append("|---|---|---|---|")
        for k, v in pack["e4"]["windows"].items():
            md.append(f"| {k} | {v['ms_p50']:.1f} | {v['ms_p95']:.1f} | {v['merge_ms']:.0f} |")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print(f"main {len(pack['main'])} rows, ablation {len(pack['ablation'])}, "
          f"e6 {len(pack['e6'])}, e6 folds {len(pack['e6_folds'])}")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
