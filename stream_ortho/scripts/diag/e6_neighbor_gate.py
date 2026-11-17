"""Re-apply the T8 gate to the stored metrics (no re-run).

The card's E6-B row says "至少 4/5 折 PC 改善" (improve), while E6-A says "↓≥20%".  The first
implementation applied the 20% bar to the folds as well, which fails a fold with +13.9%.
Both readings are reported here; nothing is re-measured.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)

    p = pathlib.Path(a.json)
    r = json.loads(p.read_text())
    dirs, folds = r["directions"], r.get("folds", [])
    out = {}
    for cfg in ("raw", "prod", "nbr", "hybrid0.1", "hybrid1", "nbr_top"):
        if cfg not in dirs[list(dirs)[0]]:
            continue
        if cfg == "raw":
            continue
        a20 = [dirs[d][cfg]["pc_rel_median"] >= 20 for d in dirs]
        imp = [dirs[d][cfg]["pct_improved"] >= 80 for d in dirs]
        worst = [dirs[d][cfg]["min_pc_rel"] >= -10 for d in dirs]
        lcr = [dirs[d][cfg]["lcr_fixed"] >= dirs[d]["raw"]["lcr_fixed"] - 0.002 for d in dirs]
        iqr = [0.95 <= dirs[d][cfg]["z_iqr_ratio"] <= 1.05 for d in dirs]
        f_imp = [f[cfg]["pc_rel_median"] > 0 for f in folds] if folds and all(cfg in f for f in folds) else []
        f_20 = [f[cfg]["pc_rel_median"] >= 20 for f in folds] if folds and all(cfg in f for f in folds) else []
        out[cfg] = {
            "E6A_both_ge20": bool(all(a20)),
            "E6B_folds_improved_4of5": bool(sum(f_imp) >= 4) if f_imp else None,
            "E6B_folds_ge20_4of5(strict reading)": bool(sum(f_20) >= 4) if f_20 else None,
            "pct_improved_ge80": bool(all(imp)),
            "worst_ge_minus10": bool(all(worst)),
            "lcr_ok": bool(all(lcr)),
            "z_iqr_ok": bool(all(iqr)),
            "detail": {"pc_rel_median": [dirs[d][cfg]["pc_rel_median"] for d in dirs],
                       "pct_improved": [dirs[d][cfg]["pct_improved"] for d in dirs],
                       "min_pc_rel": [dirs[d][cfg]["min_pc_rel"] for d in dirs],
                       "fold_pc_rel": [f[cfg]["pc_rel_median"] for f in folds] if f_imp else []},
        }
        core = [out[cfg][k] for k in ("E6A_both_ge20", "E6B_folds_improved_4of5",
                                      "pct_improved_ge80", "worst_ge_minus10", "lcr_ok", "z_iqr_ok")
                if out[cfg][k] is not None]
        out[cfg]["passed_card_wording"] = bool(all(core))
        strict = core + ([out[cfg]["E6B_folds_ge20_4of5(strict reading)"]] if f_20 else [])
        out[cfg]["passed_strict"] = bool(all(strict))

    res = {"source": str(p), "gate": out,
           "note": "E6-B uses 'at least 4/5 folds improve' (card wording); the strict "
                   "20%-per-fold reading is reported separately and is not the gate."}
    q = p.with_name("e6_neighbor_gate.json")
    q.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for cfg, g in out.items():
        print(f"{cfg:10s} card-wording {g['passed_card_wording']}  strict {g['passed_strict']}  "
              f"pc {['%.1f' % v for v in g['detail']['pc_rel_median']]}  "
              f"worst {['%.1f' % v for v in g['detail']['min_pc_rel']]}  "
              f"folds {['%.1f' % v for v in g['detail']['fold_pc_rel']]}")
    print("->", q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
