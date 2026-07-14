"""
transdiagnostic_weight_02b_metabolic.py
SECONDARY study-specific metabolic outcomes (glucose, lipids, waist) for the
3 studies that collected them: CATIE, ACLAIMS, LiTMUS. No cross-class pooling
(all 3 are antipsychotic or mood-stabilizer) - results are reported per study.

Units (verified): glucose & lipids in mg/dL in all 3 studies; waist recorded in
INCHES everywhere -> converted to cm. Change = last available value - baseline.

Outputs (results/transdiagnostic_weight/):
  metabolic_long.parquet, metabolic_results.pkl, metabolic_summary.xlsx,
  metabolic_stats.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from xlsx_format import write_pretty_sheet
from scipy import stats

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "transdiagnostic_weight"

IN_TO_CM = 2.54
# plausible ranges to drop data-entry errors
RANGES = {
    "glucose": (40, 600), "total_cholesterol": (50, 500), "hdl": (10, 150),
    "ldl": (20, 400), "triglycerides": (20, 2000), "waist_cm": (50, 180),
}
MARKER_LABEL = {
    "glucose": "Glucose (mg/dL)", "total_cholesterol": "Total cholesterol (mg/dL)",
    "hdl": "HDL cholesterol (mg/dL)", "ldl": "LDL cholesterol (mg/dL)",
    "triglycerides": "Triglycerides (mg/dL)", "waist_cm": "Waist circumference (cm)",
}
BASELINE_MAX_MONTH = 1.0   # a measurement at/before ~1 month counts as baseline


def clip_range(s, marker):
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = RANGES[marker]
    return x.where((x >= lo) & (x <= hi))


# ---------------------------------------------------------------------------
# Build unified long: subject_id, study, marker, time_months, value
# ---------------------------------------------------------------------------
def long_catie():
    frames = []
    lab = pd.read_parquet(DATA / "parquet_catie/LAB.parquet")
    name_map = {"GLUCOSE": "glucose", "TOTAL CHOLESTEROL": "total_cholesterol",
                "HDL CHOLESTEROL": "hdl", "TRIGLYCERIDES": "triglycerides"}
    for raw, marker in name_map.items():
        sub = lab[lab["Laboratory Test Name"] == raw]
        frames.append(pd.DataFrame({
            "subject_id": sub["NIMHID"].astype(str),
            "marker": marker,
            "time_months": pd.to_numeric(sub["visit_month"], errors="coerce"),
            "value": clip_range(sub["Result Numeric Value"], marker),
        }))
    vit = pd.read_parquet(DATA / "parquet_catie/VITAL.parquet")
    frames.append(pd.DataFrame({
        "subject_id": vit["NIMHID"].astype(str), "marker": "waist_cm",
        "time_months": pd.to_numeric(vit["visit_month"], errors="coerce"),
        "value": clip_range(pd.to_numeric(vit["Waist Circumference"], errors="coerce") * IN_TO_CM, "waist_cm"),
    }))
    out = pd.concat(frames, ignore_index=True)
    out["study"] = "CATIE"
    return out.dropna(subset=["time_months", "value"])


def long_aclaims():
    lab = pd.read_parquet(DATA / "parquet_Aclaims/lab.parquet")
    cmap = {"glucose": "lab_fasting_glucose_mg_dl", "total_cholesterol": "lab_cholesterol_mg_dl",
            "hdl": "lab_hdl_cholesterol_mg_dl", "ldl": "lab_ldl_cholesterol_mg_dl",
            "triglycerides": "lab_triglycerides_mg_dl"}
    frames = []
    for marker, col in cmap.items():
        frames.append(pd.DataFrame({
            "subject_id": lab["patient_id"].astype(str), "marker": marker,
            "time_months": pd.to_numeric(lab["study_month"], errors="coerce"),
            "value": clip_range(lab[col], marker),
        }))
    vsf = pd.read_parquet(DATA / "parquet_Aclaims/vsf.parquet")
    frames.append(pd.DataFrame({
        "subject_id": vsf["patient_id"].astype(str), "marker": "waist_cm",
        "time_months": pd.to_numeric(vsf["study_month"], errors="coerce"),
        "value": clip_range(pd.to_numeric(vsf["vsf_waist_inches_1st"], errors="coerce") * IN_TO_CM, "waist_cm"),
    }))
    out = pd.concat(frames, ignore_index=True)
    out["study"] = "ACLAIMS"
    return out.dropna(subset=["time_months", "value"])


def long_litmus():
    d = pd.read_parquet(DATA / "parquet_litmus/derived.parquet")
    sid = d["NIMH_ID#"].astype(str)
    WPM = 4.34524
    lip = {"glucose": "Glucose", "total_cholesterol": "Total cholesterol",
           "hdl": "HDL cholesterol", "ldl": "LDL cholesterol", "triglycerides": "Triglycerides"}
    frames = []
    for oo in range(1, 4):  # LIPID obs 01-03
        tag = f"(Obs {oo:02d})"
        wcol = f"LIPID | Study week {tag}"
        if wcol not in d.columns:
            continue
        tm = pd.to_numeric(d[wcol], errors="coerce") / WPM
        for marker, lname in lip.items():
            vcol = f"LIPID | {lname} {tag}"
            if vcol not in d.columns:
                continue
            frames.append(pd.DataFrame({
                "subject_id": sid, "marker": marker, "time_months": tm,
                "value": clip_range(d[vcol], marker),
            }))
    # waist from VS obs
    wcol_pat = [c for c in d.columns if c.startswith("VS | Waist circumference")]
    for oo in range(1, 11):
        tag = f"(Obs {oo:02d})"
        tcol = f"VS | Study week {tag}"
        wc = [c for c in wcol_pat if c.endswith(tag)]
        if not wc or tcol not in d.columns:
            continue
        frames.append(pd.DataFrame({
            "subject_id": sid, "marker": "waist_cm",
            "time_months": pd.to_numeric(d[tcol], errors="coerce") / WPM,
            "value": clip_range(pd.to_numeric(d[wc[0]], errors="coerce") * IN_TO_CM, "waist_cm"),
        }))
    out = pd.concat(frames, ignore_index=True)
    out["study"] = "LiTMUS"
    return out.dropna(subset=["time_months", "value"])


# ---------------------------------------------------------------------------
# Baseline -> last change per (study, subject, marker)
# ---------------------------------------------------------------------------
def baseline_change(long):
    long = long.sort_values(["study", "marker", "subject_id", "time_months"])
    rows = []
    for (study, marker, sid), g in long.groupby(["study", "marker", "subject_id"]):
        g = g.dropna(subset=["value"])
        if len(g) < 2:
            continue
        bl = g[g["time_months"] <= BASELINE_MAX_MONTH]
        if bl.empty:
            bl = g.iloc[[0]]
        base_t = bl["time_months"].min()
        base_v = bl.loc[bl["time_months"] == base_t, "value"].mean()
        fu = g[g["time_months"] > base_t]
        if fu.empty:
            continue
        last = fu.iloc[-1]
        rows.append(dict(study=study, marker=marker, subject_id=sid,
                         baseline=base_v, followup=last["value"],
                         change=last["value"] - base_v, fu_months=last["time_months"]))
    return pd.DataFrame(rows)


def summarize(bc):
    rows = []
    for (study, marker), g in bc.groupby(["study", "marker"]):
        ch = g["change"].dropna()
        try:
            w_p = stats.wilcoxon(ch)[1] if len(ch) > 5 else np.nan
        except Exception:
            w_p = np.nan
        rows.append(dict(
            study=study, marker=marker, label=MARKER_LABEL[marker], N=len(g),
            baseline_mean=g["baseline"].mean(), baseline_sd=g["baseline"].std(),
            followup_mean=g["followup"].mean(),
            change_mean=ch.mean(), change_sd=ch.std(),
            pct_increase=(ch > 0).mean() * 100,
            median_fu_months=g["fu_months"].median(),
            wilcoxon_p=w_p,
        ))
    df = pd.DataFrame(rows)
    # BH-FDR within study x marker family
    p = df["wilcoxon_p"].values
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan, dtype=float)
    if ok.sum() > 0:
        order = np.argsort(p[ok]); ranked = p[ok][order]
        m = ok.sum()
        qv = ranked * m / (np.arange(1, m + 1))
        qv = np.minimum.accumulate(qv[::-1])[::-1]
        tmp = np.empty(m); tmp[order] = qv
        q[ok] = tmp
    df["q_fdr"] = q
    return df.sort_values(["study", "marker"])


def catie_by_drug(bc):
    """Olanzapine vs others on glucose & triglyceride change (known AP signal)."""
    kv = pd.read_parquet(DATA / "parquet_catie/KEYVARS.parquet")
    tx = kv[["NIMHID", "Treatment Assignment Phase 1"]].copy()
    tx["subject_id"] = tx["NIMHID"].astype(str)
    tx = tx.rename(columns={"Treatment Assignment Phase 1": "drug"}).dropna(subset=["drug"])
    c = bc[bc["study"] == "CATIE"].merge(tx[["subject_id", "drug"]], on="subject_id", how="inner")
    rows = []
    for marker in ["glucose", "triglycerides", "total_cholesterol", "waist_cm"]:
        m = c[c["marker"] == marker]
        for drug, g in m.groupby("drug"):
            rows.append(dict(marker=marker, drug=drug, N=len(g),
                             change_mean=g["change"].mean()))
    return pd.DataFrame(rows)


def main():
    print("=== Secondary metabolic outcomes (CATIE / ACLAIMS / LiTMUS) ===")
    long = pd.concat([long_catie(), long_aclaims(), long_litmus()], ignore_index=True)
    print(f"metabolic_long: {len(long):,} measurement-rows")
    bc = baseline_change(long)
    print(f"baseline->change pairs: {len(bc):,} subject-markers")
    summ = summarize(bc)
    print("\nSummary (mean change baseline->last):")
    print(summ[["study", "marker", "N", "baseline_mean", "change_mean",
                "pct_increase", "median_fu_months", "wilcoxon_p", "q_fdr"]]
          .to_string(index=False))
    cbd = catie_by_drug(bc)
    print("\nCATIE glucose/TG change by antipsychotic:")
    print(cbd[cbd["marker"].isin(["glucose", "triglycerides"])].to_string(index=False))

    long.to_parquet(OUT / "metabolic_long.parquet", index=False)
    results = {"metabolic_summary": summ, "metabolic_baseline_change": bc,
               "catie_metabolic_by_drug": cbd}
    with open(OUT / "metabolic_results.pkl", "wb") as f:
        pickle.dump(results, f)
    with pd.ExcelWriter(OUT / "metabolic_summary.xlsx", engine="openpyxl") as w:
        write_pretty_sheet(w, summ, "summary")
        write_pretty_sheet(w, cbd, "catie_by_drug")

    stats_json = {
        "studies": ["CATIE", "ACLAIMS", "LiTMUS"],
        "markers": list(MARKER_LABEL.keys()),
        "baseline_max_month": BASELINE_MAX_MONTH,
        "n_subject_markers": int(len(bc)),
        "summary": [
            {k: (None if (isinstance(v, float) and np.isnan(v)) else
                 (round(v, 3) if isinstance(v, (int, float, np.floating)) else v))
             for k, v in row.items()
             if k in ("study", "marker", "N", "baseline_mean", "change_mean",
                      "pct_increase", "median_fu_months", "wilcoxon_p", "q_fdr")}
            for row in summ.to_dict("records")
        ],
    }
    (OUT / "metabolic_stats.json").write_text(json.dumps(stats_json, indent=2, default=str))
    print("\nSaved metabolic_long.parquet, metabolic_results.pkl, "
          "metabolic_summary.xlsx, metabolic_stats.json")


if __name__ == "__main__":
    main()
