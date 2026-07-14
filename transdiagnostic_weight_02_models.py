"""
transdiagnostic_weight_02_models.py
Models for the transdiagnostic weight-gain paper.

Inputs (results/transdiagnostic_weight/):
  weight_long.parquet, baseline.parquet  (from 01_harmonize.py)

Primary outcomes:
  - %weight change over time  -> linear mixed model (slope %/month) by drug class
  - clinically significant weight gain (>=7%) -> logistic (ever + 3-month landmark)
Cross-cutting predictors: baseline weight, age, sex (0=male,1=female), drug class.
Secondary (hybrid): within-study drug contrast in CATIE (olanzapine vs others).

NOTE: drug_class and diagnosis are perfectly collinear (each class = one
diagnosis), so they are never entered together. drug_class is the class/diagnosis
axis; demographic predictors are the transdiagnostic axis.

Outputs:
  model_results.pkl, model_results_raw.xlsx, stats_summary.json
ALL numbers needed by the paper are saved here (no hardcoded stats downstream).
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "transdiagnostic_weight"
DATA = ROOT / "data"

CLASS_REF = "antidepressant"          # lowest-risk reference
CLASS_ORDER = ["antidepressant", "mood_stabilizer", "antipsychotic"]
STUDY_ORDER = ["COMED", "STEP", "LiTMUS", "CATIE", "ACLAIMS"]
LANDMARK_MONTH = 3.0
LANDMARK_WINDOW = (1.5, 4.5)
CSG = 7.0
# Common early window for slope comparison. Weight gain is front-loaded and
# non-linear; follow-up length differs 3-4x across studies (CO-MED/LiTMUS ~6mo
# vs CATIE/ACLAIMS ~20mo). A linear slope over the FULL follow-up is dominated
# by follow-up length and the post-gain plateau, mis-ranking classes. Restricting
# to the first SLOPE_WINDOW_MONTHS gives a comparable rapid-gain-phase slope.
SLOPE_WINDOW_MONTHS = 6.0


def load():
    long = pd.read_parquet(OUT / "weight_long.parquet")
    base = pd.read_parquet(OUT / "baseline.parquet")
    # subjects usable for slope = >=2 weight observations
    base["analyzable"] = base["n_weight_obs"] >= 2
    return long, base


# ---------------------------------------------------------------------------
# 1. Descriptives (Table 1 source)
# ---------------------------------------------------------------------------
def descriptives(base):
    rows = []
    for s in STUDY_ORDER:
        b = base[base["study"] == s]
        rows.append(dict(
            study=s, drug_class=b["drug_class"].iloc[0], diagnosis=b["diagnosis"].iloc[0],
            formulation=b["formulation"].iloc[0],
            N=len(b), n_ge2obs=int(b["analyzable"].sum()),
            age_mean=b["age"].mean(), age_sd=b["age"].std(),
            pct_female=b["sex_female"].mean() * 100,
            baseline_wt_kg_mean=b["baseline_weight_kg"].mean(),
            baseline_wt_kg_sd=b["baseline_weight_kg"].std(),
            median_followup_mo=b["max_followup_months"].median(),
            csg7_ever_n=int(b["ever_csg7"].sum()),
            csg7_ever_pct=b["ever_csg7"].mean() * 100,
            mean_max_pct_gain=b["max_pct_gain"].mean(),
            mean_last_pct_change=b["last_pct_change"].mean(),
        ))
    desc = pd.DataFrame(rows)
    # by-class rollup
    crows = []
    for c in CLASS_ORDER:
        b = base[base["drug_class"] == c]
        crows.append(dict(
            drug_class=c, N=len(b), n_ge2obs=int(b["analyzable"].sum()),
            age_mean=b["age"].mean(), pct_female=b["sex_female"].mean() * 100,
            baseline_wt_kg_mean=b["baseline_weight_kg"].mean(),
            csg7_ever_pct=b["ever_csg7"].mean() * 100,
            mean_max_pct_gain=b["max_pct_gain"].mean(),
        ))
    desc_class = pd.DataFrame(crows)
    return desc, desc_class


# ---------------------------------------------------------------------------
# 2. Primary continuous: linear mixed model, slope (%/month) by drug class
# ---------------------------------------------------------------------------
def mixed_slope_by_class(long, base):
    ana_ids = set(base.loc[base["analyzable"], ["study", "subject_id"]]
                  .itertuples(index=False, name=None))
    d = long[long.apply(lambda r: (r["study"], r["subject_id"]) in ana_ids, axis=1)].copy()
    d = d[d["t_months"] <= SLOPE_WINDOW_MONTHS]          # common early window
    d["uid"] = d["study"] + "::" + d["subject_id"]
    d["drug_class"] = pd.Categorical(d["drug_class"], categories=CLASS_ORDER)
    print(f"  mixed model (0-{SLOPE_WINDOW_MONTHS:.0f}mo): {len(d):,} rows, {d['uid'].nunique():,} subjects")

    md = smf.mixedlm(
        f"pct_change ~ t_months * C(drug_class, Treatment('{CLASS_REF}'))",
        data=d, groups=d["uid"], re_formula="~t_months",
    )
    mdf = md.fit(method="lbfgs", maxiter=200)
    params, bse, pvals = mdf.params, mdf.bse, mdf.pvalues

    base_slope = params["t_months"]
    rows = []
    for c in CLASS_ORDER:
        if c == CLASS_REF:
            slope = base_slope
            se = bse["t_months"]
        else:
            key = [k for k in params.index if k.startswith("t_months:") and f"T.{c}]" in k]
            inter = params[key[0]]
            slope = base_slope + inter
            # SE of sum via covariance
            cov = mdf.cov_params()
            se = np.sqrt(cov.loc["t_months", "t_months"]
                         + cov.loc[key[0], key[0]]
                         + 2 * cov.loc["t_months", key[0]])
        rows.append(dict(
            drug_class=c, slope_pct_per_month=slope,
            ci_low=slope - 1.96 * se, ci_high=slope + 1.96 * se,
            slope_per_year=slope * 12,
        ))
    slope_df = pd.DataFrame(rows)

    # interaction (class x time) contrasts vs reference
    inter_rows = []
    for c in CLASS_ORDER:
        if c == CLASS_REF:
            continue
        key = [k for k in params.index if k.startswith("t_months:") and f"T.{c}]" in k][0]
        inter_rows.append(dict(
            contrast=f"{c}_vs_{CLASS_REF}", beta_interaction=params[key],
            se=bse[key], p=pvals[key],
            ci_low=params[key] - 1.96 * bse[key], ci_high=params[key] + 1.96 * bse[key],
        ))
    inter_df = pd.DataFrame(inter_rows)
    return slope_df, inter_df, mdf.summary().as_text()


# ---------------------------------------------------------------------------
# 3. Per-study weight slope (forest plot source)
# ---------------------------------------------------------------------------
def per_study_slopes(long, base):
    rows = []
    for s in STUDY_ORDER:
        ids = set(base.loc[(base["study"] == s) & base["analyzable"], "subject_id"])
        d = long[(long["study"] == s) & (long["subject_id"].isin(ids)) &
                 (long["t_months"] <= SLOPE_WINDOW_MONTHS)].copy()
        try:
            md = smf.mixedlm("pct_change ~ t_months", data=d, groups=d["subject_id"],
                             re_formula="~t_months")
            mdf = md.fit(method="lbfgs", maxiter=200)
            slope, se = mdf.params["t_months"], mdf.bse["t_months"]
        except Exception as e:
            print(f"    {s} slope failed ({e}); OLS fallback")
            mdf = smf.ols("pct_change ~ t_months", data=d).fit()
            slope, se = mdf.params["t_months"], mdf.bse["t_months"]
        rows.append(dict(
            study=s, drug_class=d["drug_class"].iloc[0],
            slope_pct_per_month=slope, se=se,
            ci_low=slope - 1.96 * se, ci_high=slope + 1.96 * se,
            slope_per_year=slope * 12, n_subjects=len(ids),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Binary >=7% gain: logistic by class (+ covariates), ever and 3-mo landmark
# ---------------------------------------------------------------------------
def landmark_pct(long, base):
    """For each subject, pct_change at visit closest to LANDMARK_MONTH within window."""
    d = long[(long["t_months"] >= LANDMARK_WINDOW[0]) &
             (long["t_months"] <= LANDMARK_WINDOW[1])].copy()
    d["dist"] = (d["t_months"] - LANDMARK_MONTH).abs()
    d = d.sort_values(["study", "subject_id", "dist"])
    lm = d.groupby(["study", "subject_id"], as_index=False).first()
    lm = lm[["study", "subject_id", "pct_change", "t_months"]].rename(
        columns={"pct_change": "lm_pct_change", "t_months": "lm_time"})
    out = base.merge(lm, on=["study", "subject_id"], how="left")
    out["lm_csg7"] = (out["lm_pct_change"] >= CSG).astype("float")
    out.loc[out["lm_pct_change"].isna(), "lm_csg7"] = np.nan
    return out


def logistic_models(base_lm):
    d = base_lm.copy()
    d["drug_class"] = pd.Categorical(d["drug_class"], categories=CLASS_ORDER)
    results = {}

    def fit_logit(formula, data, label):
        dd = data.dropna(subset=["sex_female", "age", "baseline_weight_kg"])
        m = smf.logit(formula, data=dd).fit(disp=0)
        rows = []
        for term in m.params.index:
            rows.append(dict(term=term, OR=np.exp(m.params[term]),
                             ci_low=np.exp(m.conf_int().loc[term, 0]),
                             ci_high=np.exp(m.conf_int().loc[term, 1]),
                             p=m.pvalues[term]))
        df = pd.DataFrame(rows)
        df.attrs["N"] = int(m.nobs)
        print(f"  {label}: N={int(m.nobs)}")
        return df

    f_class = f"~ C(drug_class, Treatment('{CLASS_REF}')) + baseline_weight_kg + age + sex_female"
    results["logit_ever_csg7"] = fit_logit("ever_csg7 " + f_class, d, "ever>=7% ~ class+cov")
    results["logit_landmark_csg7"] = fit_logit("lm_csg7 " + f_class, d, "3mo>=7% ~ class+cov")

    # cross-cutting predictors only (pooled, demographic axis) within each class as strata? -> pooled
    results["logit_predictors"] = fit_logit(
        "ever_csg7 ~ baseline_weight_kg + age + sex_female + C(drug_class, Treatment('antidepressant'))",
        d, "predictors")

    # sex x class interaction
    try:
        results["logit_sex_x_class"] = fit_logit(
            f"ever_csg7 ~ C(drug_class, Treatment('{CLASS_REF}')) * sex_female + baseline_weight_kg + age",
            d, "sex x class")
    except Exception as e:
        print(f"    sex x class failed: {e}")

    # Transdiagnostic-consistency test: does each predictor's effect differ by
    # drug class/diagnosis? Likelihood-ratio test of predictor x class block.
    # Non-significant p => predictor effect is consistent across classes
    # (i.e., genuinely transdiagnostic).
    from scipy import stats as _st
    dd = d.dropna(subset=["sex_female", "age", "baseline_weight_kg"]).copy()
    cl = f"C(drug_class, Treatment('{CLASS_REF}'))"
    m_main = smf.logit(f"ever_csg7 ~ {cl} + baseline_weight_kg + age + sex_female",
                       data=dd).fit(disp=0)
    rows = []
    for pred in ["baseline_weight_kg", "age", "sex_female"]:
        m_int = smf.logit(
            f"ever_csg7 ~ {cl} + baseline_weight_kg + age + sex_female + {cl}:{pred}",
            data=dd).fit(disp=0)
        lr = 2 * (m_int.llf - m_main.llf)
        ddf = int(round(m_int.df_model - m_main.df_model))
        rows.append(dict(predictor=pred, lr_chi2=lr, df=ddf, p=float(_st.chi2.sf(lr, ddf))))
    results["predictor_class_interaction"] = pd.DataFrame(rows)
    print("  predictor x class consistency (LR p):",
          {r['predictor']: round(r['p'], 3) for r in rows})
    return results


# ---------------------------------------------------------------------------
# 5. Within-study drug contrast: CATIE olanzapine vs others (hybrid secondary)
# ---------------------------------------------------------------------------
def catie_olanzapine(long, base):
    kv = pd.read_parquet(DATA / "parquet_catie/KEYVARS.parquet")
    tx = kv[["NIMHID", "Treatment Assignment Phase 1"]].copy()
    tx["subject_id"] = tx["NIMHID"].astype(str)
    tx = tx.rename(columns={"Treatment Assignment Phase 1": "drug"})
    tx = tx.dropna(subset=["drug"])

    b = base[base["study"] == "CATIE"].merge(tx[["subject_id", "drug"]], on="subject_id", how="inner")
    b["olanzapine"] = (b["drug"] == "Olanzapine").astype(int)

    # ever>=7% by CATIE drug
    by_drug = (b.groupby("drug")
                 .agg(N=("ever_csg7", "size"), csg7_pct=("ever_csg7", lambda x: x.mean() * 100),
                      mean_max_gain=("max_pct_gain", "mean"))
                 .reset_index().sort_values("csg7_pct", ascending=False))

    # logistic olanzapine vs others
    dd = b.dropna(subset=["sex_female", "age", "baseline_weight_kg"])
    m = smf.logit("ever_csg7 ~ olanzapine + baseline_weight_kg + age + sex_female",
                  data=dd).fit(disp=0)
    ola = dict(OR=np.exp(m.params["olanzapine"]),
               ci_low=np.exp(m.conf_int().loc["olanzapine", 0]),
               ci_high=np.exp(m.conf_int().loc["olanzapine", 1]),
               p=m.pvalues["olanzapine"], N=int(m.nobs))

    # slope by drug
    ids = set(base.loc[(base["study"] == "CATIE") & base["analyzable"], "subject_id"])
    ld = long[(long["study"] == "CATIE") & (long["subject_id"].isin(ids)) &
              (long["t_months"] <= SLOPE_WINDOW_MONTHS)].merge(
        tx[["subject_id", "drug"]], on="subject_id", how="inner")
    srows = []
    for drug, g in ld.groupby("drug"):
        try:
            md = smf.mixedlm("pct_change ~ t_months", data=g, groups=g["subject_id"],
                             re_formula="~t_months").fit(method="lbfgs", maxiter=200)
            srows.append(dict(drug=drug, slope_pct_per_month=md.params["t_months"],
                              se=md.bse["t_months"], n=g["subject_id"].nunique()))
        except Exception:
            pass
    slope_by_drug = pd.DataFrame(srows).sort_values("slope_pct_per_month", ascending=False)
    return by_drug, ola, slope_by_drug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Transdiagnostic weight models ===")
    long, base = load()

    print("\n[1] Descriptives")
    desc, desc_class = descriptives(base)
    print(desc_class.to_string(index=False))

    print("\n[2] Mixed model slope by class")
    slope_df, inter_df, mixed_txt = mixed_slope_by_class(long, base)
    print(slope_df.to_string(index=False))
    print(inter_df.to_string(index=False))

    print("\n[3] Per-study slopes (forest)")
    forest = per_study_slopes(long, base)
    print(forest.to_string(index=False))

    print("\n[4] Landmark + logistic")
    base_lm = landmark_pct(long, base)
    logits = logistic_models(base_lm)
    print(logits["logit_ever_csg7"].to_string(index=False))

    print("\n[5] CATIE olanzapine vs others")
    catie_by_drug, ola, catie_slope = catie_olanzapine(long, base)
    print(catie_by_drug.to_string(index=False))
    print("  olanzapine OR (ever>=7%):", {k: round(v, 3) if isinstance(v, float) else v for k, v in ola.items()})

    # ---- save ----
    results = {
        "descriptives_by_study": desc,
        "descriptives_by_class": desc_class,
        "mixed_slope_by_class": slope_df,
        "mixed_class_time_interactions": inter_df,
        "per_study_slopes": forest,
        "catie_by_drug": catie_by_drug,
        "catie_slope_by_drug": catie_slope,
    }
    results.update({k: v for k, v in logits.items()})

    with open(OUT / "model_results.pkl", "wb") as f:
        pickle.dump(results, f)
    with pd.ExcelWriter(OUT / "model_results_raw.xlsx") as w:
        for name, dat in results.items():
            if isinstance(dat, pd.DataFrame) and not dat.empty:
                dat.to_excel(w, sheet_name=name[:31], index=False)
    (OUT / "mixed_model_summary.txt").write_text(mixed_txt)

    # JSON: scalar stats the paper needs
    summary = {
        "N_total": int(len(base)),
        "N_analyzable_ge2obs": int(base["analyzable"].sum()),
        "by_study_N": base.groupby("study").size().to_dict(),
        "by_class_N": base.groupby("drug_class").size().to_dict(),
        "csg7_ever_pct_by_class": {
            c: round(base.loc[base["drug_class"] == c, "ever_csg7"].mean() * 100, 1)
            for c in CLASS_ORDER},
        "slope_pct_per_month_by_class": {
            r["drug_class"]: round(r["slope_pct_per_month"], 3)
            for _, r in slope_df.iterrows()},
        "class_time_interaction_p": {
            r["contrast"]: float(r["p"]) for _, r in inter_df.iterrows()},
        "landmark_month": LANDMARK_MONTH,
        "slope_window_months": SLOPE_WINDOW_MONTHS,
        "csg_threshold": CSG,
        "catie_olanzapine_OR_ever_csg7": {k: (round(v, 3) if isinstance(v, float) else v)
                                          for k, v in ola.items()},
    }
    # logistic class ORs for ever_csg7
    le = logits["logit_ever_csg7"]
    summary["ever_csg7_class_OR"] = {
        r["term"]: round(r["OR"], 3) for _, r in le.iterrows()
        if "drug_class" in r["term"]}
    summary["ever_csg7_predictor_OR"] = {
        r["term"]: {"OR": round(r["OR"], 3), "p": float(r["p"])}
        for _, r in logits["logit_predictors"].iterrows()
        if r["term"] in ("baseline_weight_kg", "age", "sex_female")}
    # transdiagnostic-consistency LR test p-values (predictor x class)
    summary["predictor_class_interaction_p"] = {
        r["predictor"]: round(float(r["p"]), 3)
        for _, r in logits["predictor_class_interaction"].iterrows()}

    (OUT / "stats_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nSaved model_results.pkl, model_results_raw.xlsx, stats_summary.json")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
