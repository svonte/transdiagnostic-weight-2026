"""
transdiagnostic_weight_06_sensitivity.py
Sensitivity / robustness analyses for the transdiagnostic weight-gain paper.

Referee concerns addressed:
  - the cumulative ≥7% outcome favours trials with longer follow-up
    (CATIE/ACLAIMS ~20mo vs CO-MED/LiTMUS ~6mo);
  - STEP-BD dominates the mood-stabilizer class (~95% of its n);
  - the early-slope window (0-6mo) is a choice.

Analyses:
  S1. Leave-one-study-out (LOSO) on the pooled ≥7%-gain logistic.
  S2. Two-stage random-effects (DerSimonian-Laird) meta-analysis of the early
      weight slope, pooled within drug class (handles STEP-BD dominance).
  S3. Common-window binary: ≥7% gain achieved within the first 6 months only,
      refit by class (removes the long-follow-up advantage).
  S4. ≥5% / ≥7% / ≥10% threshold sensitivity by class.
  S5. Early-slope window sensitivity (3 / 6 / 9 / 12 months) by class.
  S6. Predictor models: cumulative ever-≥7% vs 3-month landmark.
  S7. Study-clustered covariance sensitivity for the class logistic models
      (exploratory: only five study clusters).
  S8. Adjusted early-slope (0-6mo) mixed model: class-by-time slopes additionally
      adjusted for baseline weight, age, and sex (their interactions with time),
      to test robustness of the descriptive slope findings (Reviewer 1, comment 3).
  S9. Follow-up completeness / missingness by trial: number and % of randomized
      participants with a weight measurement at baseline, ~3mo, ~6mo, ~12mo, and
      any later window (Reviewer 1, comment 4).
  S10. Absolute weight-change sensitivity: whether lower baseline weight still
      predicts gain when the outcome is absolute (kg) rather than percentage-based,
      i.e. disentangling the mathematical artefact of a %-threshold (Reviewer 1,
      comment 5).
  S11. Within-antipsychotic comorbidity-adjusted sensitivity: baseline-weight and
      olanzapine effects adjusted for baseline diabetes / endocrine-metabolic
      comorbidity in CATIE+ACLAIMS (Reviewer 2, comment 3).
  S12. Estimand sensitivity: robustness of the cumulative ≥7% class comparison to
      (A) excluding subjects with no post-baseline follow-up and (B) re-anchoring
      STEP-BD to a protocol-baseline window, versus the primary earliest-weight
      baseline (structural review; Reviewer 1, comments 4/5).

Input:  results/transdiagnostic_weight/{weight_long,baseline}.parquet,
        model_results.pkl
Output: results/transdiagnostic_weight/sensitivity_results.xlsx
        results/transdiagnostic_weight/sensitivity_summary.json
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from xlsx_format import write_pretty_sheet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "transdiagnostic_weight"
DATA = ROOT / "data"

CLASS_ORDER = ["antidepressant", "mood_stabilizer", "antipsychotic"]
CLASS_REF = "antidepressant"
AP_TERM = "C(drug_class, Treatment('antidepressant'))[T.antipsychotic]"
MS_TERM = "C(drug_class, Treatment('antidepressant'))[T.mood_stabilizer]"
STUDY_ORDER = ["CATIE", "ACLAIMS", "COMED", "STEP", "LiTMUS"]

long = pd.read_parquet(OUT / "weight_long.parquet")
base = pd.read_parquet(OUT / "baseline.parquet")
base["analyzable"] = base["n_weight_obs"] >= 2
# Primary-analysis cohort (Reviewer 2, comment 2): participants with ≥1
# post-baseline weight, i.e. observed for incident ≥7% gain. Sensitivities that
# mirror the primary analysis (S1, S3, S4, S7, S10, S11) use this cohort; S9
# (completeness) and S12 (estimand sensitivity) deliberately use all randomized.
base_ana = base[base["analyzable"]].copy()
LBS_TO_KG = 0.45359237
res = pickle.load(open(OUT / "model_results.pkl", "rb"))


def fit_class_logit(d, ycol):
    d = d.dropna(subset=["sex_female", "age", "baseline_weight_kg", ycol]).copy()
    d["drug_class"] = pd.Categorical(d["drug_class"], categories=CLASS_ORDER)
    m = smf.logit(
        f"{ycol} ~ C(drug_class, Treatment('{CLASS_REF}')) + baseline_weight_kg + age + sex_female",
        data=d).fit(disp=0)
    out = {}
    for term in (AP_TERM, MS_TERM):
        out[term] = (np.exp(m.params[term]),
                     np.exp(m.conf_int().loc[term, 0]),
                     np.exp(m.conf_int().loc[term, 1]),
                     m.pvalues[term])
    out["N"] = int(m.nobs)
    return out


def fit_class_logit_clustered(d, ycol):
    d = d.dropna(subset=["sex_female", "age", "baseline_weight_kg", ycol]).copy()
    d["drug_class"] = pd.Categorical(d["drug_class"], categories=CLASS_ORDER)
    m = smf.logit(
        f"{ycol} ~ C(drug_class, Treatment('{CLASS_REF}')) + baseline_weight_kg + age + sex_female",
        data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["study"]})
    rows = []
    for term, label in ((AP_TERM, "Antipsychotic vs AD"), (MS_TERM, "Mood stabilizer vs AD")):
        rows.append(dict(
            outcome=ycol,
            term=label,
            OR=np.exp(m.params[term]),
            ci_low=np.exp(m.conf_int().loc[term, 0]),
            ci_high=np.exp(m.conf_int().loc[term, 1]),
            p=m.pvalues[term],
            N=int(m.nobs),
            n_clusters=int(d["study"].nunique()),
        ))
    return pd.DataFrame(rows)


# ── S1. Leave-one-study-out ───────────────────────────────────────────────────
def s1_loso():
    rows = []
    full = fit_class_logit(base_ana, "ever_csg7")
    rows.append(dict(dropped="(none, full)", N=full["N"],
                     AP_OR=full[AP_TERM][0], AP_lo=full[AP_TERM][1], AP_hi=full[AP_TERM][2],
                     MS_OR=full[MS_TERM][0], MS_lo=full[MS_TERM][1], MS_hi=full[MS_TERM][2]))
    for s in STUDY_ORDER:
        sub = base_ana[base_ana["study"] != s]
        try:
            r = fit_class_logit(sub, "ever_csg7")
            rows.append(dict(dropped=s, N=r["N"],
                             AP_OR=r[AP_TERM][0], AP_lo=r[AP_TERM][1], AP_hi=r[AP_TERM][2],
                             MS_OR=r[MS_TERM][0], MS_lo=r[MS_TERM][1], MS_hi=r[MS_TERM][2]))
        except Exception as e:
            rows.append(dict(dropped=s, N=np.nan, AP_OR=np.nan))
            print(f"   LOSO drop {s} failed: {e}")
    return pd.DataFrame(rows)


# ── S2. DerSimonian-Laird random-effects meta-analysis of slope ───────────────
def dersimonian_laird(yi, vi):
    yi, vi = np.asarray(yi, float), np.asarray(vi, float)
    wi = 1 / vi
    mu_fixed = np.sum(wi * yi) / np.sum(wi)
    Q = np.sum(wi * (yi - mu_fixed) ** 2)
    k = len(yi)
    C = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wi_re = 1 / (vi + tau2)
    mu = np.sum(wi_re * yi) / np.sum(wi_re)
    se = np.sqrt(1 / np.sum(wi_re))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    return dict(pooled=mu, se=se, ci_low=mu - 1.96 * se, ci_high=mu + 1.96 * se,
                tau2=tau2, Q=Q, I2=I2, k=k)


def s2_meta_slope():
    f = res["per_study_slopes"].copy()
    f["var"] = f["se"] ** 2
    rows = []
    for cls in CLASS_ORDER:
        g = f[f["drug_class"] == cls]
        if len(g) == 1:
            r = g.iloc[0]
            rows.append(dict(drug_class=cls, k=1, pooled=r["slope_pct_per_month"],
                             ci_low=r["ci_low"], ci_high=r["ci_high"], I2=np.nan, tau2=np.nan))
        else:
            dl = dersimonian_laird(g["slope_pct_per_month"], g["var"])
            rows.append(dict(drug_class=cls, k=dl["k"], pooled=dl["pooled"],
                             ci_low=dl["ci_low"], ci_high=dl["ci_high"],
                             I2=dl["I2"], tau2=dl["tau2"]))
    return pd.DataFrame(rows)


# ── S3. Common 6-month-window binary ──────────────────────────────────────────
def common_window_data(window=6.0):
    # Restrict to participants with ≥1 POST-baseline weight within the window
    # (0<t<=window), i.e. actually observed for incident gain over the common
    # window (Reviewer 2, comments 2 & 4). Denominator therefore excludes
    # baseline-only participants and anyone with no in-window follow-up.
    w = long[(long["t_months"] > 0) & (long["t_months"] <= window)]
    mx = (w.groupby(["study", "subject_id"])["pct_change"].max().reset_index()
          .rename(columns={"pct_change": "max_pct_6mo"}))
    b = base.merge(mx, on=["study", "subject_id"], how="inner")
    b["csg7_6mo"] = (b["max_pct_6mo"] >= 7).astype(float)
    return b


def s3_common_window(window=6.0):
    b = common_window_data(window)
    pct = {c: round(b.loc[b["drug_class"] == c, "csg7_6mo"].mean() * 100, 1) for c in CLASS_ORDER}
    # event counts and denominators by cohort (Reviewer 2, comment 4)
    counts = {c: dict(events=int(b.loc[b["drug_class"] == c, "csg7_6mo"].sum()),
                      n=int(b.loc[b["drug_class"] == c, "csg7_6mo"].notna().sum()))
              for c in CLASS_ORDER}
    r = fit_class_logit(b, "csg7_6mo")
    model = pd.DataFrame([
        dict(term="Antipsychotic vs AD", n_events=counts["antipsychotic"]["events"],
             n=counts["antipsychotic"]["n"], OR=r[AP_TERM][0], ci_low=r[AP_TERM][1],
             ci_high=r[AP_TERM][2], p=r[AP_TERM][3]),
        dict(term="Mood stabilizer vs AD", n_events=counts["mood_stabilizer"]["events"],
             n=counts["mood_stabilizer"]["n"], OR=r[MS_TERM][0], ci_low=r[MS_TERM][1],
             ci_high=r[MS_TERM][2], p=r[MS_TERM][3]),
    ])
    return pct, counts, model, int(r["N"])


# ── S4. Threshold sensitivity ─────────────────────────────────────────────────
def s4_thresholds():
    rows = []
    for thr in (5, 7, 10):
        for c in CLASS_ORDER:
            b = base_ana[base_ana["drug_class"] == c]          # primary cohort
            pct = (b["max_pct_gain"] >= thr).mean() * 100
            rows.append(dict(threshold_pct=thr, drug_class=c, n=len(b),
                             gain_pct=round(pct, 1)))
    return pd.DataFrame(rows)


# ── S6. Predictor models: cumulative ever-≥7% vs 3-month landmark ─────────────
def s6_predictor_models():
    """Side-by-side OR (95% CI) and p for the two logistic predictor models, so the
    cumulative vs early (landmark) contrast for age and sex is displayed explicitly."""
    lm_month = json.loads((OUT / "stats_summary.json").read_text()).get("landmark_month", 3)
    term_label = {
        AP_TERM: "Antipsychotic (vs antidepressant)",
        MS_TERM: "Mood stabilizer (vs antidepressant)",
        "baseline_weight_kg": "Baseline weight (+1 kg)",
        "age": "Age (+1 year)",
        "sex_female": "Female sex (vs male)",
    }
    ever = res["logit_ever_csg7"].set_index("term")
    land = res["logit_landmark_csg7"].set_index("term")
    rows = []
    for term, lab in term_label.items():
        r = dict(predictor=lab)
        for tag, df in (("ever≥7%", ever), (f"{int(lm_month)}-mo landmark", land)):
            if term in df.index:
                x = df.loc[term]
                r[f"OR ({tag})"] = round(x["OR"], 3)
                r[f"95% CI ({tag})"] = f"{x['ci_low']:.3f}–{x['ci_high']:.3f}"
                r[f"p ({tag})"] = round(x["p"], 4)
        rows.append(r)
    return pd.DataFrame(rows)


# ── S5. Slope-window sensitivity ──────────────────────────────────────────────
def s5_slope_windows():
    ana = base.loc[base["analyzable"], ["study", "subject_id"]]
    ana_ids = set(map(tuple, ana.values))
    d0 = long[long.apply(lambda r: (r["study"], r["subject_id"]) in ana_ids, axis=1)].copy()
    d0["uid"] = d0["study"] + "::" + d0["subject_id"]
    rows = []
    for win in (3, 6, 9, 12):
        d = d0[d0["t_months"] <= win].copy()
        d["drug_class"] = pd.Categorical(d["drug_class"], categories=CLASS_ORDER)
        try:
            md = smf.mixedlm(
                f"pct_change ~ t_months * C(drug_class, Treatment('{CLASS_REF}'))",
                data=d, groups=d["uid"], re_formula="~t_months").fit(method="lbfgs", maxiter=200)
            base_slope = md.params["t_months"]
            for c in CLASS_ORDER:
                if c == CLASS_REF:
                    slope = base_slope
                else:
                    key = [k for k in md.params.index if k.startswith("t_months:") and f"T.{c}]" in k][0]
                    slope = base_slope + md.params[key]
                rows.append(dict(window_months=win, drug_class=c,
                                 slope_pct_per_month=round(slope, 3)))
        except Exception as e:
            print(f"   slope window {win} failed: {e}")
    return pd.DataFrame(rows)


# ── S7. Study-clustered covariance sensitivity ────────────────────────────────
def s7_clustered_logit():
    rows = [
        fit_class_logit_clustered(base, "ever_csg7"),
        fit_class_logit_clustered(common_window_data(), "csg7_6mo"),
    ]
    return pd.concat(rows, ignore_index=True)


# ── S8. Adjusted early-slope mixed model (0-6mo) ──────────────────────────────
def s8_adjusted_slope(window=6.0):
    """Refit the class-by-time slope model over the common early window, now
    additionally adjusting the slope for baseline weight, age, and sex via their
    interactions with time (Reviewer 1, comment 3). Reports the covariate-adjusted
    class slopes and the class-vs-antidepressant time-interaction p-values, side by
    side with the unadjusted estimates from the primary model."""
    ana = base.loc[base["analyzable"], ["study", "subject_id"]]
    ana_ids = set(map(tuple, ana.values))
    d = long[long.apply(lambda r: (r["study"], r["subject_id"]) in ana_ids, axis=1)].copy()
    d = d[d["t_months"] <= window].copy()
    d["uid"] = d["study"] + "::" + d["subject_id"]
    d = d.merge(base[["study", "subject_id", "baseline_weight_kg", "age", "sex_female"]],
                on=["study", "subject_id"], how="left")
    d = d.dropna(subset=["baseline_weight_kg", "age", "sex_female", "pct_change"]).copy()
    # centre covariates so the main class-by-time terms remain interpretable
    for cov in ("baseline_weight_kg", "age", "sex_female"):
        d[cov + "_c"] = d[cov] - d[cov].mean()
    d["drug_class"] = pd.Categorical(d["drug_class"], categories=CLASS_ORDER)
    cl = f"C(drug_class, Treatment('{CLASS_REF}'))"
    formula = (f"pct_change ~ t_months * {cl} "
               f"+ t_months:baseline_weight_kg_c + t_months:age_c + t_months:sex_female_c")
    md = smf.mixedlm(formula, data=d, groups=d["uid"], re_formula="~t_months").fit(
        method="lbfgs", maxiter=300)
    params, cov = md.params, md.cov_params()
    base_slope = params["t_months"]
    rows = []
    for c in CLASS_ORDER:
        if c == CLASS_REF:
            slope = base_slope
            se = md.bse["t_months"]
            p_int = np.nan
        else:
            key = [k for k in params.index if k.startswith("t_months:") and f"T.{c}]" in k][0]
            slope = base_slope + params[key]
            se = np.sqrt(cov.loc["t_months", "t_months"] + cov.loc[key, key]
                         + 2 * cov.loc["t_months", key])
            p_int = md.pvalues[key]
        rows.append(dict(drug_class=c, adj_slope_pct_per_month=round(slope, 3),
                         ci_low=round(slope - 1.96 * se, 3),
                         ci_high=round(slope + 1.96 * se, 3),
                         p_vs_antidepressant=(None if np.isnan(p_int) else float(p_int))))
    adj = pd.DataFrame(rows)
    # attach unadjusted slopes for side-by-side comparison
    unadj = res["mixed_slope_by_class"].set_index("drug_class")
    adj["unadj_slope_pct_per_month"] = adj["drug_class"].map(
        lambda c: round(unadj.loc[c, "slope_pct_per_month"], 3))
    adj["n_obs"] = len(d)
    adj["n_subjects"] = d["uid"].nunique()
    return adj[["drug_class", "unadj_slope_pct_per_month", "adj_slope_pct_per_month",
                "ci_low", "ci_high", "p_vs_antidepressant", "n_subjects"]]


# ── S9. Follow-up completeness / missingness by trial ─────────────────────────
def s9_followup_completeness():
    """For each trial, count randomized participants with a weight observation in
    windows centred on baseline, 3, 6, and 12 months, plus any later visit
    (Reviewer 1, comment 4). Denominator = N randomized (baseline)."""
    windows = [("Baseline", -1.0, 0.75),
               ("~3 months", 1.5, 4.5),
               ("~6 months", 4.5, 7.5),
               ("~12 months", 10.5, 13.5),
               ("Any >13.5 mo", 13.5, 1e9)]
    rows = []
    for s in STUDY_ORDER:
        n_rand = int((base["study"] == s).sum())
        d = long[long["study"] == s]
        rec = dict(study=s, N_randomized=n_rand)
        for lab, lo, hi in windows:
            ids = d[(d["t_months"] > lo) & (d["t_months"] <= hi)]["subject_id"].nunique()
            rec[f"{lab} n"] = int(ids)
            rec[f"{lab} %"] = round(100 * ids / n_rand, 1) if n_rand else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ── S10. Absolute weight-change sensitivity ───────────────────────────────────
def s10_absolute_change():
    """Test whether lower baseline weight predicts gain when the outcome is
    ABSOLUTE (kg), not percentage-based. A ≥7% threshold is mechanically easier to
    reach at low body weight; if the baseline-weight association weakens or reverses
    for an absolute-gain outcome, the %-threshold artefact is confirmed
    (Reviewer 1, comment 5). abs_max_gain_kg = baseline_weight_kg * max_pct_gain/100."""
    d = base_ana.copy()                                        # primary cohort
    d["abs_max_gain_kg"] = d["baseline_weight_kg"] * d["max_pct_gain"] / 100.0
    d["abs_ge5kg"] = (d["abs_max_gain_kg"] >= 5.0).astype(float)
    d["drug_class"] = pd.Categorical(d["drug_class"], categories=CLASS_ORDER)
    dd = d.dropna(subset=["sex_female", "age", "baseline_weight_kg"]).copy()
    cl = f"C(drug_class, Treatment('{CLASS_REF}'))"

    # (a) logistic: absolute ≥5 kg gain
    m_log = smf.logit(f"abs_ge5kg ~ {cl} + baseline_weight_kg + age + sex_female",
                      data=dd).fit(disp=0)
    # (b) linear: absolute max gain (kg)
    m_lin = smf.ols(f"abs_max_gain_kg ~ {cl} + baseline_weight_kg + age + sex_female",
                    data=dd).fit()

    bw_log_or = float(np.exp(m_log.params["baseline_weight_kg"]))
    bw_log_ci = np.exp(m_log.conf_int().loc["baseline_weight_kg"]).values
    bw_lin_beta = float(m_lin.params["baseline_weight_kg"])
    bw_lin_ci = m_lin.conf_int().loc["baseline_weight_kg"].values

    # reference: percentage-based ≥7% OR for baseline weight (from primary model)
    pct_or = float(res["logit_ever_csg7"]
                   .set_index("term").loc["baseline_weight_kg", "OR"])

    rows = [
        dict(outcome="Cumulative ≥7% gain (percentage-based, primary)",
             baseline_weight_effect=f"OR {pct_or:.3f} per kg",
             direction="lower weight → more gain"),
        dict(outcome="Absolute ≥5 kg gain (logistic)",
             baseline_weight_effect=(f"OR {bw_log_or:.3f} per kg "
                                     f"({bw_log_ci[0]:.3f}–{bw_log_ci[1]:.3f}), "
                                     f"p={m_log.pvalues['baseline_weight_kg']:.3g}"),
             direction=("lower weight → more gain" if bw_log_or < 1
                        else "higher weight → more gain / null")),
        dict(outcome="Absolute max gain, kg (linear)",
             baseline_weight_effect=(f"β {bw_lin_beta:+.3f} kg per kg "
                                     f"({bw_lin_ci[0]:+.3f} to {bw_lin_ci[1]:+.3f}), "
                                     f"p={m_lin.pvalues['baseline_weight_kg']:.3g}"),
             direction=("lower weight → more gain" if bw_lin_beta < 0
                        else "higher weight → more gain / null")),
    ]
    summary = dict(
        pct7_baseline_OR=round(pct_or, 3),
        abs5kg_baseline_OR=round(bw_log_or, 3),
        abs5kg_baseline_OR_ci=[round(bw_log_ci[0], 3), round(bw_log_ci[1], 3)],
        abs5kg_baseline_p=float(m_log.pvalues["baseline_weight_kg"]),
        abs_kg_baseline_beta=round(bw_lin_beta, 3),
        abs_kg_baseline_p=float(m_lin.pvalues["baseline_weight_kg"]),
        N=int(m_log.nobs),
    )
    return pd.DataFrame(rows), summary


# ── S11. Within-antipsychotic comorbidity-adjusted sensitivity ────────────────
def _antipsychotic_comorbidity_flags():
    """Baseline diabetes and endocrine/metabolic-comorbidity flags for the two
    antipsychotic trials, which record them (CATIE: MedDRA-coded MEDHX; ACLAIMS:
    structured mhx fields). Absence of a record = no comorbidity (flag 0)."""
    # CATIE — MedDRA medical history (long format)
    c = pd.read_parquet(DATA / "parquet_catie/MEDHX.parquet")
    c["sid"] = c["NIMHID"].astype(str)
    pt = c["Preferred Term (MedDRA)"].astype(str).str.upper()
    soc = c["System Organ Class (MedDRA)"].astype(str).str.upper()
    catie_dia = set(c.loc[pt.str.contains("DIABETES MELLITUS"), "sid"])
    catie_endo = set(c.loc[soc.isin(["ENDOCRINE DISORDERS",
                                     "METABOLISM AND NUTRITION DISORDERS"]), "sid"])
    # ACLAIMS — structured mhx (A=active, P=past, N=no)
    a = pd.read_parquet(DATA / "parquet_Aclaims/mhx.parquet")
    a["sid"] = a["patient_id"].astype(str)
    def _present(col):
        return a[col].astype(str).str.upper().isin(["A", "P", "Y"])
    acl_dia = set(a.loc[_present("mhx_diabetes") | _present("mhx_on_insulin")
                        | _present("mhx_on_oral_hypoglycemic"), "sid"])
    acl_endo = set(a.loc[_present("mhx_endocrine") | a["sid"].isin(acl_dia), "sid"])

    def flags(row):
        sid = str(row["subject_id"])
        if row["study"] == "CATIE":
            return pd.Series({"baseline_diabetes": int(sid in catie_dia),
                              "baseline_endocrine": int(sid in catie_endo)})
        return pd.Series({"baseline_diabetes": int(sid in acl_dia),
                          "baseline_endocrine": int(sid in acl_endo)})

    ap = base_ana[base_ana["drug_class"] == "antipsychotic"].copy()   # primary cohort
    ap[["baseline_diabetes", "baseline_endocrine"]] = ap.apply(flags, axis=1)
    return ap


def _or_row(m, term):
    return (float(np.exp(m.params[term])),
            float(np.exp(m.conf_int().loc[term, 0])),
            float(np.exp(m.conf_int().loc[term, 1])),
            float(m.pvalues[term]))


def s11_comorbidity_sensitivity():
    """Reviewer 2, comment 3. The two antipsychotic trials record baseline
    metabolic comorbidity, so we can test—within the antipsychotic cohort, where
    the key metabolic findings arise—whether adjusting for baseline diabetes and
    endocrine/metabolic comorbidity changes the baseline-weight predictor and the
    within-CATIE olanzapine contrast. (A pooled 5-trial adjustment is not possible
    because the other trials record comorbidity in incompatible formats.)"""
    ap = _antipsychotic_comorbidity_flags()
    ap = ap.dropna(subset=["sex_female", "age", "baseline_weight_kg"]).copy()

    # olanzapine assignment in CATIE (as in 02_models.catie_olanzapine)
    kv = pd.read_parquet(DATA / "parquet_catie/KEYVARS.parquet")
    kv["subject_id"] = kv["NIMHID"].astype(str)
    kv = kv.rename(columns={"Treatment Assignment Phase 1": "drug"})[["subject_id", "drug"]]

    prev = {
        "CATIE_diabetes_pct": round(100 * ap.loc[ap.study == "CATIE", "baseline_diabetes"].mean(), 1),
        "CATIE_endocrine_pct": round(100 * ap.loc[ap.study == "CATIE", "baseline_endocrine"].mean(), 1),
        "ACLAIMS_diabetes_pct": round(100 * ap.loc[ap.study == "ACLAIMS", "baseline_diabetes"].mean(), 1),
        "ACLAIMS_endocrine_pct": round(100 * ap.loc[ap.study == "ACLAIMS", "baseline_endocrine"].mean(), 1),
    }

    rows, summ = [], {"prevalence": prev}

    # Model 1: baseline-weight predictor within the antipsychotic cohort
    m1 = smf.logit("ever_csg7 ~ baseline_weight_kg + age + sex_female + C(study)",
                   data=ap).fit(disp=0)
    m1a = smf.logit("ever_csg7 ~ baseline_weight_kg + age + sex_female + C(study)"
                    " + baseline_diabetes + baseline_endocrine", data=ap).fit(disp=0)
    bw_u, bw_a = _or_row(m1, "baseline_weight_kg"), _or_row(m1a, "baseline_weight_kg")
    rows.append(dict(model="Antipsychotic cohort (CATIE+ACLAIMS)",
                     term="Baseline weight (+1 kg)",
                     OR_unadjusted=f"{bw_u[0]:.3f} ({bw_u[1]:.3f}–{bw_u[2]:.3f})",
                     OR_comorbidity_adjusted=f"{bw_a[0]:.3f} ({bw_a[1]:.3f}–{bw_a[2]:.3f})",
                     N=int(m1.nobs)))
    dia_or = _or_row(m1a, "baseline_diabetes")
    rows.append(dict(model="Antipsychotic cohort (CATIE+ACLAIMS)",
                     term="Baseline diabetes (adjusted model)",
                     OR_unadjusted="—",
                     OR_comorbidity_adjusted=f"{dia_or[0]:.3f} ({dia_or[1]:.3f}–{dia_or[2]:.3f}), p={dia_or[3]:.3f}",
                     N=int(m1a.nobs)))
    summ["baseline_weight_OR_unadj"] = round(bw_u[0], 3)
    summ["baseline_weight_OR_adj"] = round(bw_a[0], 3)

    # Model 2: within-CATIE olanzapine contrast, comorbidity-adjusted
    catie = ap[ap.study == "CATIE"].merge(kv, on="subject_id", how="inner")
    catie = catie.dropna(subset=["drug"])   # Phase-1-assigned only (matches primary contrast)
    catie["olanzapine"] = (catie["drug"] == "Olanzapine").astype(int)
    m2 = smf.logit("ever_csg7 ~ olanzapine + baseline_weight_kg + age + sex_female",
                   data=catie).fit(disp=0)
    m2a = smf.logit("ever_csg7 ~ olanzapine + baseline_weight_kg + age + sex_female"
                    " + baseline_diabetes + baseline_endocrine", data=catie).fit(disp=0)
    ol_u, ol_a = _or_row(m2, "olanzapine"), _or_row(m2a, "olanzapine")
    rows.append(dict(model="CATIE (within-antipsychotic)",
                     term="Olanzapine vs other antipsychotic",
                     OR_unadjusted=f"{ol_u[0]:.2f} ({ol_u[1]:.2f}–{ol_u[2]:.2f})",
                     OR_comorbidity_adjusted=f"{ol_a[0]:.2f} ({ol_a[1]:.2f}–{ol_a[2]:.2f})",
                     N=int(m2.nobs)))
    summ["olanzapine_OR_unadj"] = round(ol_u[0], 2)
    summ["olanzapine_OR_adj"] = round(ol_a[0], 2)
    return pd.DataFrame(rows), summ


# ── helper: per-class early 0-6mo slope from an arbitrary long df ──────────────
def _perclass_slopes_from_long(long_df, window=6.0):
    """0–6-month %/month slope per drug class, from a per-class random-slope LMM
    restricted to participants with ≥2 in-window observations. Used so scenario
    slopes (e.g. STEP re-anchored) are computed on a common, self-contained basis."""
    d = long_df[(long_df["t_months"] >= 0) & (long_df["t_months"] <= window)].copy()
    cnt = d.groupby(["study", "subject_id"])["t_months"].transform("size")
    d = d[cnt >= 2].copy()
    d["uid"] = d["study"].astype(str) + "::" + d["subject_id"].astype(str)
    out = {}
    for c in CLASS_ORDER:
        g = d[d["drug_class"] == c]
        try:
            m = smf.mixedlm("pct_change ~ t_months", g, groups=g["uid"],
                            re_formula="~t_months").fit(method="lbfgs", maxiter=300)
            out[c] = round(float(m.params["t_months"]), 3)
        except Exception:
            out[c] = None
    return out


# ── S12. Estimand sensitivity: baseline-only handling & STEP baseline anchor ───
def s12_estimand_sensitivity():
    """Robustness of the cumulative ≥7% class comparison to how the outcome and
    its baseline anchor are defined (Reviewer 2, comments 2 & 3):
      PRIMARY  : participants with ≥1 post-baseline weight (outcome observed);
      Sens. (i): baseline-only participants coded as non-events (all randomized) —
                 the previous primary handling, now demoted to a sensitivity;
      Sens.(ii): STEP-BD restricted to a confirmed protocol baseline (a month-0
                 weight required) and re-anchored to it, since 75% of STEP-BD have
                 no month-0 weight and are otherwise anchored to a later visit.
    Each scenario reports N, per-cohort ≥7% event rates, the pooled class ORs
    (vs antidepressant, same covariates as the primary logistic model) with 95%
    CIs, and per-cohort early 0–6-month slopes (%/month)."""

    prim_slopes = _perclass_slopes_from_long(long)   # primary (earliest-weight anchor)

    def row(tag, d, slopes):
        r = fit_class_logit(d, "ever_csg7")
        pct = {c: round(d.loc[d["drug_class"] == c, "ever_csg7"].mean() * 100, 1)
               for c in CLASS_ORDER}
        return dict(
            scenario=tag, N=r["N"],
            csg7_pct_antidepressant=pct["antidepressant"],
            csg7_pct_mood_stabilizer=pct["mood_stabilizer"],
            csg7_pct_antipsychotic=pct["antipsychotic"],
            MS_OR=r[MS_TERM][0], MS_lo=r[MS_TERM][1], MS_hi=r[MS_TERM][2], MS_p=r[MS_TERM][3],
            AP_OR=r[AP_TERM][0], AP_lo=r[AP_TERM][1], AP_hi=r[AP_TERM][2], AP_p=r[AP_TERM][3],
            slope_AD=slopes["antidepressant"], slope_MS=slopes["mood_stabilizer"],
            slope_AP=slopes["antipsychotic"])

    # STEP re-anchored to protocol-baseline window (month-0 weight required)
    step = long[long["study"] == "STEP"]
    has0 = step.groupby("subject_id")["time_months"].apply(lambda s: (s == 0).any())
    keep = set(has0[has0].index)
    bw0 = step[step["time_months"] == 0].groupby("subject_id")["weight_lbs"].mean()
    # re-anchored STEP long (baseline = month 0) for slope recomputation
    st_long = step[step["subject_id"].isin(keep)].copy()
    st_long["t_months"] = st_long["time_months"]                      # month 0 = baseline
    st_long["pct_change"] = 100.0 * (st_long["weight_lbs"] - st_long["subject_id"].map(bw0)) \
        / st_long["subject_id"].map(bw0)
    st_long = st_long[st_long["t_months"] >= 0]
    long_stepwin = pd.concat([long[long["study"] != "STEP"], st_long], ignore_index=True)
    stepwin_slopes = _perclass_slopes_from_long(long_stepwin)
    # re-anchored STEP subject-level ≥7% and baseline weight
    st_max = st_long.groupby("subject_id")["pct_change"].max()
    b_step = base[(base["study"] == "STEP") & (base["subject_id"].isin(keep))].copy()
    b_step["ever_csg7"] = (b_step["subject_id"].map(st_max) >= 7).astype(int)
    b_step["baseline_weight_kg"] = b_step["subject_id"].map(bw0) * LBS_TO_KG
    # keep only those with a post-baseline weight (consistent with the primary)
    st_npost = st_long[st_long["t_months"] > 0].groupby("subject_id").size()
    b_step = b_step[b_step["subject_id"].map(st_npost).fillna(0) >= 1]
    B = pd.concat([base_ana[base_ana["study"] != "STEP"], b_step], ignore_index=True)

    df = pd.DataFrame([
        row("Primary (≥1 post-baseline weight; earliest-weight baseline)", base_ana, prim_slopes),
        row("Sensitivity (i): baseline-only coded as non-events (all randomized)", base, prim_slopes),
        row("Sensitivity (ii): STEP-BD confirmed protocol baseline (month-0 weight)",
            B, stepwin_slopes),
    ])
    summary = dict(
        primary_MS_OR=round(df.iloc[0]["MS_OR"], 2), primary_AP_OR=round(df.iloc[0]["AP_OR"], 2),
        primary_MS_p=float(df.iloc[0]["MS_p"]), primary_AP_p=float(df.iloc[0]["AP_p"]),
        primary_N=int(df.iloc[0]["N"]),
        primary_slopes={k: prim_slopes[k] for k in CLASS_ORDER},
        allrand_MS_OR=round(df.iloc[1]["MS_OR"], 2), allrand_AP_OR=round(df.iloc[1]["AP_OR"], 2),
        allrand_MS_pct=float(df.iloc[1]["csg7_pct_mood_stabilizer"]),
        allrand_N=int(df.iloc[1]["N"]),
        stepwin_MS_OR=round(df.iloc[2]["MS_OR"], 2),
        stepwin_AP_OR=round(df.iloc[2]["AP_OR"], 2),
        stepwin_MS_lo=round(df.iloc[2]["MS_lo"], 2), stepwin_MS_hi=round(df.iloc[2]["MS_hi"], 2),
        stepwin_AP_lo=round(df.iloc[2]["AP_lo"], 2), stepwin_AP_hi=round(df.iloc[2]["AP_hi"], 2),
        stepwin_N=int(df.iloc[2]["N"]),
        stepwin_csg7_pct={"antidepressant": float(df.iloc[2]["csg7_pct_antidepressant"]),
                          "mood_stabilizer": float(df.iloc[2]["csg7_pct_mood_stabilizer"]),
                          "antipsychotic": float(df.iloc[2]["csg7_pct_antipsychotic"])},
        stepwin_slopes={k: stepwin_slopes[k] for k in CLASS_ORDER},
        MS_OR_range=[round(df["MS_OR"].min(), 2), round(df["MS_OR"].max(), 2)],
        AP_OR_range=[round(df["AP_OR"].min(), 2), round(df["AP_OR"].max(), 2)],
        n_nofu_excluded=int((base["max_followup_months"] <= 0).sum()),
        n_nofu_moodstab=int(((base["max_followup_months"] <= 0) &
                             (base["drug_class"] == "mood_stabilizer")).sum()),
        n_step_dropped=int((base["study"] == "STEP").sum() - len(keep)),
        n_step_kept=len(keep),
    )
    return df, summary


# ── STEP-BD baseline-timing: interval from study entry to first weight ─────────
def step_baseline_timing():
    """Distribution of the interval (months) between STEP-BD study entry (protocol
    month 0) and the first available weight — the analytic baseline. Quantifies the
    new-user vs prevalent-user concern (Reviewer 2, comment 3): most STEP-BD
    participants' analytic baseline falls after study entry."""
    step = long[long["study"] == "STEP"]
    first_t = step.groupby("subject_id")["time_months"].min()
    n = int(first_t.size)
    at0 = int((first_t == 0).sum())
    summary = dict(
        n_step=n,
        n_month0=at0, pct_month0=round(100 * at0 / n, 1),
        n_after0=n - at0, pct_after0=round(100 * (n - at0) / n, 1),
        median_months=round(float(first_t.median()), 1),
        q1_months=round(float(first_t.quantile(0.25)), 1),
        q3_months=round(float(first_t.quantile(0.75)), 1),
        max_months=round(float(first_t.max()), 1),
    )
    # small distribution table (binned)
    bins = [(-0.01, 0.5, "Month 0 (protocol baseline)"),
            (0.5, 1.5, "~1 month"), (1.5, 3.5, "~2–3 months"),
            (3.5, 6.5, "~4–6 months"), (6.5, 1e9, ">6 months")]
    rows = []
    for lo, hi, lab in bins:
        k = int(((first_t > lo) & (first_t <= hi)).sum())
        rows.append(dict(interval=lab, n=k, pct=round(100 * k / n, 1)))
    return pd.DataFrame(rows), summary


# ── S13. CO-MED arm-stratified sensitivity (monotherapy vs combination) ────────
def s13_comed_arms():
    """Reviewer 2, comment 1. The CO-MED antidepressant cohort is not
    combination-only: it includes an escitalopram + placebo SSRI-MONOTHERAPY arm.
    This tabulates, by randomized CO-MED arm and by regimen type (monotherapy vs
    combination), the N, N with ≥1 post-baseline weight, cumulative ≥7% gain,
    early 0–6-month slope, and ≥7% within the common 6-month window — to show
    which regimen drives the early antidepressant trajectory."""
    rand = pd.read_parquet(DATA / "parquet_COMED/RAND.parquet")
    arm_map = {"Escitalopram plus Orange capsule": "ESC monotherapy",
               "Bupropion  plus Blue capsule": "BUP-SR + escitalopram",
               "Venlafaxine plus Green capsule": "VEN-XR + mirtazapine"}
    regimen = {"ESC monotherapy": "Monotherapy",
               "BUP-SR + escitalopram": "Combination",
               "VEN-XR + mirtazapine": "Combination"}
    rand["arm"] = rand["Study arm"].map(arm_map)
    rand["subject_id"] = rand["Patient ID"].astype(str)
    amap = dict(zip(rand["subject_id"], rand["arm"]))

    cb = base[base["study"] == "COMED"].copy()
    cb["arm"] = cb["subject_id"].map(amap)
    cb["regimen"] = cb["arm"].map(regimen)
    # common 6-month window ≥7% (post-baseline within window)
    cw = common_window_data(6.0)
    cw6 = dict(zip(cw.loc[cw["study"] == "COMED", "subject_id"],
                   cw.loc[cw["study"] == "COMED", "csg7_6mo"]))
    cb["csg7_6mo"] = cb["subject_id"].map(cw6)
    # early 0-6mo slope per subgroup
    cl = long[long["study"] == "COMED"].copy()
    cl["arm"] = cl["subject_id"].map(amap)
    cl["regimen"] = cl["arm"].map(regimen)

    def _slope(ids):
        d = cl[(cl["subject_id"].isin(ids)) & (cl["t_months"] <= 6.0)]
        try:
            m = smf.mixedlm("pct_change ~ t_months", d, groups=d["subject_id"],
                            re_formula="~t_months").fit(method="lbfgs", maxiter=300)
            return round(float(m.params["t_months"]), 3)
        except Exception:
            return None

    def _grp(mask, label, kind):
        g = cb[mask]
        ana = g[g["analyzable"]]
        n_meas = int(cl[cl["subject_id"].isin(set(g["subject_id"]))].shape[0])
        cwv = g["csg7_6mo"].dropna()
        return dict(
            group=label, kind=kind, N=len(g), n_postbaseline=int(g["analyzable"].sum()),
            n_weight_measurements=n_meas,
            csg7_ever_n=int(ana["ever_csg7"].sum()),
            csg7_ever_pct=round(ana["ever_csg7"].mean() * 100, 1),
            mean_max_gain_pct=round(ana["max_pct_gain"].mean(), 2),
            slope_0_6mo=_slope(set(ana["subject_id"])),
            csg7_6mo_n=int(cwv.sum()), csg7_6mo_denom=int(cwv.notna().sum()),
            csg7_6mo_pct=round(cwv.mean() * 100, 1) if len(cwv) else None)

    rows = []
    for arm in ["ESC monotherapy", "BUP-SR + escitalopram", "VEN-XR + mirtazapine"]:
        rows.append(_grp(cb["arm"] == arm, arm, "arm"))
    rows.append(_grp(cb["regimen"] == "Monotherapy", "Monotherapy (pooled)", "regimen"))
    rows.append(_grp(cb["regimen"] == "Combination", "Combination (pooled)", "regimen"))
    df = pd.DataFrame(rows)

    d = {r["group"]: r for r in rows}
    summary = dict(
        n_esc_mono=d["ESC monotherapy"]["N"],
        n_bup_esc=d["BUP-SR + escitalopram"]["N"],
        n_ven_mirt=d["VEN-XR + mirtazapine"]["N"],
        n_combination=d["Combination (pooled)"]["N"],
        n_monotherapy=d["Monotherapy (pooled)"]["N"],
        slope_esc_mono=d["ESC monotherapy"]["slope_0_6mo"],
        slope_bup_esc=d["BUP-SR + escitalopram"]["slope_0_6mo"],
        slope_ven_mirt=d["VEN-XR + mirtazapine"]["slope_0_6mo"],
        slope_monotherapy=d["Monotherapy (pooled)"]["slope_0_6mo"],
        slope_combination=d["Combination (pooled)"]["slope_0_6mo"],
        csg7_ever_esc_mono=d["ESC monotherapy"]["csg7_ever_pct"],
        csg7_ever_bup_esc=d["BUP-SR + escitalopram"]["csg7_ever_pct"],
        csg7_ever_ven_mirt=d["VEN-XR + mirtazapine"]["csg7_ever_pct"],
        csg7_6mo_esc_mono=d["ESC monotherapy"]["csg7_6mo_pct"],
        csg7_6mo_ven_mirt=d["VEN-XR + mirtazapine"]["csg7_6mo_pct"],
        csg7_6mo_monotherapy=d["Monotherapy (pooled)"]["csg7_6mo_pct"],
        csg7_6mo_combination=d["Combination (pooled)"]["csg7_6mo_pct"],
    )
    return df, summary


def main():
    print("=== Sensitivity analyses ===")
    print("\nS1. Leave-one-study-out (≥7% logistic)")
    loso = s1_loso()
    print(loso.to_string(index=False))

    print("\nS2. Random-effects meta-analysis of early slope (within class)")
    meta = s2_meta_slope()
    print(meta.to_string(index=False))

    print("\nS3. Common 6-month-window ≥7% binary")
    s3_pct, s3_counts, s3_model, s3_n = s3_common_window()
    print("  ≥7% within 6mo by class:", s3_pct)
    print("  event counts / denominators:", s3_counts)
    print(s3_model.to_string(index=False))

    print("\nS4. Threshold sensitivity")
    thr = s4_thresholds()
    print(thr.pivot(index="drug_class", columns="threshold_pct", values="gain_pct").to_string())

    print("\nS5. Slope-window sensitivity")
    sw = s5_slope_windows()
    print(sw.pivot(index="window_months", columns="drug_class", values="slope_pct_per_month").to_string())

    print("\nS6. Predictor models: cumulative ever-≥7% vs 3-month landmark")
    s6 = s6_predictor_models()
    print(s6.to_string(index=False))

    print("\nS7. Study-clustered covariance sensitivity")
    s7 = s7_clustered_logit()
    print(s7.to_string(index=False))

    print("\nS8. Adjusted early-slope (0-6mo) mixed model")
    s8 = s8_adjusted_slope()
    print(s8.to_string(index=False))

    print("\nS9. Follow-up completeness / missingness by trial")
    s9 = s9_followup_completeness()
    print(s9.to_string(index=False))

    print("\nS10. Absolute weight-change sensitivity (baseline-weight predictor)")
    s10, s10_summary = s10_absolute_change()
    print(s10.to_string(index=False))

    print("\nS11. Within-antipsychotic comorbidity-adjusted sensitivity")
    s11, s11_summary = s11_comorbidity_sensitivity()
    print(s11.to_string(index=False))
    print("  prevalence:", s11_summary["prevalence"])

    print("\nS12. Estimand sensitivity: baseline-only handling & STEP baseline anchor")
    s12, s12_summary = s12_estimand_sensitivity()
    print(s12.to_string(index=False))

    print("\nSTEP-BD baseline timing (entry → first weight)")
    step_tim, step_tim_summary = step_baseline_timing()
    print(step_tim.to_string(index=False))
    print("  summary:", step_tim_summary)

    print("\nS13. CO-MED arm-stratified sensitivity (monotherapy vs combination)")
    s13, s13_summary = s13_comed_arms()
    print(s13.to_string(index=False))

    with pd.ExcelWriter(OUT / "sensitivity_results.xlsx", engine="openpyxl") as w:
        write_pretty_sheet(w, loso, "S1_LOSO")
        write_pretty_sheet(w, meta, "S2_meta_slope")
        write_pretty_sheet(w, s3_model, "S3_common_window")
        write_pretty_sheet(w, thr, "S4_thresholds")
        write_pretty_sheet(w, sw, "S5_slope_windows")
        write_pretty_sheet(w, s6, "S6_predictor_models")
        write_pretty_sheet(w, s7, "S7_clustered_logit")
        write_pretty_sheet(w, s8, "S8_adjusted_slope")
        write_pretty_sheet(w, s9, "S9_followup_complete")
        write_pretty_sheet(w, s10, "S10_absolute_change")
        write_pretty_sheet(w, s11, "S11_comorbidity_adj")
        write_pretty_sheet(w, s12, "S12_estimand_sensitivity")
        write_pretty_sheet(w, step_tim, "S12b_STEP_baseline_timing")
        write_pretty_sheet(w, s13, "S13_comed_arms")

    summary = {
        "S1_loso_AP_OR_range": [round(loso["AP_OR"].min(), 2), round(loso["AP_OR"].max(), 2)],
        "S1_loso_MS_OR_range": [round(loso["MS_OR"].min(), 2), round(loso["MS_OR"].max(), 2)],
        "S2_meta_pooled_slope": {r["drug_class"]: round(r["pooled"], 3) for _, r in meta.iterrows()},
        "S2_meta_I2": {r["drug_class"]: (None if pd.isna(r["I2"]) else round(r["I2"], 1))
                       for _, r in meta.iterrows()},
        "S3_common6mo_csg7_pct_by_class": s3_pct,
        "S3_common6mo_counts": s3_counts,
        "S3_common6mo_AP_OR": round(float(s3_model.iloc[0]["OR"]), 2),
        "S3_common6mo_AP_ci": [round(float(s3_model.iloc[0]["ci_low"]), 2),
                               round(float(s3_model.iloc[0]["ci_high"]), 2)],
        "S3_common6mo_AP_p": float(s3_model.iloc[0]["p"]),
        "S3_common6mo_MS_OR": round(float(s3_model.iloc[1]["OR"]), 2),
        "S3_common6mo_MS_ci": [round(float(s3_model.iloc[1]["ci_low"]), 2),
                               round(float(s3_model.iloc[1]["ci_high"]), 2)],
        "S3_common6mo_MS_p": float(s3_model.iloc[1]["p"]),
        "S3_N": s3_n,
        "S4_thresholds": thr.to_dict("records"),
        "S5_slope_windows": sw.to_dict("records"),
        "S7_clustered_logit": s7.to_dict("records"),
        "S8_adjusted_slope": s8.to_dict("records"),
        "S8_adjusted_slope_by_class": {
            r["drug_class"]: round(float(r["adj_slope_pct_per_month"]), 2)
            for _, r in s8.iterrows()},
        "S8_adj_AP_vs_AD_p": next(
            (float(r["p_vs_antidepressant"]) for _, r in s8.iterrows()
             if r["drug_class"] == "antipsychotic"), None),
        "S8_adj_MS_vs_AD_p": next(
            (float(r["p_vs_antidepressant"]) for _, r in s8.iterrows()
             if r["drug_class"] == "mood_stabilizer"), None),
        "S9_followup_complete": s9.to_dict("records"),
        "S10_absolute_change": s10_summary,
        "S11_comorbidity_adj": s11_summary,
        "S12_estimand_sensitivity": s12_summary,
        "STEP_baseline_timing": step_tim_summary,
        "S13_comed_arms": s13_summary,
    }
    (OUT / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nSaved sensitivity_results.xlsx, sensitivity_summary.json")


if __name__ == "__main__":
    main()
