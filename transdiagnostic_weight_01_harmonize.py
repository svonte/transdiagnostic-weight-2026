"""
transdiagnostic_weight_01_harmonize.py
Extract and harmonize longitudinal WEIGHT trajectories + baseline covariates
from the 5 studies with body-weight data:
  CATIE (schizophrenia, antipsychotic oral)
  ACLAIMS (schizophrenia, antipsychotic LAI)
  CO-MED (MDD, antidepressant)
  STEP   (bipolar, mood stabilizer)
  LiTMUS (bipolar, mood stabilizer)

Spec: spec/hypotheses/transdiagnostic_weight.yaml

Outputs (results/transdiagnostic_weight/):
  weight_long.parquet  - one row per subject x visit (long format)
  baseline.parquet     - one row per subject (covariates + derived outcomes)

All weights are recorded in POUNDS in every study (verified). The primary
%-change outcome is unit-invariant; weight_kg is added for descriptive means.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "results" / "transdiagnostic_weight"
OUT.mkdir(parents=True, exist_ok=True)

LBS_TO_KG = 0.45359237
WEEKS_PER_MONTH = 4.34524
PLAUSIBLE_LBS = (50.0, 700.0)   # filter data-entry errors (e.g. LiTMUS 1 lb)
CSG_THRESHOLD = 7.0             # clinically significant weight gain (%)

# Study -> (diagnosis, drug_class, formulation)
STUDY_META = {
    "CATIE":   ("schizophrenia", "antipsychotic",   "oral"),
    "ACLAIMS": ("schizophrenia", "antipsychotic",   "lai"),
    "COMED":   ("mdd",           "antidepressant",  "oral"),
    "STEP":    ("bipolar",       "mood_stabilizer", "oral"),
    "LiTMUS":  ("bipolar",       "mood_stabilizer", "oral"),
}


def recode_sex_female(val):
    """Project convention (spec/skills/recoding.md): 0=male, 1=female, NaN=missing.
    Handles study-specific raw codes. STEP/ACLAIMS use 1=male,2=female; CO-MED
    uses 1=male,0=female; CATIE/LiTMUS use text -> normalized below per study."""
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in ("m", "male"):
        return 0
    if v in ("f", "female"):
        return 1
    return np.nan


def clean_lbs(s):
    """Numeric coercion + plausible-range filter (lbs)."""
    x = pd.to_numeric(s, errors="coerce")
    return x.where((x >= PLAUSIBLE_LBS[0]) & (x <= PLAUSIBLE_LBS[1]))


# ---------------------------------------------------------------------------
# Per-study LONGITUDINAL weight extractors -> long df:
#   subject_id, study, time_months, weight_lbs
# ---------------------------------------------------------------------------

def long_catie():
    v = pd.read_parquet(DATA / "parquet_catie/VITAL.parquet")
    out = pd.DataFrame({
        "subject_id": v["NIMHID"].astype(str),
        "time_months": pd.to_numeric(v["visit_month"], errors="coerce"),
        "weight_lbs": clean_lbs(v["Weight (lbs)"]),
    })
    out["study"] = "CATIE"
    return out.dropna(subset=["time_months", "weight_lbs"])


def long_aclaims():
    v = pd.read_parquet(DATA / "parquet_Aclaims/vsf.parquet")
    out = pd.DataFrame({
        "subject_id": v["patient_id"].astype(str),
        "time_months": pd.to_numeric(v["study_month"], errors="coerce"),
        "weight_lbs": clean_lbs(v["vsf_weight_lbs"]),
    })
    out["study"] = "ACLAIMS"
    return out.dropna(subset=["time_months", "weight_lbs"])


def long_comed():
    v = pd.read_parquet(DATA / "parquet_COMED/VS.parquet")
    out = pd.DataFrame({
        "subject_id": v["Patient ID"].astype(str),
        "time_months": pd.to_numeric(v["Protocol Week"], errors="coerce") / WEEKS_PER_MONTH,
        "weight_lbs": clean_lbs(v["Weight"]),
    })
    out["study"] = "COMED"
    return out.dropna(subset=["time_months", "weight_lbs"])


def long_step():
    frames = []
    for fn in ["CMF_1.parquet", "CMF_2.parquet"]:
        cmf = pd.read_parquet(DATA / f"parquet_step/{fn}")
        wcols = [c for c in cmf.columns
                 if c.startswith("CMF Month ") and c.endswith(" Weight") and "Repeat" not in c]
        for c in wcols:
            # "CMF Month 02 Weight" -> 2
            mm = int(c.split("Month ")[1].split(" ")[0])
            sub = pd.DataFrame({
                "subject_id": cmf["STEPID"].astype(str),
                "time_months": float(mm),
                "weight_lbs": clean_lbs(cmf[c]),
            })
            frames.append(sub.dropna(subset=["weight_lbs"]))
    out = pd.concat(frames, ignore_index=True)
    out["study"] = "STEP"
    return out


def long_litmus():
    d = pd.read_parquet(DATA / "parquet_litmus/derived.parquet")
    sid = d["NIMH_ID#"].astype(str)
    frames = []
    for oo in range(1, 11):
        tag = f"(Obs {oo:02d})"
        wcol = f"VS | Weight {tag}"
        tcol = f"VS | Study week {tag}"
        if wcol not in d.columns or tcol not in d.columns:
            continue
        sub = pd.DataFrame({
            "subject_id": sid,
            "time_months": pd.to_numeric(d[tcol], errors="coerce") / WEEKS_PER_MONTH,
            "weight_lbs": clean_lbs(d[wcol]),
        })
        frames.append(sub.dropna(subset=["time_months", "weight_lbs"]))
    out = pd.concat(frames, ignore_index=True)
    out["study"] = "LiTMUS"
    return out


# ---------------------------------------------------------------------------
# Per-study BASELINE covariate extractors -> df: subject_id, age, sex_male
# ---------------------------------------------------------------------------

def base_catie():
    demo = pd.read_parquet(DATA / "parquet_catie/DEMO.parquet")
    df = pd.DataFrame({
        "subject_id": demo["NIMHID"].astype(str),
        "age": pd.to_numeric(demo["AGE"], errors="coerce"),
        "sex_female": demo["GENDER"].map({"Male": 0, "Female": 1}),
    })
    df["study"] = "CATIE"
    return df


def base_aclaims():
    dem = pd.read_parquet(DATA / "parquet_Aclaims/dem.parquet")
    dem_bl = dem.sort_values("visit_num").groupby("patient_id").first().reset_index()
    df = pd.DataFrame({
        "subject_id": dem_bl["patient_id"].astype(str),
        "age": pd.to_numeric(dem_bl["dem_age"], errors="coerce"),
        "sex_female": dem_bl["dem_sex"].apply(recode_sex_female),   # M/F
    })
    df["study"] = "ACLAIMS"
    return df


def base_comed():
    cddf = pd.read_parquet(DATA / "parquet_COMED/CDDF.parquet")
    bl = cddf[cddf["Protocol Week"] == 0].groupby("Patient ID").first().reset_index()
    df = pd.DataFrame({
        "subject_id": bl["Patient ID"].astype(str),
        "age": pd.to_numeric(bl["Age"], errors="coerce"),
        # CO-MED coding note: original SAS Code Book defines SEX 1=Male,2=Female,
        # but the PARQUET stores 0/1 (transformed, no longer matching the codebook).
        # The majority class is 1 (338/482=70%), matching the CO-MED trial's
        # ~73% female (Rush 2011). Therefore in the parquet 1=female, 0=male,
        # which already equals the project convention (0=male, 1=female).
        "sex_female": bl["Gender"].map({1: 1, 0: 0, "Male": 0, "Female": 1}),
    })
    df["study"] = "COMED"
    return df


def base_step():
    en = pd.read_parquet(DATA / "parquet_step/ENROL.parquet")
    df = pd.DataFrame({
        "subject_id": en["STEPID"].astype(str),
        "age": pd.to_numeric(en["ENROL Age"], errors="coerce"),
        # STEP raw Gender 1=male, 2=female -> project convention 0=male, 1=female
        "sex_female": pd.to_numeric(en["ENROL Gender"], errors="coerce").map({1: 0, 2: 1}),
    })
    df["study"] = "STEP"
    return df


def base_litmus():
    d = pd.read_parquet(DATA / "parquet_litmus/derived.parquet")
    df = pd.DataFrame({
        "subject_id": d["NIMH_ID#"].astype(str),
        "age": pd.to_numeric(d["DEM | Age at time of randomization"], errors="coerce"),
        # LiTMUS raw Sex 1=male, 2=female -> project convention 0=male, 1=female
        "sex_female": pd.to_numeric(d["DEM | Sex"], errors="coerce").map({1: 0, 2: 1}),
    })
    df["study"] = "LiTMUS"
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Extracting longitudinal weight (5 studies) ===\n")
    long_extractors = [long_catie, long_aclaims, long_comed, long_step, long_litmus]
    base_extractors = [base_catie, base_aclaims, base_comed, base_step, base_litmus]

    long = pd.concat([fn() for fn in long_extractors], ignore_index=True)
    base = pd.concat([fn() for fn in base_extractors], ignore_index=True)

    # Attach study meta
    long["diagnosis"]   = long["study"].map(lambda s: STUDY_META[s][0])
    long["drug_class"]  = long["study"].map(lambda s: STUDY_META[s][1])
    long["formulation"] = long["study"].map(lambda s: STUDY_META[s][2])
    long["weight_kg"]   = long["weight_lbs"] * LBS_TO_KG

    # ---- baseline weight = earliest visit per subject ----
    long = long.sort_values(["study", "subject_id", "time_months"])
    first = (long.groupby(["study", "subject_id"], as_index=False)
                  .agg(baseline_time=("time_months", "min")))
    long = long.merge(first, on=["study", "subject_id"], how="left")
    bw = (long[long["time_months"] == long["baseline_time"]]
              .groupby(["study", "subject_id"], as_index=False)
              .agg(baseline_weight_lbs=("weight_lbs", "mean")))
    long = long.merge(bw, on=["study", "subject_id"], how="left")

    # time re-centred so baseline = 0
    long["t_months"] = long["time_months"] - long["baseline_time"]
    # unit-invariant percent change from baseline
    long["pct_change"] = 100.0 * (long["weight_lbs"] - long["baseline_weight_lbs"]) / long["baseline_weight_lbs"]

    # ---- per-subject derived outcomes ----
    g = long.groupby(["study", "subject_id"])
    subj = g.agg(
        n_weight_obs=("weight_lbs", "size"),
        baseline_weight_lbs=("baseline_weight_lbs", "first"),
        max_pct_gain=("pct_change", "max"),
        min_pct_gain=("pct_change", "min"),
        last_pct_change=("pct_change", "last"),
        max_followup_months=("t_months", "max"),
    ).reset_index()
    subj["ever_csg7"] = (subj["max_pct_gain"] >= CSG_THRESHOLD).astype(int)
    subj["baseline_weight_kg"] = subj["baseline_weight_lbs"] * LBS_TO_KG

    # merge demographics
    subj = subj.merge(base, on=["study", "subject_id"], how="left")
    subj["diagnosis"]   = subj["study"].map(lambda s: STUDY_META[s][0])
    subj["drug_class"]  = subj["study"].map(lambda s: STUDY_META[s][1])
    subj["formulation"] = subj["study"].map(lambda s: STUDY_META[s][2])

    # ---- save ----
    long_cols = ["subject_id", "study", "diagnosis", "drug_class", "formulation",
                 "time_months", "t_months", "weight_lbs", "weight_kg",
                 "baseline_weight_lbs", "pct_change"]
    long_out = long[long_cols].copy()
    long_out.to_parquet(OUT / "weight_long.parquet", index=False)
    subj.to_parquet(OUT / "baseline.parquet", index=False)

    # ---- report ----
    print(f"weight_long.parquet : {len(long_out):,} visit-rows")
    print(f"baseline.parquet    : {len(subj):,} subjects\n")
    print("Per-study counts (subjects with >=1 weight; >=2 weight for slope):")
    for s in ["CATIE", "ACLAIMS", "COMED", "STEP", "LiTMUS"]:
        ss = subj[subj["study"] == s]
        ge2 = (ss["n_weight_obs"] >= 2).sum()
        print(f"  {s:8s} n={len(ss):4d}  >=2obs={ge2:4d}  "
              f"csg7={int(ss['ever_csg7'].sum()):4d} ({ss['ever_csg7'].mean()*100:4.1f}%)  "
              f"median max_followup={ss['max_followup_months'].median():.1f}mo  "
              f"age_miss={int(ss['age'].isna().sum())} sex_miss={int(ss['sex_female'].isna().sum())} "
              f"%female={ss['sex_female'].mean()*100:.0f}")
    print("\nBy drug class:")
    for c in ["antipsychotic", "antidepressant", "mood_stabilizer"]:
        cc = subj[subj["drug_class"] == c]
        print(f"  {c:16s} n={len(cc):4d}  csg7={cc['ever_csg7'].mean()*100:4.1f}%  "
              f"mean max_pct_gain={cc['max_pct_gain'].mean():+.1f}%")
    warn = int(subj["baseline_weight_lbs"].isna().sum())
    print(f"\nWarnings: {warn} subjects without usable baseline weight")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
