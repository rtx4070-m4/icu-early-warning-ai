"""
AI Hospital Operating System
Data Engineering – Feature Engineering
=========================================
Computes clinical risk scores, delta features,
rolling statistics, and SOFA/NEWS2/qSOFA scores.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("feature_engineering")


# ─────────────────────────────────────────────
# Clinical Scoring Functions
# ─────────────────────────────────────────────

def compute_sofa(row: pd.Series) -> Dict[str, float]:
    """
    Sequential Organ Failure Assessment (SOFA) score.
    Each sub-score: 0-4 (higher = worse).
    """
    # Respiratory (approximated from SpO2)
    spo2 = row.get("spo2", np.nan)
    if pd.isna(spo2):     resp = 0
    elif spo2 >= 95:      resp = 0
    elif spo2 >= 90:      resp = 1
    elif spo2 >= 85:      resp = 2
    elif spo2 >= 80:      resp = 3
    else:                 resp = 4

    # Cardiovascular (MAP)
    map_ = row.get("map", np.nan)
    if pd.isna(map_):     cardio = 0
    elif map_ >= 70:      cardio = 0
    elif map_ >= 65:      cardio = 1
    elif map_ >= 50:      cardio = 2
    else:                 cardio = 3

    # Renal (Creatinine)
    cr = row.get("Creatinine", np.nan)
    if pd.isna(cr):       renal = 0
    elif cr < 1.2:        renal = 0
    elif cr < 2.0:        renal = 1
    elif cr < 3.5:        renal = 2
    elif cr < 5.0:        renal = 3
    else:                 renal = 4

    # Coagulation (Platelets)
    plt_ = row.get("Platelets", np.nan)
    if pd.isna(plt_):     coag = 0
    elif plt_ >= 150:     coag = 0
    elif plt_ >= 100:     coag = 1
    elif plt_ >= 50:      coag = 2
    elif plt_ >= 20:      coag = 3
    else:                 coag = 4

    # Hepatic (Bilirubin – approximated as 0 if missing)
    liver = 0  # Bilirubin not always available

    # Neurological (GCS)
    gcs = row.get("gcs_total", np.nan)
    if pd.isna(gcs):      neuro = 0
    elif gcs == 15:       neuro = 0
    elif gcs >= 13:       neuro = 1
    elif gcs >= 10:       neuro = 2
    elif gcs >= 6:        neuro = 3
    else:                 neuro = 4

    total = resp + cardio + renal + coag + liver + neuro
    return {
        "sofa_resp":   resp,
        "sofa_cardio": cardio,
        "sofa_renal":  renal,
        "sofa_coag":   coag,
        "sofa_neuro":  neuro,
        "sofa_total":  total,
    }


def compute_news2(row: pd.Series) -> int:
    """
    National Early Warning Score 2 (NEWS2).
    Range 0-20; ≥5 triggers clinical review.
    """
    score = 0

    # Respiratory rate
    rr = row.get("resp_rate", np.nan)
    if not pd.isna(rr):
        if rr <= 8:              score += 3
        elif rr <= 11:           score += 1
        elif rr <= 20:           score += 0
        elif rr <= 24:           score += 2
        else:                    score += 3

    # SpO2
    spo2 = row.get("spo2", np.nan)
    if not pd.isna(spo2):
        if spo2 <= 91:           score += 3
        elif spo2 <= 93:         score += 2
        elif spo2 <= 95:         score += 1
        else:                    score += 0

    # Systolic BP
    sbp = row.get("sbp", np.nan)
    if not pd.isna(sbp):
        if sbp <= 90:            score += 3
        elif sbp <= 100:         score += 2
        elif sbp <= 110:         score += 1
        elif sbp <= 219:         score += 0
        else:                    score += 3

    # Heart rate
    hr = row.get("heart_rate", np.nan)
    if not pd.isna(hr):
        if hr <= 40:             score += 3
        elif hr <= 50:           score += 1
        elif hr <= 90:           score += 0
        elif hr <= 110:          score += 1
        elif hr <= 130:          score += 2
        else:                    score += 3

    # Temperature
    temp = row.get("temperature", np.nan)
    if not pd.isna(temp):
        if temp <= 35.0:         score += 3
        elif temp <= 36.0:       score += 1
        elif temp <= 38.0:       score += 0
        elif temp <= 39.0:       score += 1
        else:                    score += 2

    # Consciousness (GCS proxy: any impairment = +3)
    gcs = row.get("gcs_total", np.nan)
    if not pd.isna(gcs) and gcs < 15:
        score += 3

    return score


def compute_qsofa(row: pd.Series) -> int:
    """
    Quick SOFA (qSOFA) – 3 criteria, 0-3.
    Score ≥2 indicates high risk of sepsis-related organ dysfunction.
    """
    score = 0
    if row.get("sbp", 200) <= 100:        score += 1
    if row.get("resp_rate", 0) >= 22:      score += 1
    if row.get("gcs_total", 15) < 15:      score += 1
    return score


def compute_apsiii(row: pd.Series) -> float:
    """
    Simplified APACHE III proxy using available vitals.
    (Full APACHE III requires 17 variables; we compute partial.)
    """
    score = 0.0

    hr = row.get("heart_rate", np.nan)
    if not pd.isna(hr):
        if hr < 40 or hr >= 155:  score += 17
        elif hr < 60 or hr >= 130: score += 8
        elif hr < 70 or hr >= 120: score += 4

    sbp = row.get("sbp", np.nan)
    if not pd.isna(sbp):
        if sbp < 70 or sbp >= 180: score += 13
        elif sbp < 80 or sbp >= 160: score += 7
        elif sbp < 100 or sbp >= 140: score += 2

    temp = row.get("temperature", np.nan)
    if not pd.isna(temp):
        if temp < 33 or temp >= 41: score += 20
        elif temp < 35 or temp >= 39: score += 4

    rr = row.get("resp_rate", np.nan)
    if not pd.isna(rr):
        if rr < 6 or rr >= 50: score += 17
        elif rr < 12 or rr >= 35: score += 8

    spo2 = row.get("spo2", np.nan)
    if not pd.isna(spo2):
        if spo2 < 80:  score += 15
        elif spo2 < 90: score += 7

    return score


# ─────────────────────────────────────────────
# Time-Series Delta Features
# ─────────────────────────────────────────────
def compute_deltas(
    df: pd.DataFrame,
    group_col: str = "patient_id",
    time_col: str = "charttime",
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute difference (delta) from previous observation per patient.
    Also compute rate-of-change per hour.
    """
    if cols is None:
        cols = ["heart_rate", "sbp", "spo2", "resp_rate", "map"]
    cols = [c for c in cols if c in df.columns]

    df = df.copy().sort_values([group_col, time_col])

    for col in cols:
        df[f"delta_{col}"] = (
            df.groupby(group_col)[col]
            .diff()
            .reset_index(drop=True)
        )

        # Hours between observations
        dt_hours = (
            df.groupby(group_col)[time_col]
            .diff()
            .dt.total_seconds()
            .div(3600)
            .reset_index(drop=True)
        )
        df[f"rate_{col}_per_hr"] = df[f"delta_{col}"] / dt_hours.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────
# Rolling Statistics
# ─────────────────────────────────────────────
def compute_rolling_stats(
    df: pd.DataFrame,
    group_col: str = "patient_id",
    time_col: str = "charttime",
    windows: List[int] = [3, 6, 12],  # hours
) -> pd.DataFrame:
    """
    Compute rolling mean, std, min, max for each vital
    over specified time windows (in hours).
    Handles irregular time series via time-based rolling.
    """
    vital_cols = ["heart_rate", "sbp", "spo2", "resp_rate", "map", "temperature"]
    vital_cols = [c for c in vital_cols if c in df.columns]

    df = df.copy().sort_values([group_col, time_col])

    result_frames = []
    for pid, group in df.groupby(group_col):
        group = group.set_index(time_col).copy()
        for w in windows:
            roll = group[vital_cols].rolling(window=f"{w}H", min_periods=1)
            group[[f"{c}_mean_{w}h" for c in vital_cols]] = roll.mean()
            group[[f"{c}_std_{w}h"  for c in vital_cols]] = roll.std()
            group[[f"{c}_min_{w}h"  for c in vital_cols]] = roll.min()
            group[[f"{c}_max_{w}h"  for c in vital_cols]] = roll.max()
        group[group_col] = pid
        result_frames.append(group.reset_index())

    result = pd.concat(result_frames, ignore_index=True)
    logger.info("Rolling stats computed for %d patients, windows: %s", len(result[group_col].unique()), windows)
    return result


# ─────────────────────────────────────────────
# Derived Clinical Features
# ─────────────────────────────────────────────
def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute clinically meaningful derived features:
    shock index, pulse pressure, AVPU, hypotension flag, etc.
    """
    df = df.copy()

    # Shock Index = HR / SBP (>1.0 suggests haemorrhagic shock)
    if "heart_rate" in df.columns and "sbp" in df.columns:
        df["shock_index"] = df["heart_rate"] / df["sbp"].replace(0, np.nan)

    # Pulse Pressure = SBP – DBP (< 25 mmHg → cardiogenic shock)
    if "sbp" in df.columns and "dbp" in df.columns:
        df["pulse_pressure"] = df["sbp"] - df["dbp"]

    # Perfusion Index proxy (SpO2 variation proxy)
    if "spo2" in df.columns:
        df["hypoxia_flag"]    = (df["spo2"] < 92).astype(int)
        df["severe_hypoxia"]  = (df["spo2"] < 85).astype(int)

    # Haemodynamic flags
    if "sbp" in df.columns:
        df["hypotension_flag"]      = (df["sbp"] < 90).astype(int)
        df["severe_hypotension"]    = (df["sbp"] < 70).astype(int)

    if "heart_rate" in df.columns:
        df["tachycardia_flag"]  = (df["heart_rate"] > 100).astype(int)
        df["bradycardia_flag"]  = (df["heart_rate"] < 50).astype(int)

    if "resp_rate" in df.columns:
        df["tachypnea_flag"]    = (df["resp_rate"] > 22).astype(int)

    if "temperature" in df.columns:
        df["fever_flag"]        = (df["temperature"] > 38.3).astype(int)
        df["hypothermia_flag"]  = (df["temperature"] < 36.0).astype(int)

    # Combined SIRS criteria count
    sirs_cols = ["tachycardia_flag", "tachypnea_flag", "fever_flag"]
    present   = [c for c in sirs_cols if c in df.columns]
    if present:
        df["sirs_count"] = df[present].sum(axis=1)
        df["sirs_met"]   = (df["sirs_count"] >= 2).astype(int)

    return df


# ─────────────────────────────────────────────
# Clinical Score Batch Computation
# ─────────────────────────────────────────────
def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SOFA, NEWS2, qSOFA, and APSIII to every row.
    Also compute derived features and delta features.
    """
    logger.info("Computing clinical scores for %d records", len(df))

    # SOFA
    sofa_df = df.apply(lambda r: pd.Series(compute_sofa(r)), axis=1)
    df = pd.concat([df, sofa_df], axis=1)

    # NEWS2 and qSOFA
    df["news2"]  = df.apply(compute_news2,  axis=1)
    df["qsofa"]  = df.apply(compute_qsofa,  axis=1)
    df["apsiii"] = df.apply(compute_apsiii, axis=1)

    # Derived flags
    df = compute_derived_features(df)

    # Alert levels
    df["news2_alert"] = pd.cut(
        df["news2"],
        bins=[-1, 4, 6, 20],
        labels=["low","medium","high"],
    )
    df["sofa_sepsis_likely"] = (df["sofa_total"] >= 2).astype(int)

    logger.info("Scoring complete. Columns added: sofa_total, news2, qsofa, apsiii")
    return df


# ─────────────────────────────────────────────
# Full Feature Engineering Pipeline
# ─────────────────────────────────────────────
class FeatureEngineeringPipeline:

    def run(
        self,
        vitals_df: pd.DataFrame,
        labs_wide_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Args
        ----
        vitals_df   : Cleaned vitals (long format, one row per observation)
        labs_wide_df: Wide-format labs (one row per patient; optional)

        Returns
        -------
        Feature-rich DataFrame ready for ML models.
        """
        logger.info("Feature engineering pipeline started")
        df = vitals_df.copy()

        # 1. Merge labs if available
        if labs_wide_df is not None and not labs_wide_df.empty:
            df = df.merge(labs_wide_df, on="patient_id", how="left")
            logger.info("Labs merged into vitals dataframe")

        # 2. Delta features
        df = compute_deltas(df)

        # 3. Rolling statistics
        df = compute_rolling_stats(df, windows=[3, 6, 12])

        # 4. Clinical scores
        df = compute_all_scores(df)

        logger.info(
            "Feature engineering complete: %d rows × %d columns",
            len(df), len(df.columns),
        )
        return df


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_engineering.load_data import SyntheticDataGenerator
    from data_engineering.preprocessing import PreprocessingPipeline

    gen     = SyntheticDataGenerator(seed=99)
    vitals  = gen.generate_vitals(n_patients=5, hours=24)
    labs    = gen.generate_labs(n_patients=5)

    prep    = PreprocessingPipeline(window_hours=6)
    vitals  = prep.fit_transform_vitals(vitals)["df"]

    # Wide labs
    labs_wide = labs.pivot_table(
        index="patient_id", columns="lab_name", values="value", aggfunc="mean"
    ).reset_index()

    pipe    = FeatureEngineeringPipeline()
    features = pipe.run(vitals, labs_wide)

    print(f"\nFinal feature matrix: {features.shape}")
    print(features[["patient_id","sofa_total","news2","qsofa","shock_index"]].head(10).to_string())


# ── Test-compatible helpers ────────────────────────────────────────────────────

def compute_sofa_cardiovascular(map_value: float) -> int:
    """Return cardiovascular SOFA component (0-4) from MAP."""
    if map_value < 60:
        return 3
    if map_value < 70:
        return 2
    if map_value < 80:
        return 1
    return 0


def compute_news2_score(hr: float, sbp: float, rr: float,
                         temp: float, spo2: float, gcs: int = 15) -> int:
    """Compute NEWS2 score from individual vital parameters."""
    import pandas as pd
    row = pd.Series({
        "heart_rate": hr, "sbp": sbp, "respiratory_rate": rr,
        "temperature": temp, "spo2": spo2, "gcs": gcs,
    })
    return compute_news2(row)


def compute_shock_index(hr: float, sbp: float) -> float:
    """Shock index = HR / SBP."""
    return round(hr / max(sbp, 1), 3)


def add_delta_features(df, columns):
    """Add delta (per-patient diff) columns to a dataframe."""
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[f"{col}_delta"] = result.groupby("patient_id")[col].diff().fillna(0)
    return result


def add_rolling_features(df, columns, windows=None):
    """Add rolling mean/std features."""
    windows = windows or [3, 6]
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        for w in windows:
            result[f"{col}_roll{w}_mean"] = (
                result.groupby("patient_id")[col]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
    return result
