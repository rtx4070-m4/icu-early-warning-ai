"""
AI Hospital Operating System
Data Engineering – Preprocessing
===================================
Handles missing data imputation, outlier treatment,
normalization, and time-series preparation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

logger = logging.getLogger("preprocessing")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
VITAL_COLS = [
    "heart_rate", "sbp", "dbp", "map",
    "temperature", "spo2", "resp_rate", "gcs_total",
]

LAB_COLS = [
    "Creatinine", "WBC", "Hemoglobin", "Platelets",
    "Sodium", "Potassium", "Glucose", "BUN", "Lactate", "Troponin",
]

CATEGORICAL_COLS = ["icu_unit", "gender", "admission_type"]


# ─────────────────────────────────────────────
# Missing Value Handlers
# ─────────────────────────────────────────────
def impute_vitals_locf(df: pd.DataFrame, group_col: str = "patient_id") -> pd.DataFrame:
    """
    Last Observation Carried Forward (LOCF) imputation for vitals.
    Within each patient's time series, fill forward then backward.
    """
    df = df.copy().sort_values([group_col, "charttime"])
    cols = [c for c in VITAL_COLS if c in df.columns]

    df[cols] = (
        df.groupby(group_col)[cols]
        .transform(lambda s: s.ffill().bfill())
    )
    remaining_na = df[cols].isna().sum().sum()
    if remaining_na > 0:
        # Fall back to population median for any remaining NaN
        for col in cols:
            df[col].fillna(df[col].median(), inplace=True)

    logger.info("LOCF imputation complete; residual NaNs after median fill: %d",
                df[cols].isna().sum().sum())
    return df


def impute_labs_knn(df_labs: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    KNN imputation for lab values pivoted to wide format.
    Returns wide-format dataframe with one row per patient per time window.
    """
    if df_labs.empty:
        return df_labs

    # Pivot to wide format: patient_id × lab_name
    pivot = df_labs.pivot_table(
        index="patient_id",
        columns="lab_name",
        values="value",
        aggfunc="mean",
    )

    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed = imputer.fit_transform(pivot)
    result = pd.DataFrame(imputed, index=pivot.index, columns=pivot.columns)
    result.reset_index(inplace=True)
    logger.info("KNN lab imputation complete: %s", result.shape)
    return result


def fill_missing_with_normal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with clinically normal values
    (last resort when LOCF and KNN cannot be applied).
    """
    normal_values = {
        "heart_rate":  75.0,
        "sbp":        120.0,
        "dbp":         75.0,
        "map":         90.0,
        "temperature": 37.0,
        "spo2":        98.0,
        "resp_rate":   16.0,
        "gcs_total":   15.0,
    }
    for col, val in normal_values.items():
        if col in df.columns:
            df[col].fillna(val, inplace=True)
    return df


# ─────────────────────────────────────────────
# Outlier Treatment
# ─────────────────────────────────────────────
def winsorize_vitals(df: pd.DataFrame, quantile: float = 0.01) -> pd.DataFrame:
    """
    Clip extreme values to [q, 1-q] percentiles per vital.
    Preserves clinical outliers while removing data entry errors.
    """
    df = df.copy()
    cols = [c for c in VITAL_COLS if c in df.columns]
    for col in cols:
        lo = df[col].quantile(quantile)
        hi = df[col].quantile(1 - quantile)
        df[col] = df[col].clip(lo, hi)
    return df


def detect_outliers_iqr(
    df: pd.DataFrame, cols: Optional[List[str]] = None, multiplier: float = 3.0
) -> pd.Series:
    """Return boolean mask of rows containing outliers (IQR method)."""
    if cols is None:
        cols = [c for c in VITAL_COLS if c in df.columns]
    outlier_mask = pd.Series(False, index=df.index)
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outlier_mask |= (df[col] < lower) | (df[col] > upper)
    return outlier_mask


# ─────────────────────────────────────────────
# Normalization / Scaling
# ─────────────────────────────────────────────
class VitalsNormalizer:
    """
    Fit scalers on training data; apply to new streams.
    Uses RobustScaler (median/IQR) to handle clinical outliers.
    """

    def __init__(self, method: str = "robust"):
        self.method  = method
        self.scalers: Dict[str, object] = {}
        self.fitted  = False

    def _make_scaler(self):
        if self.method == "robust":
            return RobustScaler()
        elif self.method == "standard":
            return StandardScaler()
        elif self.method == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown scaler method: {self.method}")

    def _ensure_df(self, X):
        """Accept both DataFrames and numpy arrays."""
        if isinstance(X, pd.DataFrame):
            return X, True   # is_df
        arr = np.asarray(X, dtype=float)
        n = arr.shape[1] if arr.ndim == 2 else 1
        col_names = (list(VITAL_COLS) + [f"feat_{i}" for i in range(n)])[:n]
        return pd.DataFrame(arr, columns=col_names), False   # is_df

    def fit(self, df) -> "VitalsNormalizer":
        df_frame, is_df = self._ensure_df(df)
        cols = [c for c in VITAL_COLS if c in df_frame.columns]
        if not cols:
            # pure-numpy path: fit one shared scaler on all columns
            scaler = self._make_scaler()
            scaler.fit(df_frame.values)
            self.scalers["_all"] = scaler
            self._numpy_mode = True
            self.fitted = True
            return self
        for col in cols:
            scaler = self._make_scaler()
            scaler.fit(df_frame[[col]].dropna())
            self.scalers[col] = scaler
        self._numpy_mode = False
        self.fitted = True
        logger.info("VitalsNormalizer fitted on %d columns (%s)", len(cols), self.method)
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        df_frame, is_df = self._ensure_df(df)
        if getattr(self, "_numpy_mode", False):
            # Return numpy array of same shape as input
            return self.scalers["_all"].transform(df_frame.values)
        # Column-by-column scaling
        out = df_frame.copy()
        result_cols = []
        for col in df_frame.columns:
            if col in self.scalers:
                scaled = self.scalers[col].transform(df_frame[[col]].fillna(df_frame[col].median())).flatten()
                result_cols.append(scaled)
            else:
                result_cols.append(df_frame[col].values)
        import numpy as _np
        result_arr = _np.column_stack(result_cols)
        # Return DataFrame if input was DataFrame, else numpy array (same shape)
        if is_df:
            return pd.DataFrame(result_arr, columns=df_frame.columns, index=df_frame.index)
        return result_arr

    def fit_transform(self, df):
        return self.fit(df).transform(df)



# ─────────────────────────────────────────────
# Time-Series Windowing
# ─────────────────────────────────────────────
def create_sliding_windows(
    df: pd.DataFrame,
    window_hours: int = 12,
    step_hours: int = 1,
    group_col: str = "patient_id",
    time_col: str = "charttime",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed-length sliding windows over patient time series.

    Returns
    -------
    X : np.ndarray  shape (n_windows, window_size, n_features)
    meta : np.ndarray  patient_id and window end-time for each window
    """
    df = df.copy().sort_values([group_col, time_col])
    feature_cols = [c for c in VITAL_COLS if c in df.columns]

    X_list, meta_list = [], []

    for pid, group in df.groupby(group_col):
        group = group.set_index(time_col)[feature_cols]

        # Resample to 1-hour frequency
        group = group.resample("1H").mean()

        values = group.values
        window_size = window_hours
        step = step_hours

        for start in range(0, len(values) - window_size + 1, step):
            window = values[start : start + window_size]
            if window.shape[0] == window_size:
                X_list.append(window)
                end_time = group.index[start + window_size - 1]
                meta_list.append((pid, end_time))

    if not X_list:
        logger.warning("No windows created – check input data length vs window size")
        return np.array([]), np.array([])

    X = np.stack(X_list, axis=0)
    meta = np.array(meta_list, dtype=object)
    logger.info("Created %d windows of shape %s", X.shape[0], X.shape[1:])
    return X, meta


def create_sequence_dataset(
    df: pd.DataFrame,
    label_col: str = "label_deteriorating",
    sequence_length: int = 12,
    group_col: str = "patient_id",
    time_col: str = "charttime",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labeled sequence dataset for LSTM training.

    Returns
    -------
    X : shape (n_samples, sequence_length, n_features)
    y : shape (n_samples,)
    """
    df = df.copy().sort_values([group_col, time_col])
    feature_cols = [c for c in VITAL_COLS if c in df.columns]

    X_list, y_list = [], []

    for _, group in df.groupby(group_col):
        vals   = group[feature_cols].values
        labels = group[label_col].values if label_col in group.columns else None

        for i in range(len(vals) - sequence_length):
            seq = vals[i : i + sequence_length]
            if seq.shape[0] == sequence_length:
                X_list.append(seq)
                if labels is not None:
                    y_list.append(labels[i + sequence_length])
                else:
                    y_list.append(0)

    if not X_list:
        return np.array([]), np.array([])

    return np.stack(X_list), np.array(y_list)


# ─────────────────────────────────────────────
# One-hot encode categoricals
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns present in the dataframe."""
    cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not cols:
        return df
    dummies = pd.get_dummies(df[cols], drop_first=False, prefix=cols)
    df = pd.concat([df.drop(columns=cols), dummies], axis=1)
    logger.info("Encoded %d categorical columns → %d dummy columns",
                len(cols), len(dummies.columns))
    return df


# ─────────────────────────────────────────────
# Full preprocessing pipeline
# ─────────────────────────────────────────────
class PreprocessingPipeline:
    """
    End-to-end preprocessing: impute → winsorize → encode → normalize → window.
    """

    def __init__(self, window_hours: int = 12, sequence_length: int = 12):
        self.window_hours    = window_hours
        self.sequence_length = sequence_length
        self.normalizer      = VitalsNormalizer(method="robust")

    def fit_transform_vitals(
        self, df: pd.DataFrame, label_col: Optional[str] = None
    ) -> Dict:
        """
        Full preprocessing for vital signs.
        Returns dict with processed DataFrame, windows, and sequences.
        """
        logger.info("Preprocessing pipeline starting on %d rows", len(df))

        # 1. Impute missing values
        df = impute_vitals_locf(df)

        # 2. Winsorize outliers
        df = winsorize_vitals(df, quantile=0.005)

        # 3. Encode categoricals
        df = encode_categoricals(df)

        # 4. Normalize
        df = self.normalizer.fit_transform(df)

        # 5. Windows for anomaly detection
        X_windows, meta = create_sliding_windows(
            df, window_hours=self.window_hours
        )

        # 6. Sequences for LSTM (if label available)
        X_seq, y_seq = (
            create_sequence_dataset(df, label_col=label_col,
                                    sequence_length=self.sequence_length)
            if label_col and label_col in df.columns
            else (np.array([]), np.array([]))
        )

        return {
            "df":        df,
            "X_windows": X_windows,
            "meta":      meta,
            "X_seq":     X_seq,
            "y_seq":     y_seq,
        }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from load_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=0)
    vitals = gen.generate_vitals(n_patients=10, hours=48)

    pipeline = PreprocessingPipeline(window_hours=6, sequence_length=6)
    result   = pipeline.fit_transform_vitals(vitals, label_col="label_deteriorating")

    print(f"Processed DataFrame:  {result['df'].shape}")
    print(f"Sliding windows:       {result['X_windows'].shape}")
    print(f"LSTM sequences:        {result['X_seq'].shape}")
    print(f"Labels:                {result['y_seq'].shape}")
