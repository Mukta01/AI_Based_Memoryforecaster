#!/usr/bin/env python3
"""
features.py — Feature Engineering Pipeline

Loads data/memory_log.csv and produces a machine-learning-ready feature
matrix saved to data/features.csv.

Engineered features
-------------------
- Lag features:       mem_lag1, mem_lag5, mem_lag10
- Rolling (w=30):     mem_roll_mean, mem_roll_std, mem_roll_min, mem_roll_max
- Rate of change:     mem_delta (MB), time_delta (sec), rate_mb_per_sec
- Process context:    rss1 ... rss5 (top-5 process RSS, numeric, fill NaN -> 0)
- Target variable y:  used_mb shifted by -10 (predict 10 steps ahead)

Rows with any NaN (from lag/rolling/target) are dropped.
"""

import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required.  pip install pandas numpy")
    sys.exit(1)


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "memory_log.csv"
FEATURES_CSV = DATA_DIR / "features.csv"


def build_features(input_path=None, output_path=None):
    """Load raw memory log, engineer features, and save the result.

    Parameters
    ----------
    input_path : str or Path, optional
        Path to the raw CSV.  Defaults to data/memory_log.csv.
    output_path : str or Path, optional
        Path to write the feature matrix.  Defaults to data/features.csv.

    Returns
    -------
    pd.DataFrame
        The cleaned feature matrix.
    """
    input_path = Path(input_path) if input_path else RAW_CSV
    output_path = Path(output_path) if output_path else FEATURES_CSV

    # ------------------------------------------------------------------
    # 1.  Load & sort
    # ------------------------------------------------------------------
    if not input_path.exists():
        print("ERROR: {} not found.  Run collector.py first.".format(input_path))
        sys.exit(1)

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("[features] Loaded {} rows from {}".format(len(df), input_path))

    # ------------------------------------------------------------------
    # 2.  Lag features
    # ------------------------------------------------------------------
    df["mem_lag1"] = df["used_mb"].shift(1)
    df["mem_lag5"] = df["used_mb"].shift(5)
    df["mem_lag10"] = df["used_mb"].shift(10)

    # ------------------------------------------------------------------
    # 3.  Rolling window features (window=30)
    # ------------------------------------------------------------------
    roll = df["used_mb"].rolling(window=30)
    df["mem_roll_mean"] = roll.mean()
    df["mem_roll_std"] = roll.std()
    df["mem_roll_min"] = roll.min()
    df["mem_roll_max"] = roll.max()

    # ------------------------------------------------------------------
    # 4.  Rate of change
    # ------------------------------------------------------------------
    df["mem_delta"] = df["used_mb"].diff()
    # Compute time_delta in seconds between consecutive samples
    df["time_delta"] = df["timestamp"].diff().dt.total_seconds()
    df["rate_mb_per_sec"] = df["mem_delta"] / df["time_delta"]

    # ------------------------------------------------------------------
    # 5.  Process context — keep only rss columns (numeric)
    # ------------------------------------------------------------------
    rss_cols = ["rss{}".format(i) for i in range(1, 6)]
    for col in rss_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Drop name/pid/cpu process columns (not needed for ML)
    drop_cols = []
    for i in range(1, 6):
        for prefix in ["pid", "name", "cpu"]:
            col = "{}{}".format(prefix, i)
            if col in df.columns:
                drop_cols.append(col)
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ------------------------------------------------------------------
    # 6.  Target variable: predict used_mb 10 steps ahead
    # ------------------------------------------------------------------
    df["y"] = df["used_mb"].shift(-10)

    # ------------------------------------------------------------------
    # 7.  Drop NaN and non-feature columns
    # ------------------------------------------------------------------
    # Drop the original timestamp (not a numeric feature; index suffices)
    df.drop(columns=["timestamp"], inplace=True, errors="ignore")

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 8.  Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("[features] Feature matrix shape: {}".format(df.shape))
    print("[features] Columns: {}".format(list(df.columns)))
    print("[features] Saved to {}".format(output_path))

    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_features()
