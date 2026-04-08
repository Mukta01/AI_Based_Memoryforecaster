#!/usr/bin/env python3
"""
rf_model.py — Random Forest Regressor for Memory Forecasting

Trains a Random Forest baseline model on the engineered feature matrix
(data/features.csv) to predict future system memory usage.

Outputs
-------
- Console:  MAE, RMSE, MAPE on the 20 % test set.
- Plots:    data/rf_results.png     (actual vs predicted)
            data/rf_importances.png (top-15 feature importances)
- Model:    model/rf_model.pkl      (serialized via joblib)
"""

import os
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
except ImportError as e:
    print("ERROR: Missing dependency -- {}".format(e))
    print("Install with:  pip install pandas numpy scikit-learn matplotlib joblib")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = Path(__file__).resolve().parent
FEATURES_CSV = DATA_DIR / "features.csv"


def _mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_rf(features_path=None):
    """Train RandomForest on the feature matrix and return results.

    Parameters
    ----------
    features_path : str or Path, optional
        Path to features.csv.  Defaults to data/features.csv.

    Returns
    -------
    tuple
        (model, mae, rmse)
    """
    features_path = Path(features_path) if features_path else FEATURES_CSV
    if not features_path.exists():
        print("ERROR: {} not found.  Run features.py first.".format(features_path))
        sys.exit(1)

    df = pd.read_csv(features_path)
    print("[rf_model] Loaded {} rows, {} columns".format(*df.shape))

    # Separate features and target
    X = df.drop(columns=["y"])
    y = df["y"].values

    # 80/20 split — time-ordered, NO shuffle
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print("[rf_model] Train: {}  |  Test: {}".format(len(X_train), len(X_test)))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = _mape(y_test, y_pred)

    print("\n===== Random Forest Results =====")
    print("  MAE  : {:.2f} MB".format(mae))
    print("  RMSE : {:.2f} MB".format(rmse))
    print("  MAPE : {:.2f} %".format(mape))
    print("=================================\n")

    # ------------------------------------------------------------------
    # Plot — Actual vs Predicted
    # ------------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test, label="Actual", alpha=0.8, linewidth=0.8)
    ax.plot(y_pred, label="Predicted", alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("used_mb")
    ax.set_title("Random Forest — Actual vs Predicted Memory Usage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(DATA_DIR / "rf_results.png"), dpi=150)
    plt.close(fig)
    print("[rf_model] Saved plot -> data/rf_results.png")

    # ------------------------------------------------------------------
    # Plot — Feature Importances (top 15)
    # ------------------------------------------------------------------
    importances = model.feature_importances_
    feat_names = X.columns.tolist()
    indices = np.argsort(importances)[::-1][:15]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(
        [feat_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color="steelblue",
    )
    ax2.set_xlabel("Importance")
    ax2.set_title("Random Forest — Top 15 Feature Importances")
    fig2.tight_layout()
    fig2.savefig(str(DATA_DIR / "rf_importances.png"), dpi=150)
    plt.close(fig2)
    print("[rf_model] Saved plot -> data/rf_importances.png")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = MODEL_DIR / "rf_model.pkl"
    joblib.dump(model, str(model_path))
    print("[rf_model] Saved model -> {}".format(model_path))

    return model, mae, rmse


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_rf()
