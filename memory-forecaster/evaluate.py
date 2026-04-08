#!/usr/bin/env python3
"""
evaluate.py — Model Comparison & Final Evaluation

Trains both the Random Forest and LSTM models, collects their metrics,
and produces a side-by-side comparison.

Outputs
-------
- Console:  comparison table (MAE, RMSE, MAPE)
- Plot:     data/model_comparison.png   (both predictions overlaid)
- Console:  conclusion — which model won and by how much
"""

import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as e:
    print("ERROR: Missing dependency -- {}".format(e))
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURES_CSV = DATA_DIR / "features.csv"


def _mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run_evaluation():
    """Train both models, compare, and produce a visual report."""

    if not FEATURES_CSV.exists():
        print("ERROR: {} not found.  Run features.py first.".format(FEATURES_CSV))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Import model trainers
    # ------------------------------------------------------------------
    sys.path.insert(0, str(BASE_DIR))
    from model.rf_model import train_rf
    from model.lstm_model import train_lstm

    # ------------------------------------------------------------------
    # Train RF
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Training Random Forest ...")
    print("=" * 60)
    rf_model, rf_mae, rf_rmse = train_rf()

    # ------------------------------------------------------------------
    # Train LSTM
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Training LSTM ...")
    print("=" * 60)
    lstm_model, lstm_scaler, lstm_mae, lstm_rmse = train_lstm()

    # ------------------------------------------------------------------
    # Compute MAPE for both on the same test split
    # ------------------------------------------------------------------
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["y"])
    y = df["y"].values

    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y[split_idx:]

    # RF predictions
    rf_preds = rf_model.predict(X_test)
    rf_mape = _mape(y_test, rf_preds)

    # LSTM predictions -- need to load scalers from saved file
    import torch
    import joblib
    from model.lstm_model import MemoryDataset, WINDOW_SIZE
    from sklearn.preprocessing import StandardScaler

    scaler_data = joblib.load(str(BASE_DIR / "model" / "scaler.pkl"))
    if isinstance(scaler_data, dict):
        feat_scaler = scaler_data["feature_scaler"]
        tgt_scaler = scaler_data.get("target_scaler", None)
    else:
        feat_scaler = scaler_data
        tgt_scaler = None

    X_test_scaled = feat_scaler.transform(X_test.values.astype(np.float32))

    # Normalize targets the same way as training
    if tgt_scaler is not None:
        y_test_norm = tgt_scaler.transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_test_norm = y_test.copy()

    lstm_ds = MemoryDataset(X_test_scaled, y_test_norm, WINDOW_SIZE)
    from torch.utils.data import DataLoader
    lstm_loader = DataLoader(lstm_ds, batch_size=64, shuffle=False)

    lstm_model.eval()
    device = next(lstm_model.parameters()).device
    lstm_all_preds = []
    lstm_all_targets = []
    with torch.no_grad():
        for xb, yb in lstm_loader:
            xb = xb.to(device)
            preds = lstm_model(xb).cpu().numpy()
            lstm_all_preds.extend(preds)
            lstm_all_targets.extend(yb.numpy())

    lstm_preds_raw = np.array(lstm_all_preds).reshape(-1, 1)
    lstm_targets_raw = np.array(lstm_all_targets).reshape(-1, 1)

    # Denormalize back to MB
    if tgt_scaler is not None:
        lstm_preds = tgt_scaler.inverse_transform(lstm_preds_raw).flatten()
        lstm_targets = tgt_scaler.inverse_transform(lstm_targets_raw).flatten()
    else:
        lstm_preds = lstm_preds_raw.flatten()
        lstm_targets = lstm_targets_raw.flatten()

    lstm_mape = _mape(lstm_targets, lstm_preds)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  {:20s} {:>10s} {:>10s} {:>10s}".format("Model", "MAE (MB)", "RMSE (MB)", "MAPE (%)"))
    print("  " + "-" * 50)
    print("  {:20s} {:>10.2f} {:>10.2f} {:>10.2f}".format(
        "Random Forest", rf_mae, rf_rmse, rf_mape))
    print("  {:20s} {:>10.2f} {:>10.2f} {:>10.2f}".format(
        "LSTM", lstm_mae, lstm_rmse, lstm_mape))
    print("=" * 55)

    # ------------------------------------------------------------------
    # Conclusion
    # ------------------------------------------------------------------
    if rf_mae < lstm_mae:
        winner = "Random Forest"
        diff = lstm_mae - rf_mae
    else:
        winner = "LSTM"
        diff = rf_mae - lstm_mae

    print("\n  >> {} wins by {:.2f} MB MAE\n".format(winner, diff))

    # ------------------------------------------------------------------
    # Overlay plot
    # ------------------------------------------------------------------
    # Align lengths: RF uses all test rows, LSTM is shorter by WINDOW_SIZE
    rf_plot_len = min(len(rf_preds), len(y_test))
    lstm_plot_len = len(lstm_preds)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(rf_plot_len), y_test[:rf_plot_len],
            label="Actual", alpha=0.7, linewidth=0.8, color="black")
    ax.plot(range(rf_plot_len), rf_preds[:rf_plot_len],
            label="RF Predicted", alpha=0.7, linewidth=0.8, color="steelblue")
    ax.plot(range(WINDOW_SIZE, WINDOW_SIZE + lstm_plot_len), lstm_preds,
            label="LSTM Predicted", alpha=0.7, linewidth=0.8, color="coral")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("used_mb")
    ax.set_title("Model Comparison -- Actual vs Predicted Memory Usage")
    ax.legend()
    fig.tight_layout()
    comparison_path = DATA_DIR / "model_comparison.png"
    fig.savefig(str(comparison_path), dpi=150)
    plt.close(fig)
    print("[evaluate] Saved comparison plot -> {}".format(comparison_path))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
