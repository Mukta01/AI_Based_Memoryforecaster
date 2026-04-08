#!/usr/bin/env python3
"""
decision.py — Memory Allocation Decision Engine

Loads a trained forecasting model (Random Forest or LSTM) and makes
proactive memory-management recommendations based on predicted usage.

Decision thresholds (configurable)
-----------------------------------
- forecast > 95 % total RAM  ->  **throttle_oom**  (imminent OOM)
- forecast > 85 % total RAM  ->  **swap_early**    (high pressure)
- forecast - current > 200 MB ->  **prealloc**      (sudden spike)
- else                        ->  **none**          (no action)

Usage
-----
    from decision import MemoryDecisionEngine
    engine = MemoryDecisionEngine(model_type='rf')
    forecast = engine.predict(feature_row_dict)
    action   = engine.decide(forecast, current_mb)
"""

import sys
from pathlib import Path

import numpy as np

try:
    import psutil
    import joblib
    import pandas as pd
except ImportError as e:
    print("ERROR: Missing dependency -- {}".format(e))
    sys.exit(1)

# Optional — LSTM needs torch
try:
    import torch
except ImportError:
    torch = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
FEATURES_CSV = DATA_DIR / "features.csv"


class MemoryDecisionEngine:
    """Proactive memory allocation decision engine.

    Parameters
    ----------
    model_type : str
        ``'rf'`` for Random Forest, ``'lstm'`` for LSTM.
    total_ram_mb : float, optional
        Total system RAM in MB.  Auto-detected via psutil if not given.
    oom_pct : float
        Forecast percentage threshold for OOM throttle (default 95).
    swap_pct : float
        Forecast percentage threshold for early swap (default 85).
    prealloc_delta_mb : float
        Minimum MB jump to trigger pre-allocation (default 200).
    """

    def __init__(self, model_type="rf", total_ram_mb=None,
                 oom_pct=95.0, swap_pct=85.0, prealloc_delta_mb=200.0):
        self.model_type = model_type.lower()
        self.total_ram_mb = total_ram_mb or (psutil.virtual_memory().total / (1024 * 1024))
        self.oom_pct = oom_pct
        self.swap_pct = swap_pct
        self.prealloc_delta_mb = prealloc_delta_mb

        self.model = None
        self.scaler = None
        self.feature_columns = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the saved model from disk."""
        if self.model_type == "rf":
            pkl_path = MODEL_DIR / "rf_model.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(
                    "RF model not found at {}.  Run rf_model.py first.".format(pkl_path)
                )
            self.model = joblib.load(str(pkl_path))

        elif self.model_type == "lstm":
            if torch is None:
                raise ImportError("PyTorch is required for the LSTM model.")

            ckpt_path = MODEL_DIR / "lstm_checkpoint.pt"
            scaler_path = MODEL_DIR / "scaler.pkl"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    "LSTM checkpoint not found at {}.  Run lstm_model.py first.".format(ckpt_path)
                )
            if not scaler_path.exists():
                raise FileNotFoundError(
                    "Scaler not found at {}.  Run lstm_model.py first.".format(scaler_path)
                )

            scaler_data = joblib.load(str(scaler_path))
            # Support both old (single scaler) and new (dict) format
            if isinstance(scaler_data, dict):
                self.scaler = scaler_data["feature_scaler"]
                self.target_scaler = scaler_data.get("target_scaler", None)
            else:
                self.scaler = scaler_data
                self.target_scaler = None

            # Infer input_size from scaler
            from model.lstm_model import MemoryLSTM, WINDOW_SIZE
            input_size = self.scaler.n_features_in_
            self.model = MemoryLSTM(input_size=input_size)
            self.model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
            self.model.eval()

        else:
            raise ValueError("model_type must be 'rf' or 'lstm', got '{}'".format(self.model_type))

        # Load feature columns for RF from features.csv header
        if FEATURES_CSV.exists():
            header = pd.read_csv(FEATURES_CSV, nrows=0).columns.tolist()
            self.feature_columns = [c for c in header if c != "y"]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, feature_row):
        """Return forecasted memory (MB) from a single feature row.

        Parameters
        ----------
        feature_row : dict or pd.Series
            Feature values keyed by column name.

        Returns
        -------
        float
            Predicted used_mb for the future timestep.
        """
        if self.model_type == "rf":
            if self.feature_columns:
                vals = [feature_row.get(c, 0.0) for c in self.feature_columns]
            else:
                vals = list(feature_row.values())
            arr = np.array(vals, dtype=np.float64).reshape(1, -1)
            return float(self.model.predict(arr)[0])

        elif self.model_type == "lstm":
            # For single-row prediction the caller should pass a window of
            # rows.  As a fallback, duplicate the single row across the window.
            from model.lstm_model import WINDOW_SIZE
            if self.feature_columns:
                vals = [feature_row.get(c, 0.0) for c in self.feature_columns]
            else:
                vals = list(feature_row.values())
            arr = np.array(vals, dtype=np.float32).reshape(1, -1)
            arr_scaled = self.scaler.transform(arr)
            # Tile to fill window
            window = np.tile(arr_scaled, (WINDOW_SIZE, 1))
            tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(tensor).item()
            # Denormalize if target scaler is available
            if self.target_scaler is not None:
                pred = self.target_scaler.inverse_transform(
                    np.array([[pred]])
                ).flatten()[0]
            return float(pred)

        return 0.0

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def decide(self, forecast_mb, current_mb):
        """Map a forecast to a proactive memory-management action.

        Parameters
        ----------
        forecast_mb : float
            Predicted future memory usage (MB).
        current_mb : float
            Current memory usage (MB).

        Returns
        -------
        dict
            Keys: action, reason, forecast_mb, current_mb, threshold_pct.
        """
        forecast_pct = (forecast_mb / self.total_ram_mb) * 100.0

        if forecast_pct > self.oom_pct:
            action = "throttle_oom"
            reason = (
                "Forecast {:.0f} MB ({:.1f}%) exceeds {:.0f}% of total RAM -- "
                "recommend throttling / killing low-priority processes"
            ).format(forecast_mb, forecast_pct, self.oom_pct)

        elif forecast_pct > self.swap_pct:
            action = "swap_early"
            reason = (
                "Forecast {:.0f} MB ({:.1f}%) exceeds {:.0f}% of total RAM -- "
                "recommend early swap-out of cold pages"
            ).format(forecast_mb, forecast_pct, self.swap_pct)

        elif (forecast_mb - current_mb) > self.prealloc_delta_mb:
            action = "prealloc"
            reason = (
                "Predicted spike of {:.0f} MB (current {:.0f} -> forecast {:.0f}) "
                "exceeds {:.0f} MB delta -- recommend pre-allocation"
            ).format(
                forecast_mb - current_mb, current_mb, forecast_mb,
                self.prealloc_delta_mb,
            )

        else:
            action = "none"
            reason = "Memory usage within normal range ({:.1f}% forecast)".format(forecast_pct)

        return {
            "action": action,
            "reason": reason,
            "forecast_mb": round(forecast_mb, 2),
            "current_mb": round(current_mb, 2),
            "threshold_pct": round(forecast_pct, 2),
        }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demo — load RF model and predict on the first feature row

    if not FEATURES_CSV.exists():
        print("ERROR: data/features.csv not found.  Run features.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    row = df.drop(columns=["y"]).iloc[0].to_dict()
    current_mb = df.iloc[0]["used_mb"]

    try:
        engine = MemoryDecisionEngine(model_type="rf")
    except FileNotFoundError as e:
        print("ERROR: {}".format(e))
        sys.exit(1)

    forecast = engine.predict(row)
    result = engine.decide(forecast, current_mb)

    print("\n===== Decision Engine Demo =====")
    for k, v in result.items():
        print("  {}: {}".format(k, v))
    print("================================\n")
