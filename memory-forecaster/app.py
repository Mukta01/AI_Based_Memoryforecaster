#!/usr/bin/env python3
"""
app.py — Flask Web Dashboard for AI-Based Memory Usage Forecaster

Serves an interactive dashboard that visualises every stage of the
memory-forecasting pipeline: data collection, feature engineering,
model training, decision simulation, and evaluation.

Usage:
    python app.py
    → Open http://localhost:5000 in your browser
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from flask import Flask, jsonify, render_template, send_from_directory
except ImportError:
    print("ERROR: Flask is required.  Install with:  pip install flask")
    sys.exit(1)

import numpy as np

try:
    import pandas as pd
    import psutil
except ImportError as e:
    print("ERROR: Missing dependency -- {}".format(e))
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
MEMORY_LOG = DATA_DIR / "memory_log.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
DECISIONS_CSV = DATA_DIR / "decisions.csv"

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

# Store training metrics in memory so the dashboard can display them
_metrics_cache = {
    "rf": None,   # {"mae": ..., "rmse": ..., "mape": ...}
    "lstm": None,
}
_running_lock = threading.Lock()
_running_task = None  # name of the currently running task, or None


# ===================================================================
# Routes — Pages
# ===================================================================

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


# ===================================================================
# Routes — API
# ===================================================================

@app.route("/api/live")
def api_live():
    """Return current system memory snapshot."""
    vm = psutil.virtual_memory()
    return jsonify({
        "used_mb": round(vm.used / (1024 * 1024), 2),
        "avail_mb": round(vm.available / (1024 * 1024), 2),
        "total_mb": round(vm.total / (1024 * 1024), 2),
        "mem_pct": vm.percent,
    })


@app.route("/api/status")
def api_status():
    """Return which pipeline stages have completed (by checking for files)."""
    result = {
        "collect": MEMORY_LOG.exists(),
        "features": FEATURES_CSV.exists(),
        "train": (MODEL_DIR / "rf_model.pkl").exists(),
        "simulate": DECISIONS_CSV.exists(),
        "evaluate": (DATA_DIR / "model_comparison.png").exists(),
        "running": _running_task,
    }

    # Row counts
    if MEMORY_LOG.exists():
        try:
            df = pd.read_csv(MEMORY_LOG)
            result["collect_rows"] = len(df)
        except Exception:
            result["collect_rows"] = 0

    if FEATURES_CSV.exists():
        try:
            df = pd.read_csv(FEATURES_CSV, nrows=0)
            cols = df.columns.tolist()
            result["feature_cols"] = len(cols)
            # Count total rows without loading all data
            with open(FEATURES_CSV, "r") as f:
                result["feature_rows"] = sum(1 for _ in f) - 1
        except Exception:
            pass

    if DECISIONS_CSV.exists():
        try:
            with open(DECISIONS_CSV, "r") as f:
                result["decision_count"] = sum(1 for _ in f) - 1
        except Exception:
            result["decision_count"] = 0

    # Cached metrics
    if _metrics_cache["rf"]:
        result["rf_metrics"] = _metrics_cache["rf"]
    if _metrics_cache["lstm"]:
        result["lstm_metrics"] = _metrics_cache["lstm"]

    return jsonify(result)


# ===================================================================
# Routes — Data Previews
# ===================================================================

@app.route("/api/data/memory_log")
def api_data_memory_log():
    """Return a preview of the raw memory log."""
    if not MEMORY_LOG.exists():
        return jsonify({"rows": [], "columns": []})

    try:
        df = pd.read_csv(MEMORY_LOG)
        columns = df.columns.tolist()
        # Send first 100 rows
        preview = df.head(100)
        rows = json.loads(preview.to_json(orient="records"))
        return jsonify({"rows": rows, "columns": columns, "total_rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/features")
def api_data_features():
    """Return feature matrix info and preview."""
    if not FEATURES_CSV.exists():
        return jsonify({"rows": [], "columns": []})

    try:
        df = pd.read_csv(FEATURES_CSV)
        columns = df.columns.tolist()
        total_rows = len(df)
        preview = df.head(50)
        rows = json.loads(preview.to_json(orient="records"))

        # Calculate dropped rows
        dropped = 0
        if MEMORY_LOG.exists():
            try:
                raw = pd.read_csv(MEMORY_LOG)
                dropped = len(raw) - total_rows
            except Exception:
                pass

        return jsonify({
            "rows": rows,
            "columns": columns,
            "total_rows": total_rows,
            "dropped": dropped,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/decisions")
def api_data_decisions():
    """Return decision summary."""
    if not DECISIONS_CSV.exists():
        return jsonify({"summary": {}, "total": 0})

    try:
        df = pd.read_csv(DECISIONS_CSV)
        summary = df["action"].value_counts().to_dict()
        return jsonify({"summary": summary, "total": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================================================
# Routes — Serve Plots
# ===================================================================

@app.route("/api/plots/<filename>")
def api_plots(filename):
    """Serve a generated plot PNG from the data directory."""
    if not (DATA_DIR / filename).exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(str(DATA_DIR), filename)


# ===================================================================
# Routes — Pipeline Execution
# ===================================================================

def _set_running(task):
    global _running_task
    _running_task = task


def _clear_running():
    global _running_task
    _running_task = None


@app.route("/api/collect", methods=["POST"])
def api_collect():
    """Generate synthetic test data (for demo purposes)."""
    global _running_task
    with _running_lock:
        if _running_task:
            return jsonify({"success": False, "error": "Already running: " + _running_task})

    try:
        _set_running("collect")
        from generate_test_data import generate
        generate()
        _clear_running()
        return jsonify({"success": True, "message": "Data generated successfully"})
    except Exception as e:
        _clear_running()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/features", methods=["POST"])
def api_features():
    """Run feature engineering."""
    global _running_task
    with _running_lock:
        if _running_task:
            return jsonify({"success": False, "error": "Already running: " + _running_task})

    try:
        _set_running("features")
        from features import build_features
        build_features()
        _clear_running()
        return jsonify({"success": True, "message": "Features built successfully"})
    except Exception as e:
        _clear_running()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """Train both RF and LSTM models."""
    global _running_task
    with _running_lock:
        if _running_task:
            return jsonify({"success": False, "error": "Already running: " + _running_task})

    try:
        _set_running("train")

        # Train RF
        from model.rf_model import train_rf
        rf_model, rf_mae, rf_rmse = train_rf()

        # Compute RF MAPE
        df = pd.read_csv(FEATURES_CSV)
        X = df.drop(columns=["y"])
        y = df["y"].values
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y[split_idx:]
        rf_preds = rf_model.predict(X_test)
        mask = y_test != 0
        rf_mape = float(np.mean(np.abs((y_test[mask] - rf_preds[mask]) / y_test[mask])) * 100)

        _metrics_cache["rf"] = {
            "mae": round(rf_mae, 2),
            "rmse": round(rf_rmse, 2),
            "mape": round(rf_mape, 2),
        }

        # Train LSTM
        from model.lstm_model import train_lstm
        lstm_model, lstm_scaler, lstm_mae, lstm_rmse = train_lstm()

        # Compute LSTM MAPE (approximate — use the same approach as evaluate.py)
        import joblib
        import torch
        from model.lstm_model import MemoryDataset, WINDOW_SIZE
        from torch.utils.data import DataLoader

        scaler_data = joblib.load(str(MODEL_DIR / "scaler.pkl"))
        if isinstance(scaler_data, dict):
            feat_scaler = scaler_data["feature_scaler"]
            tgt_scaler = scaler_data.get("target_scaler", None)
        else:
            feat_scaler = scaler_data
            tgt_scaler = None

        X_test_scaled = feat_scaler.transform(X_test.values.astype(np.float32))
        if tgt_scaler is not None:
            y_test_norm = tgt_scaler.transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_test_norm = y_test.copy()

        lstm_ds = MemoryDataset(X_test_scaled, y_test_norm.astype(np.float32), WINDOW_SIZE)
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

        if tgt_scaler is not None:
            lstm_preds_mb = tgt_scaler.inverse_transform(lstm_preds_raw).flatten()
            lstm_targets_mb = tgt_scaler.inverse_transform(lstm_targets_raw).flatten()
        else:
            lstm_preds_mb = lstm_preds_raw.flatten()
            lstm_targets_mb = lstm_targets_raw.flatten()

        mask_l = lstm_targets_mb != 0
        lstm_mape = float(np.mean(np.abs(
            (lstm_targets_mb[mask_l] - lstm_preds_mb[mask_l]) / lstm_targets_mb[mask_l]
        )) * 100)

        _metrics_cache["lstm"] = {
            "mae": round(lstm_mae, 2),
            "rmse": round(lstm_rmse, 2),
            "mape": round(lstm_mape, 2),
        }

        _clear_running()
        return jsonify({
            "success": True,
            "rf": _metrics_cache["rf"],
            "lstm": _metrics_cache["lstm"],
        })
    except Exception as e:
        _clear_running()
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Run decision simulation."""
    global _running_task
    with _running_lock:
        if _running_task:
            return jsonify({"success": False, "error": "Already running: " + _running_task})

    try:
        _set_running("simulate")
        from simulate import run_simulation
        run_simulation(model_type="rf")
        _clear_running()
        return jsonify({"success": True, "message": "Simulation complete"})
    except Exception as e:
        _clear_running()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Run full evaluation."""
    global _running_task
    with _running_lock:
        if _running_task:
            return jsonify({"success": False, "error": "Already running: " + _running_task})

    try:
        _set_running("evaluate")
        from evaluate import run_evaluation
        run_evaluation()

        # Re-read metrics from the training cache or re-compute
        _clear_running()
        return jsonify({"success": True, "message": "Evaluation complete"})
    except Exception as e:
        _clear_running()
        return jsonify({"success": False, "error": str(e)}), 500


# ===================================================================
# Startup — Load Metrics from Saved Models
# ===================================================================

def _load_cached_metrics():
    """Compute metrics from existing saved models so the dashboard
    shows numbers immediately without re-training."""
    rf_pkl = MODEL_DIR / "rf_model.pkl"
    scaler_pkl = MODEL_DIR / "scaler.pkl"
    lstm_ckpt = MODEL_DIR / "lstm_checkpoint.pt"

    if not FEATURES_CSV.exists():
        return

    try:
        import joblib
        df = pd.read_csv(FEATURES_CSV)
        X = df.drop(columns=["y"])
        y = df["y"].values
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y[split_idx:]
    except Exception:
        return

    # --- RF metrics ---
    if rf_pkl.exists() and _metrics_cache["rf"] is None:
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            rf_model = joblib.load(str(rf_pkl))
            rf_preds = rf_model.predict(X_test)
            rf_mae = mean_absolute_error(y_test, rf_preds)
            rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
            mask = y_test != 0
            rf_mape = float(np.mean(np.abs(
                (y_test[mask] - rf_preds[mask]) / y_test[mask]
            )) * 100)
            _metrics_cache["rf"] = {
                "mae": round(rf_mae, 2),
                "rmse": round(rf_rmse, 2),
                "mape": round(rf_mape, 2),
            }
            print("[startup] Loaded RF metrics: MAE={:.2f}, RMSE={:.2f}, MAPE={:.2f}%".format(
                rf_mae, rf_rmse, rf_mape))
        except Exception as e:
            print("[startup] Could not load RF metrics: {}".format(e))

    # --- LSTM metrics ---
    if lstm_ckpt.exists() and scaler_pkl.exists() and _metrics_cache["lstm"] is None:
        try:
            import torch
            from model.lstm_model import MemoryLSTM, MemoryDataset, WINDOW_SIZE
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            from torch.utils.data import DataLoader

            scaler_data = joblib.load(str(scaler_pkl))
            if isinstance(scaler_data, dict):
                feat_scaler = scaler_data["feature_scaler"]
                tgt_scaler = scaler_data.get("target_scaler", None)
            else:
                feat_scaler = scaler_data
                tgt_scaler = None

            input_size = feat_scaler.n_features_in_
            model = MemoryLSTM(input_size=input_size)
            model.load_state_dict(torch.load(str(lstm_ckpt), map_location="cpu"))
            model.eval()

            X_test_scaled = feat_scaler.transform(X_test.values.astype(np.float32))
            if tgt_scaler is not None:
                y_test_norm = tgt_scaler.transform(y_test.reshape(-1, 1)).flatten()
            else:
                y_test_norm = y_test.copy()

            lstm_ds = MemoryDataset(X_test_scaled, y_test_norm.astype(np.float32), WINDOW_SIZE)
            lstm_loader = DataLoader(lstm_ds, batch_size=64, shuffle=False)

            all_preds, all_targets = [], []
            with torch.no_grad():
                for xb, yb in lstm_loader:
                    preds = model(xb).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(yb.numpy())

            preds_raw = np.array(all_preds).reshape(-1, 1)
            targets_raw = np.array(all_targets).reshape(-1, 1)

            if tgt_scaler is not None:
                preds_mb = tgt_scaler.inverse_transform(preds_raw).flatten()
                targets_mb = tgt_scaler.inverse_transform(targets_raw).flatten()
            else:
                preds_mb = preds_raw.flatten()
                targets_mb = targets_raw.flatten()

            lstm_mae = mean_absolute_error(targets_mb, preds_mb)
            lstm_rmse = float(np.sqrt(mean_squared_error(targets_mb, preds_mb)))
            mask_l = targets_mb != 0
            lstm_mape = float(np.mean(np.abs(
                (targets_mb[mask_l] - preds_mb[mask_l]) / targets_mb[mask_l]
            )) * 100)

            _metrics_cache["lstm"] = {
                "mae": round(lstm_mae, 2),
                "rmse": round(lstm_rmse, 2),
                "mape": round(lstm_mape, 2),
            }
            print("[startup] Loaded LSTM metrics: MAE={:.2f}, RMSE={:.2f}, MAPE={:.2f}%".format(
                lstm_mae, lstm_rmse, lstm_mape))
        except Exception as e:
            print("[startup] Could not load LSTM metrics: {}".format(e))


# ===================================================================
# Entry Point
# ===================================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AI Memory Forecaster — Web Dashboard")
    print("  Open: http://localhost:5000")
    print("=" * 55 + "\n")
    _load_cached_metrics()
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
