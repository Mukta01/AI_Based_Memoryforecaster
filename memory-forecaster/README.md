# AI-Based Memory Usage Forecaster

An Operating Systems course project that predicts application memory demand using
machine learning and adjusts memory allocation decisions proactively.

The system collects real-time memory telemetry, engineers time-series features,
trains a **Random Forest** and **LSTM** model to forecast future memory usage,
and feeds those forecasts into a decision engine that recommends proactive OS
actions — pre-allocation, early swap-out, or OOM throttling — before memory
pressure becomes critical.

---

## Requirements

- **Python 3.8+**
- Dependencies: `psutil`, `pandas`, `numpy`, `scikit-learn`, `torch`,
  `matplotlib`, `joblib`

## Installation

```bash
pip install psutil pandas numpy scikit-learn torch matplotlib joblib
```

---

## Usage (step-by-step)

### 1. Collect memory data (run for 2+ hours)

```bash
python main.py --mode collect
```

Polls system memory and top-5 processes every 2 seconds.
Press **Ctrl+C** to stop. Output: `data/memory_log.csv`

### 2. Build feature matrix

```bash
python main.py --mode features
```

Engineers lag, rolling-window, rate-of-change, and process-context features.
Output: `data/features.csv`

### 3. Train models

```bash
python main.py --mode train
```

Trains both Random Forest and LSTM. Prints MAE/RMSE comparison.
Outputs:
- `data/rf_results.png`, `data/rf_importances.png`
- `data/lstm_loss.png`, `data/lstm_results.png`
- `model/rf_model.pkl`, `model/lstm_checkpoint.pt`, `model/scaler.pkl`

### 4. Simulate decisions

```bash
python main.py --mode simulate
```

Replays collected data through the decision engine.
Outputs: `data/decisions.csv`, `data/decisions_timeline.png`

### 5. Full evaluation

```bash
python main.py --mode evaluate
```

Trains both models from scratch and produces a side-by-side comparison.
Output: `data/model_comparison.png`

---

## Output Files

| File | Description |
|------|-------------|
| `data/memory_log.csv` | Raw memory samples (2s intervals) |
| `data/features.csv` | Engineered feature matrix for ML |
| `data/rf_results.png` | RF actual vs predicted plot |
| `data/rf_importances.png` | RF top-15 feature importances |
| `data/lstm_loss.png` | LSTM training loss curve |
| `data/lstm_results.png` | LSTM actual vs predicted plot |
| `data/decisions.csv` | Per-row decision log |
| `data/decisions_timeline.png` | Decision timeline with colour markers |
| `data/model_comparison.png` | Both models overlaid on same chart |

---

## AI Model Choices

| Model | Why |
|-------|-----|
| **Random Forest** | Strong baseline; handles tabular features well; provides feature importances for interpretability |
| **LSTM** | Captures temporal/sequential patterns in memory time-series; can learn long-range dependencies |

## Decision Thresholds

| Condition | Action | Meaning |
|-----------|--------|---------|
| Forecast > 95% RAM | `throttle_oom` | Imminent OOM — throttle/kill low-priority processes |
| Forecast > 85% RAM | `swap_early` | High pressure — begin early swap-out |
| Forecast - Current > 200 MB | `prealloc` | Sudden spike — pre-allocate memory |
| Otherwise | `none` | Normal range, no action |

All thresholds are configurable via the `MemoryDecisionEngine` constructor.

---

## Project Structure

```
memory-forecaster/
├── collector.py          # Data collection
├── features.py           # Feature engineering
├── model/
│   ├── __init__.py
│   ├── rf_model.py       # Random Forest model
│   └── lstm_model.py     # LSTM model (PyTorch)
├── decision.py           # Decision engine
├── simulate.py           # Decision replay simulation
├── evaluate.py           # Model comparison
├── main.py               # CLI entry point
├── data/                 # Created at runtime
└── README.md
```

---

## License

This project is for educational purposes (OS course mini-project).
