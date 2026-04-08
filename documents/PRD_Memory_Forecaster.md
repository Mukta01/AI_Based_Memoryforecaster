# Product Requirements Document (PRD)
# AI-Based Memory Usage Forecaster

**Course:** Operating Systems  
**Version:** 1.0  
**Date:** April 2026  
**Status:** Final  

---

## 1. Executive Summary

The **AI-Based Memory Usage Forecaster** is an OS-course project that demonstrates how
machine learning can be integrated with operating system memory management. The system
collects real-time memory telemetry from a Linux host, engineers time-series features,
trains both a Random Forest and LSTM model to forecast future memory demand, and feeds
those forecasts into a decision engine that recommends proactive allocation actions
(pre-allocation, early swap-out, OOM throttling).

The project bridges two disciplines — **machine learning** and **operating systems** —
by showing that predictive analytics can improve memory management beyond traditional
reactive policies.

---

## 2. Problem Statement

Traditional OS memory managers are **reactive**: they detect pressure only after it
occurs (page faults, OOM kills, swap storms). This project explores a **proactive**
approach:

| Reactive (Traditional)           | Proactive (This Project)                    |
|----------------------------------|---------------------------------------------|
| Wait for page fault → allocate   | Predict demand → pre-allocate ahead of time |
| Swap only when RAM full          | Forecast spike → swap early to avoid stall  |
| OOM-kill after exhaustion        | Forecast exhaustion → throttle before crash  |

---

## 3. Project Objectives

1. **Collect** real-time memory statistics (system-wide + per-process) at 2-second granularity.
2. **Engineer** time-series features (lags, rolling windows, rates of change) suitable for regression.
3. **Train** two forecasting models:
   - **Random Forest** (baseline, interpretable)
   - **LSTM** (deep learning, captures temporal patterns)
4. **Build** a decision engine that maps forecasted memory to actionable OS decisions.
5. **Simulate** the decision engine on historical data and evaluate decision quality.
6. **Compare** both models quantitatively (MAE, RMSE, MAPE) and qualitatively.

---

## 4. Target Environment

| Attribute        | Value                                           |
|------------------|-------------------------------------------------|
| OS               | Linux (Ubuntu 20.04+) or any POSIX system       |
| Python           | 3.8+                                            |
| Hardware         | Any system with ≥ 4 GB RAM                      |
| Data Collection  | Recommended ≥ 2 hours (≥ 3,600 samples at 2s)  |
| External Network | Not required — fully offline                    |

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                        │
│   --mode collect | features | train | simulate | evaluate   │
└────┬──────────┬───────────┬───────────┬───────────┬─────────┘
     │          │           │           │           │
     ▼          ▼           ▼           ▼           ▼
 collector   features   rf_model    simulate    evaluate
    .py        .py      lstm_model    .py         .py
     │          │         .py          │           │
     ▼          ▼           │          ▼           │
  data/       data/         ▼       decision      │
  memory_     features    model/      .py         │
  log.csv     .csv        *.pkl       │           │
                          *.pt        ▼           ▼
                                   data/        data/
                                   decisions    model_
                                   .csv         comparison
                                   _timeline    .png
                                   .png
```

---

## 6. Module Specifications

### 6.1 collector.py — Data Collection

**Purpose:** Continuously poll system memory statistics and per-process memory usage.

| Requirement | Detail |
|-------------|--------|
| Polling interval | 2 seconds |
| System metrics | `used_mb`, `avail_mb`, `mem_pct` from `psutil.virtual_memory()` |
| Process metrics | Top 5 processes by RSS: `pid`, `name`, `rss_mb`, `cpu_percent` |
| Output | `data/memory_log.csv` — one row per sample |
| Live readout | Print current RAM usage every 10 seconds |
| Termination | Ctrl+C → print total rows collected |

**CSV Schema (26 columns):**
```
timestamp, used_mb, avail_mb, mem_pct,
pid1, name1, rss1, cpu1,
pid2, name2, rss2, cpu2,
pid3, name3, rss3, cpu3,
pid4, name4, rss4, cpu4,
pid5, name5, rss5, cpu5
```

### 6.2 features.py — Feature Engineering

**Purpose:** Transform raw memory logs into an ML-ready feature matrix.

| Feature Group | Columns Generated |
|---------------|-------------------|
| Lag features | `mem_lag1`, `mem_lag5`, `mem_lag10` |
| Rolling window (w=30) | `mem_roll_mean`, `mem_roll_std`, `mem_roll_min`, `mem_roll_max` |
| Rate of change | `mem_delta`, `time_delta`, `rate_mb_per_sec` |
| Process context | `rss1` … `rss5` (numeric, fill NaN with 0) |
| Target variable | `y` = `used_mb` shifted by -10 (predict 10 steps = 20 seconds ahead) |

**Output:** `data/features.csv` — rows with NaN dropped.

### 6.3 model/rf_model.py — Random Forest Model

| Parameter | Value |
|-----------|-------|
| Algorithm | `RandomForestRegressor` |
| n_estimators | 100 |
| random_state | 42 |
| Train/Test split | 80 / 20 (time-ordered, no shuffle) |
| Metrics | MAE, RMSE, MAPE |
| Artifacts | `data/rf_results.png`, `data/rf_importances.png`, `model/rf_model.pkl` |

### 6.4 model/lstm_model.py — LSTM Model

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer LSTM, hidden=64, dropout=0.2 |
| Window size | 20 timesteps |
| Normalization | StandardScaler (fit on train only) |
| Optimizer | Adam, lr=1e-3 |
| Loss | MSELoss |
| Epochs | 30 |
| Batch size | 32 |
| Metrics | MAE, RMSE |
| Artifacts | `data/lstm_loss.png`, `data/lstm_results.png`, `model/lstm_checkpoint.pt`, `model/scaler.pkl` |

### 6.5 decision.py — Decision Engine

**Class:** `MemoryDecisionEngine`

| Threshold | Action | Meaning |
|-----------|--------|---------|
| forecast > 95% total RAM | `throttle_oom` | Imminent OOM — recommend throttle/kill |
| forecast > 85% total RAM | `swap_early` | High pressure — recommend early swap |
| forecast − current > 200 MB | `prealloc` | Sudden spike — recommend pre-allocation |
| Otherwise | `none` | No action needed |

All thresholds are configurable via constructor kwargs.

### 6.6 simulate.py — Decision Replay

**Purpose:** Replay the collected data through the decision engine and record outcomes.

| Output | Description |
|--------|-------------|
| `data/decisions.csv` | Timestamp, current_mb, forecast_mb, action, reason |
| `data/decisions_timeline.png` | Line chart with color-coded decision markers |
| Console summary | Count and percentage of each action type |

### 6.7 evaluate.py — Model Comparison

**Purpose:** Train both models and produce a comparative evaluation.

| Output | Description |
|--------|-------------|
| Comparison table | Model, MAE (MB), RMSE (MB), MAPE (%) |
| `data/model_comparison.png` | Both models' predictions overlaid |
| Console conclusion | Which model won, by how much |

### 6.8 main.py — CLI Entry Point

```
python main.py --mode {collect, features, train, simulate, evaluate}
```

Each mode invokes the corresponding module. If `--mode` is omitted, prints help.

---

## 7. Data Flow

```
  [System]          [collector.py]         [features.py]
  psutil  ──2s──▶  memory_log.csv  ──▶  features.csv
                                              │
                      ┌───────────────────────┤
                      ▼                       ▼
                 rf_model.py            lstm_model.py
                      │                       │
                      ▼                       ▼
                 rf_model.pkl           lstm_checkpoint.pt
                      │                  scaler.pkl
                      └───────┬───────────┘
                              ▼
                         decision.py
                              │
                              ▼
                        simulate.py
                              │
                              ▼
                  decisions.csv + timeline plot
                              │
                              ▼
                        evaluate.py
                              │
                              ▼
                   model_comparison.png
```

---

## 8. Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| psutil | System memory/process data | ≥ 5.9 |
| pandas | Data manipulation | ≥ 1.3 |
| numpy | Numerical computation | ≥ 1.21 |
| scikit-learn | Random Forest, metrics | ≥ 1.0 |
| torch | LSTM neural network | ≥ 1.9 |
| matplotlib | Plotting | ≥ 3.4 |
| joblib | Model serialization | ≥ 1.1 |

**Install:**
```bash
pip install psutil pandas numpy scikit-learn torch matplotlib joblib
```

---

## 9. Success Criteria

| Criterion | Target |
|-----------|--------|
| Data collection | ≥ 3,000 rows (≈ 1.5 hours at 2s interval) |
| Feature matrix | ≥ 15 engineered features |
| RF MAE | < 50 MB on test set |
| LSTM MAE | < 40 MB on test set (ideally better than RF) |
| Decision accuracy | Correct action triggered in ≥ 80% of pressure events |
| All scripts | Run end-to-end without error on fresh Ubuntu |

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Insufficient data variation | Medium | Run collector during diverse workloads |
| LSTM overfitting | Medium | Dropout, early stopping, sliding window |
| PyTorch install issues | Low | Provide CPU-only install command |
| Memory-intensive training | Low | Models are small; < 1 GB RAM needed |

---

## 11. Timeline

| Day | Phase | Deliverables |
|-----|-------|-------------|
| 1 | Setup & Collection | `collector.py`, `memory_log.csv` |
| 2 | Feature Engineering | `features.py`, `features.csv` |
| 3 | Baseline Model | `rf_model.py`, RF plots & metrics |
| 4 | LSTM Model | `lstm_model.py`, LSTM plots & metrics |
| 5 | Decision Engine | `decision.py`, `simulate.py` |
| 6 | Evaluation & Report | `evaluate.py`, comparison plots |
| 7 | Polish & Buffer | `main.py`, `README.md`, final review |

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| RSS | Resident Set Size — physical memory used by a process |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| OOM | Out of Memory |
| Lag feature | Value of a variable at a previous timestep |
| Rolling window | Moving average/statistic over a fixed number of recent samples |

---

*End of PRD*
