#!/usr/bin/env python3
"""
simulate.py — Decision Replay Simulator

Replays the feature matrix (data/features.csv) through the decision
engine row by row and records every forecast + action.

Outputs
-------
- data/decisions.csv           — all per-row decisions
- data/decisions_timeline.png  — visual timeline with colour-coded markers
- Console summary              — count and percentage of each action type
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("ERROR: Missing dependency -- {}".format(e))
    sys.exit(1)

from decision import MemoryDecisionEngine


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURES_CSV = DATA_DIR / "features.csv"
DECISIONS_CSV = DATA_DIR / "decisions.csv"
TIMELINE_PNG = DATA_DIR / "decisions_timeline.png"


def run_simulation(model_type="rf"):
    """Replay features through the decision engine and save results.

    Parameters
    ----------
    model_type : str
        ``'rf'`` or ``'lstm'``.
    """
    if not FEATURES_CSV.exists():
        print("ERROR: {} not found.  Run features.py first.".format(FEATURES_CSV))
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    print("[simulate] Loaded {} rows from features.csv".format(len(df)))

    try:
        engine = MemoryDecisionEngine(model_type=model_type)
    except FileNotFoundError as e:
        print("ERROR: {}".format(e))
        sys.exit(1)

    feature_cols = [c for c in df.columns if c != "y"]
    records = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        feature_row = {c: row[c] for c in feature_cols}
        current_mb = float(row["used_mb"])
        forecast_mb = engine.predict(feature_row)
        decision = engine.decide(forecast_mb, current_mb)

        records.append({
            "timestamp": idx,        # row index as a proxy for time
            "current_mb": current_mb,
            "forecast_mb": decision["forecast_mb"],
            "action": decision["action"],
            "reason": decision["reason"],
        })

        if (idx + 1) % 500 == 0:
            print("  processed {}/{} rows …".format(idx + 1, len(df)))

    results = pd.DataFrame(records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(str(DECISIONS_CSV), index=False)
    print("[simulate] Saved decisions -> {}".format(DECISIONS_CSV))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(results)
    action_counts = results["action"].value_counts()
    print("\n===== Simulation Summary =====")
    for action, count in action_counts.items():
        pct = count / total * 100
        print("  {:15s}  {:>6d}  ({:.1f}%)".format(action, count, pct))
    print("==============================\n")

    # ------------------------------------------------------------------
    # Timeline plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(results["timestamp"], results["current_mb"],
            label="Current MB", alpha=0.7, linewidth=0.7, color="gray")
    ax.plot(results["timestamp"], results["forecast_mb"],
            label="Forecast MB", alpha=0.7, linewidth=0.7, color="steelblue")

    # Overlay decision markers
    colour_map = {
        "prealloc": ("blue", "^", "Pre-alloc"),
        "swap_early": ("orange", "s", "Swap early"),
        "throttle_oom": ("red", "X", "Throttle/OOM"),
    }
    for action_name, (colour, marker, label) in colour_map.items():
        mask = results["action"] == action_name
        if mask.any():
            ax.scatter(
                results.loc[mask, "timestamp"],
                results.loc[mask, "forecast_mb"],
                c=colour, marker=marker, s=18, label=label, zorder=5, alpha=0.8,
            )

    ax.set_xlabel("Sample index (time ->)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Decision Timeline — {} model".format(model_type.upper()))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(TIMELINE_PNG), dpi=150)
    plt.close(fig)
    print("[simulate] Saved timeline plot -> {}".format(TIMELINE_PNG))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replay decisions on features.csv")
    parser.add_argument("--model", default="rf", choices=["rf", "lstm"],
                        help="Model type to use (default: rf)")
    args = parser.parse_args()
    run_simulation(model_type=args.model)
