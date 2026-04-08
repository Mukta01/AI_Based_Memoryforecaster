#!/usr/bin/env python3
"""
main.py — CLI Entry Point for the AI-Based Memory Usage Forecaster

Usage:
    python main.py --mode collect
    python main.py --mode features
    python main.py --mode train
    python main.py --mode simulate
    python main.py --mode evaluate
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def mode_collect():
    """Run the data collector."""
    from collector import run_collector
    run_collector()


def mode_features():
    """Run feature engineering."""
    from features import build_features
    build_features()


def mode_train():
    """Train both RF and LSTM models."""
    from model.rf_model import train_rf
    from model.lstm_model import train_lstm

    print("=" * 60)
    print("  Training Random Forest ...")
    print("=" * 60)
    _, rf_mae, rf_rmse = train_rf()

    print("=" * 60)
    print("  Training LSTM ...")
    print("=" * 60)
    _, _, lstm_mae, lstm_rmse = train_lstm()

    print("\n" + "=" * 50)
    print("  {:15s} {:>10s} {:>10s}".format("Model", "MAE (MB)", "RMSE (MB)"))
    print("  " + "-" * 45)
    print("  {:15s} {:>10.2f} {:>10.2f}".format("Random Forest", rf_mae, rf_rmse))
    print("  {:15s} {:>10.2f} {:>10.2f}".format("LSTM", lstm_mae, lstm_rmse))
    print("=" * 50 + "\n")


def mode_simulate():
    """Run the decision simulation."""
    from simulate import run_simulation
    run_simulation(model_type="rf")


def mode_evaluate():
    """Run the full evaluation."""
    from evaluate import run_evaluation
    run_evaluation()


def main():
    """Parse CLI arguments and dispatch."""
    parser = argparse.ArgumentParser(
        description="AI-Based Memory Usage Forecaster",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Modes:\n"
            "  collect   - Poll system memory every 2s\n"
            "  features  - Engineer ML features\n"
            "  train     - Train RF and LSTM models\n"
            "  simulate  - Replay data through decision engine\n"
            "  evaluate  - Compare models, generate report\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "features", "train", "simulate", "evaluate"],
        help="Operation mode",
    )
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "collect": mode_collect,
        "features": mode_features,
        "train": mode_train,
        "simulate": mode_simulate,
        "evaluate": mode_evaluate,
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
