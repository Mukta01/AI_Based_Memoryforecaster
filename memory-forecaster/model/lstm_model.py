#!/usr/bin/env python3
"""
lstm_model.py — LSTM Model for Sequential Memory Forecasting (PyTorch)

Defines a 2-layer LSTM network that ingests a sliding window of 20
timesteps and predicts the memory usage 1 step into the future.

Architecture
-----------
- nn.LSTM(input_size, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
- nn.Linear(64, 1)

Training
--------
- Adam optimiser, lr=1e-3
- MSELoss
- 30 epochs, batch_size=32
- Features normalised with StandardScaler (fit on train only)

Outputs
-------
- Console:   MAE, RMSE on the test set (un-normalised back to MB)
- Plots:     data/lstm_loss.png     (training loss curve)
             data/lstm_results.png  (actual vs predicted)
- Artefacts: model/lstm_checkpoint.pt  (state_dict)
             model/scaler.pkl          (StandardScaler)
"""

import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
except ImportError as e:
    print("ERROR: Missing dependency -- {}".format(e))
    print("Install with:  pip install pandas numpy torch scikit-learn matplotlib joblib")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = Path(__file__).resolve().parent
FEATURES_CSV = DATA_DIR / "features.csv"

WINDOW_SIZE = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3


# ===================================================================
# Dataset
# ===================================================================

class MemoryDataset(Dataset):
    """Sliding-window dataset for LSTM training.

    Parameters
    ----------
    X : np.ndarray, shape (N, num_features)
        Normalised feature matrix.
    y : np.ndarray, shape (N,)
        Target values (raw MB — not normalised).
    window : int
        Number of past timesteps in each sample.
    """

    def __init__(self, X, y, window=WINDOW_SIZE):
        self.X = X
        self.y = y
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.window]
        target = self.y[idx + self.window]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


# ===================================================================
# Model
# ===================================================================

class MemoryLSTM(nn.Module):
    """Two-layer LSTM for memory-usage forecasting.

    Parameters
    ----------
    input_size : int
        Number of features per timestep.
    hidden_size : int
        LSTM hidden dimension.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers.
    """

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(MemoryLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, window, features)

        Returns
        -------
        Tensor, shape (batch, 1)
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last time step
        out = self.fc(out)
        return out.squeeze(-1)


# ===================================================================
# Training routine
# ===================================================================

def train_lstm(features_path=None):
    """Train the LSTM model and return results.

    Parameters
    ----------
    features_path : str or Path, optional

    Returns
    -------
    tuple
        (model, scaler, mae, rmse)
    """
    features_path = Path(features_path) if features_path else FEATURES_CSV
    if not features_path.exists():
        print("ERROR: {} not found.  Run features.py first.".format(features_path))
        sys.exit(1)

    df = pd.read_csv(features_path)
    print("[lstm_model] Loaded {} rows, {} columns".format(*df.shape))

    X = df.drop(columns=["y"]).values.astype(np.float32)
    y = df["y"].values.astype(np.float32).reshape(-1, 1)

    # 80/20 split -- time-ordered, no shuffle
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Normalize target
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw).flatten()
    y_test = y_scaler.transform(y_test_raw).flatten()

    # Datasets and loaders
    train_ds = MemoryDataset(X_train, y_train, WINDOW_SIZE)
    test_ds = MemoryDataset(X_test, y_test, WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[lstm_model] Using device: {}".format(device))

    num_features = X_train.shape[1]
    model = MemoryLSTM(input_size=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    loss_history = []
    print("[lstm_model] Training for {} epochs ...".format(EPOCHS))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print("  Epoch {:3d}/{} -- loss: {:.4f}".format(epoch, EPOCHS, avg_loss))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)

    # Denormalize back to MB
    all_preds_mb = y_scaler.inverse_transform(all_preds).flatten()
    all_targets_mb = y_scaler.inverse_transform(all_targets).flatten()

    mae = mean_absolute_error(all_targets_mb, all_preds_mb)
    rmse = float(np.sqrt(mean_squared_error(all_targets_mb, all_preds_mb)))

    print("\n===== LSTM Results =====")
    print("  MAE  : {:.2f} MB".format(mae))
    print("  RMSE : {:.2f} MB".format(rmse))
    print("========================\n")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Loss curve
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(range(1, EPOCHS + 1), loss_history, marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("LSTM — Training Loss Curve")
    fig1.tight_layout()
    fig1.savefig(str(DATA_DIR / "lstm_loss.png"), dpi=150)
    plt.close(fig1)
    print("[lstm_model] Saved plot -> data/lstm_loss.png")

    # Actual vs Predicted
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(all_targets_mb, label="Actual", alpha=0.8, linewidth=0.8)
    ax2.plot(all_preds_mb, label="Predicted", alpha=0.8, linewidth=0.8)
    ax2.set_xlabel("Test sample index")
    ax2.set_ylabel("used_mb")
    ax2.set_title("LSTM — Actual vs Predicted Memory Usage")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(str(DATA_DIR / "lstm_results.png"), dpi=150)
    plt.close(fig2)
    print("[lstm_model] Saved plot -> data/lstm_results.png")

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    ckpt_path = MODEL_DIR / "lstm_checkpoint.pt"
    torch.save(model.state_dict(), str(ckpt_path))
    print("[lstm_model] Saved checkpoint -> {}".format(ckpt_path))

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump({"feature_scaler": scaler, "target_scaler": y_scaler}, str(scaler_path))
    print("[lstm_model] Saved scaler -> {}".format(scaler_path))

    return model, scaler, mae, rmse


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_lstm()
