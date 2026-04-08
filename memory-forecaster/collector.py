#!/usr/bin/env python3
"""
collector.py — System Memory Data Collector

Polls system memory statistics every 2 seconds using psutil and appends
each sample as a row to data/memory_log.csv.  Captures both system-wide
metrics (used_mb, avail_mb, mem_pct) and the top-5 processes sorted by
RSS memory (pid, name, rss_mb, cpu_percent).

The collector runs indefinitely until interrupted with Ctrl+C, at which
point it prints the total number of rows collected.
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required.  Install with:  pip install psutil")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLL_INTERVAL = 2          # seconds between samples
READOUT_INTERVAL = 10      # seconds between live console readouts
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "memory_log.csv"

# CSV header columns
SYSTEM_COLS = ["timestamp", "used_mb", "avail_mb", "mem_pct"]
PROC_COLS = []
for i in range(1, 6):
    PROC_COLS.extend([
        "pid{}".format(i),
        "name{}".format(i),
        "rss{}".format(i),
        "cpu{}".format(i),
    ])
HEADER = SYSTEM_COLS + PROC_COLS


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _top_processes(n=5):
    """Return the top *n* processes sorted by RSS memory (descending).

    Each entry is a dict with keys: pid, name, rss_mb, cpu_percent.
    Missing or zombie processes are silently skipped.
    """
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
        try:
            info = p.info
            rss_mb = info["memory_info"].rss / (1024 * 1024) if info["memory_info"] else 0.0
            procs.append({
                "pid": info["pid"],
                "name": info["name"] or "unknown",
                "rss_mb": round(rss_mb, 2),
                "cpu_percent": info["cpu_percent"] or 0.0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    procs.sort(key=lambda x: x["rss_mb"], reverse=True)
    return procs[:n]


def collect_sample():
    """Collect a single memory sample and return it as a list of values
    matching the CSV header order.
    """
    vm = psutil.virtual_memory()
    ts = time.time()
    used_mb = round(vm.used / (1024 * 1024), 2)
    avail_mb = round(vm.available / (1024 * 1024), 2)
    mem_pct = vm.percent

    row = [ts, used_mb, avail_mb, mem_pct]

    top5 = _top_processes(5)
    for i in range(5):
        if i < len(top5):
            p = top5[i]
            row.extend([p["pid"], p["name"], p["rss_mb"], p["cpu_percent"]])
        else:
            row.extend([0, "none", 0.0, 0.0])

    return row


# ---------------------------------------------------------------------------
# Main collector loop
# ---------------------------------------------------------------------------

def run_collector():
    """Run the data collector loop until interrupted."""
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Open CSV in append mode; write header if file is new / empty
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(HEADER)
        csv_file.flush()

    rows_collected = 0
    last_readout = 0.0

    print("[collector] Sampling every {}s.  Press Ctrl+C to stop.".format(POLL_INTERVAL))
    print("[collector] Writing to {}".format(CSV_PATH))

    try:
        while True:
            row = collect_sample()
            writer.writerow(row)
            csv_file.flush()
            rows_collected += 1

            now = time.time()
            if now - last_readout >= READOUT_INTERVAL:
                vm = psutil.virtual_memory()
                dt = datetime.now().strftime("%H:%M:%S")
                print(
                    "[{}]  RAM: {:.0f} MB used / {:.0f} MB avail  ({:.1f}%)  |  rows: {}".format(
                        dt,
                        vm.used / (1024 * 1024),
                        vm.available / (1024 * 1024),
                        vm.percent,
                        rows_collected,
                    )
                )
                last_readout = now

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[collector] Stopped.  Total rows collected: {}".format(rows_collected))
    finally:
        csv_file.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_collector()
