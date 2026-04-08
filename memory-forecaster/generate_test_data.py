#!/usr/bin/env python3
"""
generate_test_data.py — Generate synthetic memory_log.csv for testing.

Creates ~4000 rows of realistic-looking memory data so the full pipeline
(features → train → simulate → evaluate) can be verified without
running the collector for 2 hours.
"""

import csv
import os
import time
import random
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "memory_log.csv"

HEADER = ["timestamp", "used_mb", "avail_mb", "mem_pct"]
for i in range(1, 6):
    HEADER.extend(["pid{}".format(i), "name{}".format(i),
                    "rss{}".format(i), "cpu{}".format(i)])

TOTAL_RAM = 16384.0  # simulate 16 GB
PROCESS_NAMES = [
    "chrome", "code", "python3", "node", "postgres",
    "firefox", "java", "docker", "slack", "spotify",
]

random.seed(42)


def generate():
    """Generate synthetic memory log data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    num_rows = 4000
    base_time = time.time() - num_rows * 2  # start 4000*2s ago

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        used_mb = 6000.0 + random.uniform(-200, 200)

        for i in range(num_rows):
            ts = base_time + i * 2.0

            # Simulate realistic memory patterns: slow drift + periodic +
            # occasional spikes
            drift = 0.05 * math.sin(2 * math.pi * i / 1800)  # ~1hr cycle
            noise = random.gauss(0, 15)
            spike = 0.0
            if random.random() < 0.02:
                spike = random.uniform(100, 500)  # occasional spike
            elif random.random() < 0.01:
                spike = random.uniform(-300, -100)   # occasional drop

            used_mb = max(2000, min(TOTAL_RAM - 500,
                                     used_mb + drift * 50 + noise + spike))
            avail_mb = TOTAL_RAM - used_mb
            mem_pct = round(used_mb / TOTAL_RAM * 100, 1)

            row = [round(ts, 2), round(used_mb, 2), round(avail_mb, 2), mem_pct]

            # Generate 5 fake processes
            for j in range(5):
                pid = 1000 + j * 111 + random.randint(0, 10)
                name = random.choice(PROCESS_NAMES)
                rss = round(random.uniform(50, 800) * (1.0 / (j + 1)), 2)
                cpu = round(random.uniform(0, 30), 1)
                row.extend([pid, name, rss, cpu])

            writer.writerow(row)

    print("[generate_test_data] Created {} rows -> {}".format(num_rows, CSV_PATH))


if __name__ == "__main__":
    generate()
