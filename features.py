"""Turns a rolling window of raw telemetry into the vector the model reads.

Training and inference both call `featurize`, so the two can never disagree
about column order — the classic cause of a model that scores well offline and
behaves like a coin flip in production.
"""

import numpy as np

POLL_SECONDS = 3  # how often the agent scrapes each service
WINDOW = 20  # 20 samples = 60s of history per prediction
HORIZON = 25  # predict a failure inside the next 25 samples = 75s

SIGNALS = ["cpu", "mem", "latency", "err"]
STATS = ["last", "mean", "std", "slope", "accel"]

FEATURE_NAMES = [f"{s}_{k}" for s in SIGNALS for k in STATS] + ["rps_mean"]


def _slope(values):
    """Change per second, from a least-squares fit over the window."""
    x = np.arange(len(values)) * POLL_SECONDS
    return float(np.polyfit(x, values, 1)[0])


def featurize(window):
    """window: list of metric dicts, oldest first, length == WINDOW."""
    if len(window) != WINDOW:
        raise ValueError(f"expected {WINDOW} samples, got {len(window)}")

    row = []
    for signal in SIGNALS:
        series = np.array([float(sample[signal]) for sample in window])
        half = len(series) // 2
        # Acceleration: is the trend itself getting steeper? A leak that is
        # speeding up matters far more than one drifting at a constant rate.
        accel = _slope(series[half:]) - _slope(series[:half])
        row += [
            float(series[-1]),
            float(series.mean()),
            float(series.std()),
            _slope(series),
            accel,
        ]

    row.append(float(np.mean([float(s["rps"]) for s in window])))
    return row
