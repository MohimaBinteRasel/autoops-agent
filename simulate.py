"""Degradation model shared by the running services and the dataset generator.

Keeping the maths in one place means the model is trained on the same failure
physics it will see in production. If the two drift apart, the model silently
gets worse, and that is a very hard bug to find later.

Every fault degrades slowly and leaves a trend in the telemetry long before the
service actually returns 503. That gap is the whole point of the project.
"""

import random

BASELINE = {"cpu": 24.0, "mem": 38.0, "latency": 85.0, "err": 0.4, "rps": 120.0}

FAULTS = ["none", "memory_leak", "latency_creep", "error_storm"]

# A service is considered down once any of these is crossed.
LIMITS = {"mem": 97.0, "latency": 1200.0, "err": 45.0}


def degrade(fault, seconds, rate=1.0):
    """Metrics for a service that has been running `fault` for `seconds`.

    `rate` scales how fast the fault develops. No two leaks in the real world
    grow at the same speed, and a model trained on one fixed speed learns a
    stopwatch instead of a trend.
    """
    cpu = BASELINE["cpu"]
    mem = BASELINE["mem"]
    latency = BASELINE["latency"]
    err = BASELINE["err"]
    t = max(0.0, seconds) * rate

    if fault == "memory_leak":
        mem = BASELINE["mem"] + 0.42 * t
        cpu = BASELINE["cpu"] + 0.10 * t
        if mem > 78:  # garbage collection starts thrashing
            over = mem - 78
            latency = BASELINE["latency"] * (1 + 0.10 * over)
            err = BASELINE["err"] + 0.45 * over
    elif fault == "latency_creep":
        latency = BASELINE["latency"] * (1 + 0.075 * t)
        cpu = BASELINE["cpu"] + 0.32 * t
        mem = BASELINE["mem"] + 0.08 * t
        err = BASELINE["err"] + 0.06 * t
    elif fault == "error_storm":
        err = BASELINE["err"] + 0.30 * t
        latency = BASELINE["latency"] * (1 + 0.012 * t)
        cpu = BASELINE["cpu"] + 0.18 * t

    return {
        "cpu": min(cpu, 99.0),
        "mem": min(mem, 100.0),
        "latency": min(latency, 5000.0),
        "err": min(err, 100.0),
        "rps": BASELINE["rps"],
    }


def add_noise(metrics, rng=random):
    """Real telemetry is never smooth. Noise is what stops the model cheating."""
    noisy = dict(metrics)
    noisy["cpu"] = max(0.0, metrics["cpu"] + rng.gauss(0, 2.4))
    noisy["mem"] = max(0.0, metrics["mem"] + rng.gauss(0, 1.1))
    noisy["latency"] = max(1.0, metrics["latency"] * (1 + rng.gauss(0, 0.07)))
    noisy["err"] = max(0.0, metrics["err"] + rng.gauss(0, 0.35))
    noisy["rps"] = max(1.0, metrics["rps"] + rng.gauss(0, 14))
    return noisy


def is_down(metrics):
    return (
        metrics["mem"] >= LIMITS["mem"]
        or metrics["latency"] >= LIMITS["latency"]
        or metrics["err"] >= LIMITS["err"]
    )


class Jitter:
    """Occasional harmless spikes: a slow query, a noisy neighbour, a cold cache.

    These are the reason a naive threshold alert is useless. The agent has to
    learn the difference between a service having a bad second and a service on
    its way down, otherwise it restarts healthy workloads all day.
    """

    def __init__(self, rng, chance=0.035):
        self.rng = rng
        self.chance = chance
        self.ticks_left = 0
        self.size = 1.0

    def apply(self, metrics):
        if self.ticks_left <= 0 and self.rng.random() < self.chance:
            self.ticks_left = self.rng.randint(2, 4)
            self.size = self.rng.uniform(2.0, 4.5)

        if self.ticks_left <= 0:
            return metrics

        self.ticks_left -= 1
        spiked = dict(metrics)
        # Clamped just under the failure limits: a spike is allowed to look
        # alarming, never to actually take the service down.
        spiked["latency"] = max(
            metrics["latency"], min(metrics["latency"] * self.size, LIMITS["latency"] - 1)
        )
        spiked["cpu"] = min(99.0, metrics["cpu"] * self.rng.uniform(1.3, 1.9))
        spiked["err"] = max(
            metrics["err"], min(metrics["err"] + self.rng.uniform(0.5, 3.5), LIMITS["err"] - 1)
        )
        return spiked
