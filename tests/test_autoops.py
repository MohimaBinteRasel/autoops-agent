"""Tests for the parts that would fail silently if they broke."""

import random

import numpy as np
import pytest

import simulate
from features import FEATURE_NAMES, POLL_SECONDS, WINDOW, featurize


def sample(fault="none", seconds=0.0):
    return simulate.degrade(fault, seconds)


def test_healthy_service_stays_inside_limits():
    for t in range(0, 600, 10):
        assert not simulate.is_down(sample("none", t))


@pytest.mark.parametrize("fault", ["memory_leak", "latency_creep", "error_storm"])
def test_every_fault_eventually_takes_the_service_down(fault):
    assert simulate.is_down(sample(fault, 400))


def test_feature_vector_matches_its_names():
    window = [simulate.add_noise(sample("memory_leak", i * POLL_SECONDS)) for i in range(WINDOW)]
    assert len(featurize(window)) == len(FEATURE_NAMES)


def test_featurize_rejects_a_short_window():
    with pytest.raises(ValueError):
        featurize([sample()] * (WINDOW - 1))


def test_slope_is_positive_while_memory_leaks():
    window = [sample("memory_leak", i * POLL_SECONDS) for i in range(WINDOW)]
    row = dict(zip(FEATURE_NAMES, featurize(window)))
    assert row["mem_slope"] > 0


def test_jitter_never_causes_a_failure_on_its_own():
    rng = random.Random(3)
    jitter = simulate.Jitter(rng, chance=1.0)
    for _ in range(300):
        spiked = jitter.apply(simulate.add_noise(sample("none"), rng))
        assert not simulate.is_down(spiked)


def test_model_gives_a_low_risk_to_a_healthy_window():
    joblib = pytest.importorskip("joblib")
    import os

    if not os.path.exists("model.joblib"):
        pytest.skip("model not trained yet; run python train_model.py")

    bundle = joblib.load("model.joblib")
    rng = random.Random(11)
    window = [simulate.add_noise(sample("none"), rng) for _ in range(WINDOW)]
    risk = bundle["clf"].predict_proba([featurize(window)])[0][1]
    assert risk < bundle["threshold"]
