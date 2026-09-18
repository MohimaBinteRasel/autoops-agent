"""Builds the dataset, trains the incident predictor, and reports whether it is
actually useful.

Two things here matter more than the model choice:

1. The train/test split is by episode, not by row. Consecutive windows from the
   same degrading service overlap heavily, so a random row split would leak the
   answer across the split and produce a flattering, meaningless score.
2. The headline metric is lead time — how many seconds of warning the agent gets
   before the service returns 503. A model with great ROC-AUC that only fires two
   seconds early is worthless for remediation.
"""

import argparse
import json
import random
import time

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

import simulate
from features import FEATURE_NAMES, HORIZON, POLL_SECONDS, WINDOW, featurize

MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"


def episode(rng, fault, max_ticks=220):
    """One service's life: healthy for a while, then a fault, then maybe death."""
    warmup = rng.randint(WINDOW, WINDOW + 18)
    rate = rng.uniform(0.55, 1.7)
    jitter = simulate.Jitter(rng)
    rows, fail_tick = [], None

    for i in range(max_ticks):
        active = fault if i >= warmup else "none"
        seconds = (i - warmup) * POLL_SECONDS if i >= warmup else 0.0
        sample = jitter.apply(simulate.add_noise(simulate.degrade(active, seconds, rate), rng))
        rows.append(sample)
        if simulate.is_down(sample):
            fail_tick = i
            break

    return rows, fail_tick


def windows(rows, fail_tick):
    """Yield (features, label, end_tick) for every full window in an episode."""
    limit = fail_tick if fail_tick is not None else len(rows)
    for end in range(WINDOW - 1, limit):
        x = featurize(rows[end - WINDOW + 1 : end + 1])
        ahead = None if fail_tick is None else fail_tick - end
        label = 1 if ahead is not None and 0 < ahead <= HORIZON else 0
        yield x, label, end


def build_episodes(n, seed):
    rng = random.Random(seed)
    # Roughly a third of episodes stay healthy, so the model sees plenty of
    # normal behaviour and does not learn "everything eventually dies".
    faults = ["none", "memory_leak", "latency_creep", "error_storm"]
    weights = [3, 2, 2, 2]
    out = []
    for _ in range(n):
        fault = rng.choices(faults, weights=weights)[0]
        rows, fail_tick = episode(rng, fault)
        out.append({"fault": fault, "rows": rows, "fail_tick": fail_tick})
    return out


def to_matrix(episodes):
    X, y = [], []
    for ep in episodes:
        for x, label, _ in windows(ep["rows"], ep["fail_tick"]):
            X.append(x)
            y.append(label)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def pick_threshold(y_true, scores):
    """Highest-F1 threshold. Nothing clever, but it is chosen on held-out data
    rather than hard-coded at 0.5."""
    best = (0.5, -1.0)
    for t in np.arange(0.05, 0.96, 0.01):
        pred = scores >= t
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best[1]:
            best = (float(t), f1)
    return best[0]


def operational_report(clf, episodes, threshold):
    """The metrics an on-call engineer would actually ask about."""
    lead_times, missed, false_alarms, healthy = [], 0, 0, 0

    for ep in episodes:
        alert_tick = None
        for x, _, end in windows(ep["rows"], ep["fail_tick"]):
            if clf.predict_proba([x])[0][1] >= threshold:
                alert_tick = end
                break

        if ep["fail_tick"] is None:
            healthy += 1
            if alert_tick is not None:
                false_alarms += 1
        elif alert_tick is None:
            missed += 1
        else:
            lead_times.append((ep["fail_tick"] - alert_tick) * POLL_SECONDS)

    failing = len([e for e in episodes if e["fail_tick"] is not None])
    return {
        "episodes_tested": len(episodes),
        "failing_episodes": failing,
        "caught_before_failure": len(lead_times),
        "missed_failures": missed,
        "median_lead_time_s": float(np.median(lead_times)) if lead_times else 0.0,
        "min_lead_time_s": float(np.min(lead_times)) if lead_times else 0.0,
        "false_alarm_rate": round(false_alarms / healthy, 3) if healthy else 0.0,
    }


def train(n_train=260, n_test=90, seed=7, quiet=False):
    started = time.time()
    train_eps = build_episodes(n_train, seed)
    test_eps = build_episodes(n_test, seed + 1000)

    X_train, y_train = to_matrix(train_eps)
    X_test, y_test = to_matrix(test_eps)

    clf = GradientBoostingClassifier(
        n_estimators=180, max_depth=3, learning_rate=0.06, subsample=0.9, random_state=seed
    )
    clf.fit(X_train, y_train)

    scores = clf.predict_proba(X_test)[:, 1]
    threshold = pick_threshold(y_test, scores)

    # Unsupervised backstop: flags windows that look unlike anything healthy,
    # which is what catches a failure mode the labelled data never contained.
    iso = IsolationForest(n_estimators=120, contamination=0.04, random_state=seed)
    iso.fit(X_train[y_train == 0])

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "positive_rate": round(float(y_train.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
        "threshold": round(threshold, 2),
        "classification_report": classification_report(
            y_test, scores >= threshold, target_names=["stable", "pre-failure"], output_dict=True
        ),
        "operational": operational_report(clf, test_eps, threshold),
        "top_features": sorted(
            zip(FEATURE_NAMES, clf.feature_importances_.round(4).tolist()),
            key=lambda p: -p[1],
        )[:8],
        "train_seconds": round(time.time() - started, 1),
    }

    joblib.dump(
        {"clf": clf, "iso": iso, "threshold": threshold, "features": FEATURE_NAMES}, MODEL_PATH
    )
    with open(METRICS_PATH, "w") as fh:
        json.dump(report, fh, indent=2)

    if not quiet:
        op = report["operational"]
        print(f"windows      : {report['train_windows']} train / {report['test_windows']} test")
        print(f"ROC-AUC      : {report['roc_auc']}   PR-AUC: {report['pr_auc']}")
        print(f"threshold    : {report['threshold']}")
        print(
            f"caught       : {op['caught_before_failure']}/{op['failing_episodes']} failures, "
            f"{op['missed_failures']} missed"
        )
        print(f"lead time    : {op['median_lead_time_s']}s median, {op['min_lead_time_s']}s worst")
        print(f"false alarms : {op['false_alarm_rate']} of healthy episodes")
        print(f"saved        : {MODEL_PATH}, {METRICS_PATH}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=260)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    train(n_train=args.episodes, quiet=args.quiet)
