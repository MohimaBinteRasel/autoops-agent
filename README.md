<img width="1920" height="1080" alt="predictive-restart-demo" src="https://github.com/user-attachments/assets/11a4a241-33d7-41ca-8d03-4896e5ce5dde" />
# AutoOps-Agent
![tests](https://github.com/MohimaBinteRasel/autoops-agent/actions/workflows/ci.yml/badge.svg)

**Self-healing microservices and a predictive cloud incident resolver.**

Most monitoring tells you a service is down. By then you are already paying for
the outage. AutoOps-Agent watches the telemetry of a small fleet, learns what the
sixty seconds before a failure look like, and restarts the service while it is
still serving traffic.

On the held-out test set it catches **59 of 59 failures a median of 75 seconds
before they happen**, with no false alarms on healthy services.

---

## What it actually does

```
  payments ─┐
  orders   ─┼──▶  collector  ──▶  60s rolling window  ──▶  feature builder
  inventory─┘      (every 3s)                                   │
                                                                ▼
        docker restart  ◀──  remediation  ◀──  risk score (GradientBoosting)
        or POST /reset        + cooldown         + novelty score (IsolationForest)
                                   │
                                   ▼
                        SQLite event log ──▶ dashboard
```

Three FastAPI services report CPU, memory, p50 latency, error rate and RPS. Each
one can be pushed into a fault on purpose — a memory leak, latency creep, or an
error storm — so the platform can be tested deliberately instead of waiting for
a real incident.

The agent scrapes all three every three seconds, turns the last twenty samples
into a feature vector, and asks the model for the probability that the service
will cross a failure threshold within the next 75 seconds. Two consecutive
readings above the decision threshold trigger a restart, followed by a 60-second
cooldown so a recovering service is not restarted again while its window refills.

If prediction fails and a service goes down anyway, the agent still restarts it —
just reactively, and the event log records which of the two happened. That
distinction is the whole scoreboard for this project.

## Results

Trained on 260 simulated service lifetimes, tested on 90 unseen ones.

| | |
|---|---|
| Windows (train / test) | 28,568 / 9,577 |
| ROC-AUC | 0.998 |
| PR-AUC | 0.988 |
| Precision / recall on pre-failure windows | 0.95 / 0.94 |
| Failures caught before they happened | 59 / 59 |
| Median lead time | 75s (worst case 54s) |
| False alarms on healthy services | 0 of 31 episodes |

Two choices behind those numbers are worth more than the numbers themselves:

**The split is by episode, not by row.** Consecutive windows from the same
degrading service overlap by 95%, so a random row split would put near-duplicates
on both sides and produce a flattering, meaningless score. Grouping by episode
keeps the test set honest.

**Lead time is the headline metric, not AUC.** A classifier with excellent AUC
that only fires four seconds before the crash cannot be acted on. Fifty seconds
is enough for a restart to complete and the service to warm up again, so that is
what the evaluation measures.

The top features the model leans on are mean CPU over the window, the latest
error rate, and the variance and level of memory — which matches how these faults
actually present. `metrics.json` has the full report, regenerated on every training run.

**On the environment:** the fleet is simulated, not production traffic. The
degradation physics live in `simulate.py` and are shared by the services and the
dataset generator, so the model is trained on exactly the behaviour it will see
at runtime. The scores above describe the model on that simulator. Pointed at
real telemetry, the pipeline is the same but the numbers would not be.

## Design decisions

- **Noise and harmless spikes are built into the simulator.** Real services have
  bad seconds — a slow query, a cold cache, a noisy neighbour. `Jitter` injects
  spikes that look alarming but never cause a failure. Without them, a plain
  threshold alert would solve the problem and the model would be pointless.
- **Fault speed varies per episode.** Two memory leaks never grow at the same
  rate, and a model trained on one fixed speed learns a stopwatch instead of a trend.
- **Acceleration is a feature, not just slope.** A leak that is speeding up is a
  different situation from one drifting steadily, and it is the difference that
  buys the extra warning.
- **An IsolationForest runs alongside the classifier.** The supervised model can
  only recognise the three failure modes in the training data. The unsupervised
  score flags windows that simply look unlike anything healthy, which is the
  backstop for a failure mode nobody labelled.
- **Two consecutive readings before acting.** Restarting a healthy service costs
  real availability, so a single high reading is never enough.

## Running it

Needs Python 3.10+.

```bash
pip install -r requirements.txt
python train_model.py     # ~1 minute, writes model.joblib and metrics.json
python run_demo.py        # starts everything, opens the dashboard
```

The dashboard is at http://localhost:8000. Each service row has buttons to inject
a fault; watch the risk trace climb and the agent step in before the service dies.

With Docker, remediation becomes a genuine container restart instead of an
in-process reset:

```bash
docker compose up --build
```

Tests:

```bash
pytest -q
```

## Files

| | |
|---|---|
| `simulate.py` | Degradation physics, shared by the services and the dataset generator |
| `features.py` | Rolling-window feature engineering, shared by training and inference |
| `service.py` | One microservice; three copies run with different names |
| `train_model.py` | Dataset generation, training, and the operational evaluation |
| `agent.py` | Collector, predictor, healer and API |
| `dashboard.html` | Operator view — risk traces, telemetry, action log, chaos buttons |
| `run_demo.py` | Starts the whole system locally without Docker |

## What I would do next

- Replace the simulator with a Prometheus scrape against real workloads and
  retrain on recorded incidents.
- Swap the blanket restart for a remediation policy per fault type: a memory leak
  wants a restart, latency creep usually wants a scale-out, an error storm often
  wants a circuit breaker on the dependency underneath.
- Track model drift in production — if lead time degrades week over week, the
  failure modes have changed and the model needs new data.
