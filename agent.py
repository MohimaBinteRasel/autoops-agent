"""AutoOps agent — the control plane.

Loop: scrape every service, turn the last 60s into features, ask the model how
likely a failure is in the next 75s, and restart the service before it happens.
Everything it does is written to SQLite so the dashboard and the post-mortem
both read from the same record.

Run with:  python agent.py
"""

import asyncio
import os
import sqlite3
import time
from collections import deque
from contextlib import asynccontextmanager

import httpx
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from features import POLL_SECONDS, WINDOW, featurize

DB_PATH = os.getenv("DB_PATH", "autoops.db")
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
HEAL_MODE = os.getenv("HEAL_MODE", "http")  # "http" or "docker"
COOLDOWN = float(os.getenv("COOLDOWN", "60"))  # seconds of quiet after a restart
CONFIRM_POLLS = 2  # a single high reading is never enough to act on


def parse_services():
    raw = os.getenv("SERVICES", "payments=http://localhost:8001,orders=http://localhost:8002,inventory=http://localhost:8003")
    out = {}
    for item in raw.split(","):
        name, _, url = item.partition("=")
        if name.strip():
            out[name.strip()] = url.strip().rstrip("/")
    return out


SERVICES = parse_services()

if not os.path.exists(MODEL_PATH):
    print("No model found, training one before starting the control loop...")
    import train_model

    train_model.train(quiet=True)

bundle = joblib.load(MODEL_PATH)
MODEL, ISO, THRESHOLD = bundle["clf"], bundle["iso"], bundle["threshold"]

state = {
    name: {
        "window": deque(maxlen=WINDOW),
        "risk_history": deque(maxlen=60),
        "risk": 0.0,
        "anomaly": 0.0,
        "status": "starting",
        "metrics": {},
        "high_streak": 0,
        "cooldown_until": 0.0,
        "restarts": 0,
        "down_since": None,
    }
    for name in SERVICES
}

stats = {"predicted_saves": 0, "reactive_restarts": 0, "downtime_seconds": 0.0, "polls": 0}
actions = deque(maxlen=40)


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events "
        "(ts REAL, service TEXT, kind TEXT, risk REAL, detail TEXT)"
    )
    return conn


def log_event(service, kind, risk, detail):
    row = {"ts": time.time(), "service": service, "kind": kind, "risk": round(risk, 3), "detail": detail}
    actions.appendleft(row)
    with db() as conn:
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?)",
            (row["ts"], service, kind, row["risk"], detail),
        )


async def heal(client, name, reason, risk):
    """Restart a service. Two backends so the project runs with or without Docker."""
    svc = state[name]
    try:
        if HEAL_MODE == "docker":
            import docker  # imported lazily: not needed in http mode

            docker.from_env().containers.get(f"autoops-{name}").restart()
        else:
            await client.post(f"{SERVICES[name]}/reset", timeout=5)
    except Exception as exc:  # a failed remediation is an incident of its own
        log_event(name, "heal_failed", risk, f"{reason}: {exc}")
        return

    svc["window"].clear()
    svc["high_streak"] = 0
    svc["cooldown_until"] = time.time() + COOLDOWN
    svc["restarts"] += 1
    svc["status"] = "healing"
    log_event(name, reason, risk, f"restarted {name} via {HEAL_MODE}")


async def poll_once(client):
    stats["polls"] += 1
    now = time.time()

    for name, url in SERVICES.items():
        svc = state[name]

        try:
            resp = await client.get(f"{url}/metrics", timeout=2.5)
            metrics = resp.json()
        except Exception:
            # Unreachable is the worst case: we are already past prevention.
            svc["status"] = "unreachable"
            svc["down_since"] = svc["down_since"] or now
            if now >= svc["cooldown_until"]:
                await heal(client, name, "reactive_restart", svc["risk"])
                stats["reactive_restarts"] += 1
            continue

        svc["metrics"] = metrics
        svc["window"].append(metrics)

        if metrics.get("down"):
            stats["downtime_seconds"] += POLL_SECONDS
            svc["status"] = "down"
            svc["down_since"] = svc["down_since"] or now
            if now >= svc["cooldown_until"]:
                await heal(client, name, "reactive_restart", svc["risk"])
                stats["reactive_restarts"] += 1
            continue

        svc["down_since"] = None

        if len(svc["window"]) < WINDOW:
            svc["status"] = "warming up"
            continue

        x = featurize(list(svc["window"]))
        svc["risk"] = float(MODEL.predict_proba([x])[0][1])
        # decision_function is negative for outliers; flip it so higher = stranger
        svc["anomaly"] = round(float(-ISO.decision_function([x])[0]), 3)
        svc["risk_history"].append(round(svc["risk"], 3))

        if svc["risk"] >= THRESHOLD:
            svc["high_streak"] += 1
        else:
            svc["high_streak"] = 0

        if svc["high_streak"] >= CONFIRM_POLLS and now >= svc["cooldown_until"]:
            await heal(client, name, "predictive_restart", svc["risk"])
            stats["predicted_saves"] += 1
        elif now < svc["cooldown_until"]:
            svc["status"] = "recovering"
        elif svc["risk"] >= THRESHOLD:
            svc["status"] = "at risk"
        elif svc["risk"] >= THRESHOLD / 2:
            svc["status"] = "watch"
        else:
            svc["status"] = "healthy"


async def control_loop():
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await poll_once(client)
            except Exception as exc:
                log_event("agent", "loop_error", 0.0, str(exc))
            await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(control_loop())
    yield
    task.cancel()


app = FastAPI(title="AutoOps-Agent", lifespan=lifespan)


@app.get("/")
def dashboard():
    return FileResponse("dashboard.html")


@app.get("/api/state")
def api_state():
    return JSONResponse(
        {
            "threshold": round(THRESHOLD, 2),
            "poll_seconds": POLL_SECONDS,
            "heal_mode": HEAL_MODE,
            "stats": {**stats, "downtime_seconds": round(stats["downtime_seconds"], 1)},
            "services": [
                {
                    "name": name,
                    "status": svc["status"],
                    "risk": round(svc["risk"], 3),
                    "anomaly": svc["anomaly"],
                    "restarts": svc["restarts"],
                    "fault": svc["metrics"].get("fault", "none"),
                    "cpu": round(svc["metrics"].get("cpu", 0), 1),
                    "mem": round(svc["metrics"].get("mem", 0), 1),
                    "latency": round(svc["metrics"].get("latency", 0), 0),
                    "err": round(svc["metrics"].get("err", 0), 2),
                    "history": list(svc["risk_history"]),
                }
                for name, svc in state.items()
            ],
            "actions": list(actions)[:12],
        }
    )


@app.post("/api/chaos/{service}/{fault}")
async def api_chaos(service: str, fault: str):
    """Break a service on purpose, from the dashboard."""
    if service not in SERVICES:
        return JSONResponse({"error": "unknown service"}, status_code=404)
    async with httpx.AsyncClient() as client:
        await client.post(f"{SERVICES[service]}/chaos/{fault}", timeout=5)
    log_event(service, "chaos_injected", 0.0, f"{fault} injected by operator")
    return {"service": service, "fault": fault}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="warning")
