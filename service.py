"""One microservice. Three copies of this file run as payments, orders and
inventory — only the SERVICE_NAME environment variable differs.

It exposes the three things a self-healing platform needs from a workload:
a health check, a metrics endpoint, and a way to be put under stress so the
platform can be tested on purpose instead of waiting for a real outage.
"""

import os
import random
import time

from fastapi import FastAPI, Response

import simulate

NAME = os.getenv("SERVICE_NAME", "service")

app = FastAPI(title=f"autoops-{NAME}")
rng = random.Random(hash(NAME) & 0xFFFF)

state = {"fault": "none", "fault_started": time.time(), "booted": time.time(), "restarts": 0}


def current_metrics():
    elapsed = time.time() - state["fault_started"] if state["fault"] != "none" else 0.0
    raw = simulate.degrade(state["fault"], elapsed)
    return simulate.add_noise(raw, rng)


@app.get("/metrics")
def metrics():
    m = current_metrics()
    m.update(
        service=NAME,
        fault=state["fault"],
        uptime=round(time.time() - state["booted"], 1),
        restarts=state["restarts"],
        down=simulate.is_down(m),
        ts=time.time(),
    )
    return m


@app.get("/health")
def health(response: Response):
    m = current_metrics()
    if simulate.is_down(m):
        response.status_code = 503
        return {"service": NAME, "status": "down", "fault": state["fault"]}
    return {"service": NAME, "status": "ok", "fault": state["fault"]}


@app.post("/chaos/{fault}")
def chaos(fault: str):
    """Inject a fault. Used by the demo script and by anyone poking the API."""
    if fault not in simulate.FAULTS:
        return {"error": f"unknown fault, pick one of {simulate.FAULTS}"}
    state["fault"] = fault
    state["fault_started"] = time.time()
    return {"service": NAME, "fault": fault}


@app.post("/reset")
def reset():
    """Recover. This is what the agent calls when it heals without Docker —
    it stands in for the process restart a container runtime would do."""
    state["fault"] = "none"
    state["fault_started"] = time.time()
    state["booted"] = time.time()
    state["restarts"] += 1
    return {"service": NAME, "status": "restarted", "restarts": state["restarts"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="warning")
