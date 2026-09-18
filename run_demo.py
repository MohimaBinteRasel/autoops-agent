"""Starts the whole system locally, no Docker required.

Three services on 8001-8003, the agent and dashboard on 8000.
Stop everything with Ctrl+C.
"""

import os
import subprocess
import sys
import time
import webbrowser

SERVICES = {"payments": 8001, "orders": 8002, "inventory": 8003}
AGENT_PORT = 8000

procs = []


def spawn(args, env):
    full_env = {**os.environ, **env}
    procs.append(subprocess.Popen(args, env=full_env))


def main():
    if not os.path.exists("model.joblib"):
        print("No model found, training one first (about a minute)...")
        subprocess.run([sys.executable, "train_model.py"], check=True)

    for name, port in SERVICES.items():
        spawn([sys.executable, "service.py"], {"SERVICE_NAME": name, "PORT": str(port)})
        print(f"  {name} on :{port}")

    time.sleep(2.5)

    wiring = ",".join(f"{n}=http://localhost:{p}" for n, p in SERVICES.items())
    spawn([sys.executable, "agent.py"], {"SERVICES": wiring, "HEAL_MODE": "http", "PORT": str(AGENT_PORT)})
    print(f"  agent on :{AGENT_PORT}")

    time.sleep(2)
    url = f"http://localhost:{AGENT_PORT}"
    print(f"\nDashboard: {url}   (Ctrl+C to stop)")
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
