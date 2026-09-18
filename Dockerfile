FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The agent trains its own model on first boot if one is not baked in,
# so a clean `docker compose up` needs no extra step.
CMD ["python", "agent.py"]
