# Deployment Guide

Operational reference for running the Dynamic Regime Risk Management Platform in production.

---

## Running Locally

```bash
# Development (single worker, auto-reload)
uvicorn main_v4:app --reload --port 8000

# Production-like (4 workers, no reload)
uvicorn main_v4:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Note on multiple workers:** Each uvicorn worker process has its own `SessionRegistry`. Live streaming sessions are not shared across workers. For multi-worker deployments with streaming, use a single worker or add Redis-backed session storage (see Extending below).

---

## Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main_v2:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t regime-platform .
docker run -p 8000:8000 regime-platform
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Listening port |
| `WORKERS` | `1` | Uvicorn worker count |
| `LOG_LEVEL` | `info` | Uvicorn log level: `debug` \| `info` \| `warning` \| `error` |

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Tick latency (p50) | ~7 ms | HMM predict + Kelly + VaR/CVaR |
| HMM refit latency | 200–500 ms | Background, non-blocking |
| Max concurrent sessions | ~500 | Per worker, memory-bound |
| Price buffer memory | ~32 KB | Per session at window=500 |
| SSE fan-out overhead | ~0.1 ms/subscriber | Queue put_nowait |

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "2.0.0"}
```

---

## Extending

### Redis-backed session persistence

Replace the in-memory `deque` in `StreamSession` with a Redis list:

```python
import redis.asyncio as redis

r = redis.from_url("redis://localhost:6379")

async def _append_price(self, price: float):
    key = f"prices:{self.symbol}"
    await r.rpush(key, price)
    await r.ltrim(key, -self.window, -1)  # keep last N

async def _load_prices(self) -> list[float]:
    raw = await r.lrange(f"prices:{self.symbol}", 0, -1)
    return [float(x) for x in raw]
```

### GPU-accelerated HMM

Replace `hmmlearn.hmm.GaussianHMM` with `pomegranate`:

```python
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

model = DenseHMM([Normal() for _ in range(3)])
model.fit(sequences)  # GPU via PyTorch backend
```

### Kafka consumer integration

```python
from aiokafka import AIOKafkaConsumer

async def consume_kafka(topic: str, symbol: str):
    consumer = AIOKafkaConsumer(topic, bootstrap_servers="localhost:9092")
    await consumer.start()
    session = await registry.get_or_create(symbol)
    async for msg in consumer:
        price = float(msg.value)
        await session.tick(price)
```

### Authentication (JWT on WebSocket)

```python
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer

@router.websocket("/ws/{symbol}")
async def websocket_stream(websocket: WebSocket, symbol: str, token: str = Query(...)):
    payload = verify_jwt(token)  # raises if invalid
    await websocket.accept()
    ...
```

---

## Monitoring

Key metrics to track in production:

| Metric | Alert threshold | Description |
|--------|-----------------|-------------|
| `tick_latency_p95` | > 100ms | Slow HMM predict path |
| `refit_latency_p95` | > 2000ms | HMM EM fitting taking too long |
| `session_count` | > 400 | Approaching per-worker memory limit |
| `alert_queue_depth` | > 500 | Alert broadcast queue backing up |
| `regime_change_rate` | > 10/hour | Possible noisy data or too-short window |

---

## HMM Tuning

| Scenario | Recommendation |
|----------|----------------|
| Short seed (< 150 bars) | Use `covariance_type="diag"`, `n_iter=50` |
| Long seed (≥ 500 bars) | Can use `covariance_type="full"`, `n_iter=150` |
| Slow regime convergence | Increase `refit_every` to 100+ bars |
| Too many false alerts | Increase debounce from 3 to 5 bars in `stream_session.py` |
| High-frequency data | Reduce `window` to 200; increase `refit_every` to 100 |
