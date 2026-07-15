# 🛡️ ZaiGuard Alert Engine

> **Intelligent, Multi-Layer Event Filtering & Alert Prioritization Engine for Real-Time Video Analytics**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Asyncpg-316192.svg?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-In--Memory-DC382D.svg?style=flat&logo=redis&logoColor=white)](https://redis.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-E0234E.svg?style=flat)](https://qdrant.tech)

---

## 📌 Overview

Upstream computer vision models (detecting fire, violence, trespassing, accidents, and dog attacks) generate high-frequency raw detections. In real-world deployments, sending every raw detection directly to security operators causes **severe alert fatigue**, duplicate notification floods, and false positives.

**ZaiGuard Alert Engine** acts as an intelligent middleware funnel. It processes asynchronous event streams through a deterministic **5-Layer Pipeline**, filtering out 65%–90% of noise while ensuring zero dropped critical incidents and sub-second end-to-end latency.

---

## 🏗️ The 5-Layer Pipeline Architecture

```
[ Raw Vision Detections ] (JSON / REST API)
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  GATE 1: Threshold & Sensitivity Layer                 │
│  • Calculates Effective Threshold based on time/zone   │
│  • Drops low-confidence background noise               │
└──────────────────────────┬─────────────────────────────┘
                           │ Passed
                           ▼
┌────────────────────────────────────────────────────────┐
│  LAYER 2: Contextual Enrichment                        │
│  • Attaches zone risk multipliers & camera metadata    │
└──────────────────────────┬─────────────────────────────┘
                           │ Enriched Event
                           ▼
┌────────────────────────────────────────────────────────┐
│  LAYER 3: In-Memory Deduplication & Escalation (Redis) │
│  • Suppresses duplicate burst frames within TTL        │
│  • BREAKS dedup lock if confidence spikes (Escalation) │
└──────────────────────────┬─────────────────────────────┘
                           │ Deduplicated Event
                           ▼
┌────────────────────────────────────────────────────────┐
│  LAYER 4: False-Positive & Semantic Suppression        │
│  • 4A: Exact Rule Matching (Postgres SQL)              │
│  • 4B: Semantic Similarity Search (Qdrant Vector DB)   │
└──────────────────────────┬─────────────────────────────┘
                           │ Unsuppressed Threat
                           ▼
┌────────────────────────────────────────────────────────┐
│  LAYER 5: Alert Tiering & Dispatch                     │
│  • Assigns tier (CRITICAL, HIGH, MEDIUM, LOW)          │
│  • Persists to audit log & dispatches to dashboard     │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started (Teammate Onboarding)

### Prerequisites
* **Python 3.10+**
* **Docker & Docker Compose** (for local PostgreSQL, Redis, and Qdrant instances)

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/YOUR_ORG/ZaiGuard-Alert-Engine.git
cd ZaiGuard-Alert-Engine

# Create and activate virtual environment
python -m venv z_env

# On Windows:
z_env\Scripts\activate
# On macOS/Linux:
source z_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Configuration
We use a `.env` file for local development configuration.

> [!NOTE]
> **Why we use `.env.example`:**  
> The `.env` file is excluded from Git to prevent leaking sensitive passwords or keys. The repository provides a `.env.example` file containing placeholder names and harmless local defaults so teammates can set up immediately.

Copy the template to create your local `.env` file:
```bash
# On Windows PowerShell / CMD:
copy .env.example .env

# On macOS/Linux:
cp .env.example .env
```

### 3. Start Supporting Infrastructure
Spin up local Redis, PostgreSQL, and Qdrant containers:
```bash
docker compose up -d
```

### 4. Run the API Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
* Interactive Swagger API Docs: `http://127.0.0.1:8000/docs`
* Health Check Endpoint: `http://127.0.0.1:8000/health`

---

## 🧪 Testing & Verification

Our test suite includes **232 unit and integration tests** verifying all five pipeline layers, database transactions, vector searches, and API endpoints.

```bash
# Run full test suite
python -m pytest tests/ -v
```

---

## 🎮 Traffic Simulation & Load Testing

The repository includes a multi-mode traffic simulator (`scripts/simulate_traffic.py`) to demonstrate the engine's real-time filtering capabilities without needing live cameras.

```bash
# Mode 1: Realistic Campus Traffic (Dynamic multi-camera filtering)
python scripts/simulate_traffic.py --mode realistic --duration 15

# Mode 2: Burst Deduplication Demo (Proving Layer 3 storm prevention)
python scripts/simulate_traffic.py --mode burst

# Mode 3: Escalation Demo (Proving severity spikes bypass deduplication)
python scripts/simulate_traffic.py --mode escalation

# Mode 4: False-Positive Feedback Loop Demo (Proving Qdrant semantic suppression)
python scripts/simulate_traffic.py --mode feedback-loop

# Mode 5: Concurrency Stress Benchmark (Measuring throughput & latency profile)
python scripts/simulate_traffic.py --mode stress --requests 100 --concurrency 10
```

---

## 📂 Project Structure

```text
ZaiGuard-Alert-Engine/
├── config/              # Single source of truth settings & threshold calculations
├── db/                  # Async SQLAlchemy session management & initial SQL seeds
├── layers/              # Core 5-layer filtering pipeline implementations
│   ├── enrichment/      # Layer 2: Metadata enrichment
│   ├── suppression/     # Layer 3: Redis Dedup & Layer 4: Postgres + Qdrant
│   └── tiering/         # Layer 5: Alert classification & persistence
├── models/              # Pydantic schemas & SQLAlchemy ORM models
├── scripts/             # Simulation CLI tools & background cron jobs
├── tests/               # 232 automated test cases (unit + API integration)
├── docker-compose.yml   # Infrastructure orchestration (Postgres, Redis, Qdrant)
├── Dockerfile           # Multi-stage production container build
├── main.py              # FastAPI application & lifecycle handlers
└── pipeline.py          # Main engine orchestrator
```

---

## 🐳 Docker Production Build

Build and run the entire self-contained application container:
```bash
docker build -t zaiguard-engine:latest .
docker run -p 8000:8000 --env-file .env zaiguard-engine:latest
```
