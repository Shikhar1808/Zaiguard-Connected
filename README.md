# 🛡️ ZaiGuard — Integrated AI Surveillance & Alert System

Welcome to the unified repository for **ZaiGuard**, an intelligent, multi-layer AI-powered surveillance and event filtering system.

This repository integrates two core components:
1. **Zaiguard-Prototype**: The edge/node-side video ingestion, person detection (YOLOv8), tracking (ByteTrack), local schedule validation, and event deduplication pipeline.
2. **ZAIgaurd-alert-engine**: The FastAPI-based central middleware service that runs central database persistence (PostgreSQL/TimescaleDB), in-memory rate limiting (Redis), and semantic suppression filtering (Qdrant).

---

## 🏗️ System Architecture & Integration Flow

```
[ IP Camera / RTSP Feed ]
          │
          ▼
┌──────────────────────────────────────────────┐
│  Zaiguard-Prototype (Edge Node)              │
│  • Reads camera stream (cv2.VideoCapture)    │
│  • Runs YOLOv8 object detector               │
│  • Tracks people (ByteTrack)                 │
│  • unauth_access Classifier (Time schedule)  │
│  • Deduplicator (Burst suppression)          │
└──────────────────────┬───────────────────────┘
                       │
                       │ Confirmed Alert Candidate
                       │ (non-blocking background thread)
                       ▼
┌──────────────────────────────────────────────┐
│  ZaiGuard Alert Engine (FastAPI Service)     │
│  • Layer 1: Sensitivity Gate (time/zone)     │
│  • Layer 2: Metadata Enrichment              │
│  • Layer 3: Redis Deduplication & Escalation  │
│  • Layer 4A: Exact Rule Suppression (Postgres)│
│  • Layer 4B: Semantic Suppression (Qdrant)   │
│  • Layer 5: Alert Tiering & Dispatch         │
└──────────────────────┬───────────────────────┘
                       │
                       ├─────────────────────────┐
                       ▼                         ▼
            ┌─────────────────────┐   ┌─────────────────────┐
            │ PostgreSQL          │   │ Qdrant Vector DB    │
            │ (alert_log          │   │ (dismissed_alerts   │
            │ hypertable)         │   │ embeddings index)   │
            └─────────────────────┘   └─────────────────────┘
```

---

## 📌 Project Integration Status

| Component | Status | Details |
|---|---|---|
| **Multi-Camera Ingestion** | ✅ Complete | Ingests RTSP, webcam, HTTP streams, or video files with auto-fallback. |
| **YOLOv8 ONNX Backbone** | ✅ Complete | Runs YOLOv8 detections using ONNX Runtime with CUDA GPU/CPU fallback. |
| **ByteTrack Tracking** | ✅ Complete | Tracks people across frames with stable IDs. |
| **Re-ID Model Extraction** | 🚫 Removed | Removed the redundant local ONNX Re-ID appearance embedding extractor to align with the active integration (the Alert Engine automatically generates 384-d semantic embeddings from description text). |
| **Schedule Verification** | ✅ Complete | Validates camera violations based on schedules defined in `config/schedules.yaml`. |
| **Edge Deduplication** | ✅ Complete | Local confidence + camera cooldown gates prevent burst events from overloading the API. |
| **5-Layer Central Pipeline** | ✅ Complete |central filter funnel (Layer 1 Sensitivity, Layer 2 Enrichment, Layer 3 Redis Dedup, Layer 4 postgres/qdrant, Layer 5 tiering). |
| **Daily Rotating Logs** | ✅ Complete | Log files rotate daily at midnight with a 30-day backup retention. |
| **Automated Test Coverage** | ✅ Complete | Both folders feature comprehensive pytest suites verifying core pipeline logic. |

---

## ⚙️ Quick Start Setup

### Step 1: Clone the Repository & Configure Subprojects

Ensure you have **Python 3.10 to 3.12**, **uv** (for Python package management), and **Docker Desktop** installed.

#### Configure the Alert Engine (`ZAIgaurd-alert-engine`)
1. Open a terminal and navigate to the alert engine folder:
   ```bash
   cd ZAIgaurd-alert-engine
   ```
2. Set up the virtual environment:
   ```bash
   python -m venv z_env
   z_env\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy the environment file template:
   ```bash
   copy .env.example .env
   ```

#### Configure the Prototype (`Zaiguard-Prototype`)
1. Open a second terminal and navigate to the prototype folder:
   ```bash
   cd Zaiguard-Prototype
   ```
2. Synchronize dependencies using `uv`:
   ```bash
   uv sync
   ```

---

## 🚀 Running the Integrated System

### 1. Start Database and Caching Infrastructure
In the `ZAIgaurd-alert-engine` folder:
```bash
docker compose up -d
```
*This starts PostgreSQL, Redis, and Qdrant containers.*

### 2. Start the Alert Engine REST API
Within the `ZAIgaurd-alert-engine` folder (ensure the virtual environment `z_env` is active):
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Access the Swagger API documentation at: http://localhost:8000/docs.*

### 3. Start the Video Analytics Pipeline
In the `Zaiguard-Prototype` folder:
```bash
uv run main.py
```
*This launches the multi-camera pipeline, opens a live preview window, and starts detecting/tracking violations and pushing alert events to the central API.*

---

## 🧪 Verification & Testing

### Test Suite: Alert Engine
To verify the central 5-layer pipeline:
```bash
cd ZAIgaurd-alert-engine
z_env\Scripts\pytest tests/
```

### Test Suite: Prototype
To verify edge analytics and client dispatch:
```bash
cd Zaiguard-Prototype
uv run pytest tests/
```

---

## 📂 Repository Documentation Guide
* **[INTEGRATION_GUIDE.md](file:///c:/Users/saxen/Desktop/work/capstone/newMix/INTEGRATION_GUIDE.md)**: Deep dive into the integration protocol, schema versioning, and client configuration.
* **[Zaiguard-Prototype/remainingShikhar.md](file:///c:/Users/saxen/Desktop/work/capstone/newMix/Zaiguard-Prototype/remainingShikhar.md)**: Task list tracker detailing completed items and future tasks.
* **[Zaiguard-Prototype/FUTURE_IMPLEMENTATION_GUIDE.md](file:///c:/Users/saxen/Desktop/work/capstone/newMix/Zaiguard-Prototype/FUTURE_IMPLEMENTATION_GUIDE.md)**: Architectural guide for upcoming features (dashboard, storage adapters, federated learning).
