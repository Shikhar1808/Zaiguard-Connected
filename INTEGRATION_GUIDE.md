# 🛡️ ZaiGuard Integrated System Guide

This guide explains how the **Zaiguard-Prototype** (Video analytics, tracking, and local classification) and the **ZaiGuard Alert Engine** (Middleware event filtering, database persistence, and semantic suppression) are integrated to work together as a single unified system.

---

## 🏗️ System Architecture & Data Flow

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
│  • Layer 2: Metadata Enrichment             │
│  • Layer 3: Redis Deduplication & Escalation │
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

### Flow of a Detection Event
1. **Detection & Tracking:** The Prototype processes video frames at a capped framerate. When a person is detected, their coordinates are converted to a ByteTrack track ID.
2. **Access Schedule Check:** The `unauth_access` classifier checks the current time against the camera's schedule in `config/schedules.yaml`. Detections outside allowed windows are marked as violations.
3. **Local Deduplication:** If consecutive violation frames cross the sliding window threshold (e.g. 7 out of 10 frames), a `ConfirmedAlert` is generated.
4. **Non-blocking Dispatch:** The Prototype's `AlertEngineClient` runs in a background thread and POSTs a JSON representation (`RawDetectionEvent` schema) to the Alert Engine's `/events` endpoint. The main analytics loop is never blocked by API latency.
5. **5-Layer Filtering:** The Alert Engine processes the incoming event:
   * **Layer 1 (Sensitivity):** Adjusts threshold sensitivity dynamically (e.g., night hours might have `0.75` multiplier, increasing sensitivity). If the event's raw confidence falls below the adjusted threshold, it is dropped.
   * **Layer 2 (Enrichment):** Precomputes time fields (hour of day, day of week) and camera details.
   * **Layer 3 (Redis Dedup):** Suppresses repeated events from the same camera unless a confidence spike occurs (Escalation).
   * **Layer 4A (Postgres Suppression):** Checks if the operator has defined an explicit suppression rule for this camera/hour.
   * **Layer 4B (Qdrant Suppression):** Computes sentence embeddings of the alert description and performs an approximate nearest neighbor (ANN) search in Qdrant to see if it matches historically operator-dismissed alerts. If highly similar (e.g., >90% similarity), the alert is suppressed.
   * **Layer 5 (Tiering & Storage):** Assigns a tier (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) and persists the alert to the permanent `alert_log` database.

---

## ⚙️ Configuration Setup

Both components have corresponding configuration parameters to connect with each other.

### 1. Prototype Side (`Zaiguard-Prototype`)
In [config/thresholds.yaml](file:///c:/Users/saxen/Desktop/work/capstone/newMix/Zaiguard-Prototype/config/thresholds.yaml):
```yaml
extensions:
  alert_engine:
    enabled: true
    url: "http://localhost:8000"
    timeout_s: 5
```
* `enabled`: Set to `true` to forward alerts to the Alert Engine.
* `url`: The endpoint where the Alert Engine server is running.
* `timeout_s`: Request timeout so that a lagging Alert Engine server does not hang background threads.

### 2. Alert Engine Side (`ZAIgaurd-alert-engine`)
In [.env](file:///c:/Users/saxen/Desktop/work/capstone/newMix/ZAIgaurd-alert-engine/.env):
```env
DATABASE_URL=postgresql+asyncpg://zaiguard:zaiguard_dev@localhost:5433/zaiguard
REDIS_HOST=localhost
REDIS_PORT=6380
QDRANT_HOST=localhost
QDRANT_PORT=6333
```
* Note that in `docker-compose.yml`, Postgres is forwarded to local port `5433` and Redis to `6380` to prevent ports colliding with any existing system-wide services.

---

## 🚀 Running the Integrated Project

Follow these steps to run the complete integrated stack locally:

### Step 1: Start the Infrastructure Services
In the `ZAIgaurd-alert-engine` folder, start PostgreSQL, Redis, and Qdrant containers:
```bash
cd ZAIgaurd-alert-engine
docker compose up -d
```
Verify that all containers are healthy:
```bash
docker compose ps
```

### Step 2: Run the Alert Engine API Server
Within the `ZAIgaurd-alert-engine` directory, activate the virtual environment and start the FastAPI server:
```bash
# Activate virtual environment
# On Windows:
z_env\Scripts\activate
# On macOS/Linux:
source z_env/bin/activate

# Run Uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The server will start at `http://localhost:8000`. You can inspect the Swagger API docs at `http://localhost:8000/docs`.

### Step 3: Run the Video Analytics Prototype
Open another terminal, navigate to the `Zaiguard-Prototype` directory, and start the camera analytics pipeline:
```bash
cd Zaiguard-Prototype
uv run main.py
```
* The pipeline will open a live OpenCV preview window showing the feed from your configured camera (e.g. `device:0`).
* Press `Q` inside the preview window to exit.

---

## 🧪 Testing and Verification

### Running Automated Test Suites
You can verify the logic of both repositories by running their respective automated test suites.

**For the Alert Engine:**
```bash
cd ZAIgaurd-alert-engine
z_env\Scripts\python -m pytest tests/ -v
```
*(Runs 232 unit & API integration tests verifying all 5 layers, feedback loops, and DB transactions)*

**For the Prototype:**
```bash
cd Zaiguard-Prototype
uv run pytest tests/ -v
```
*(Runs 37 unit & integration tests verifying camera source resolver, YOLOv8 backbone, ByteTrack, schedule validation, and the Alert Engine client mapping)*
