# Future Implementation Guide — Zaiguard Surveillance System

This document covers everything that's **not yet built** and explains how to
implement each piece and wire it into the existing project. Read this alongside
`docs/PIPELINE_CONTRACT.md` for classifier-specific patterns.

> **Current state (July 2026):** The `unauth_access` pipeline is fully
> functional end-to-end — ingestion, backbone, tracker, classifier,
> deduplicator, alerter, preview, clip saving. Everything below extends
> that foundation.

---

## Table of Contents

1. [Streaming Source Setup (RTSP / Remote Cameras)](#1-streaming-source-setup)
2. [FastAPI Dashboard Backend](#2-fastapi-dashboard-backend)
3. [Next.js Frontend](#3-nextjs-frontend)
4. [Full Working Alert System (Storage Backends)](#4-full-working-alert-system)
5. [Remaining Pipelines (Violence, Dog Attack, Road Accident)](#5-remaining-pipelines)
6. [Federated Learning](#6-federated-learning)
7. [Integration Map — How Everything Connects](#7-integration-map)

---

## 1. Streaming Source Setup

### 1.1 Same network (simplest)

If your machine and IP camera are on the same LAN/Wi-Fi:

```yaml
# config/cameras.yaml
cameras:
  - camera_id: cam_01
    label: Front entrance
    source: rtsp://admin:password@192.168.1.100:554/stream1
    enabled: true
    fps_cap: 25
    width: 1920
    height: 1080
```

That's it. `ingestion/source_resolver.py` handles RTSP natively via OpenCV.

### 1.2 Different network (camera is remote)

When the camera is on a completely different network (different location,
different ISP), you need a network bridge. Options ranked by ease of use:

#### Option A — Tailscale (recommended, 5 min setup)

1. Install [Tailscale](https://tailscale.com) on your dev machine
2. Install Tailscale on **any device** at the camera's site (laptop, Raspberry Pi, NVR, etc.)
3. On that device, enable subnet routing:
   ```bash
   tailscale up --advertise-routes=192.168.1.0/24
   ```
4. Approve the subnet route in the [Tailscale admin console](https://login.tailscale.com/admin/machines)
5. Your dev machine can now reach `192.168.1.100` (the camera) as if it were local
6. Use the camera's local IP in `cameras.yaml` — no URL changes needed

**Why this works:** Tailscale creates an encrypted WireGuard tunnel between
both devices. The subnet router forwards traffic to the camera's LAN.

#### Option B — Port forwarding

1. On the router at the camera's location, forward external port `9554` → `192.168.1.100:554`
2. Get the public IP of that network (or set up DDNS like `mycam.ddns.net`)
3. Use: `source: rtsp://admin:pass@mycam.ddns.net:9554/stream1`

⚠️ **Security risk**: RTSP is unencrypted. Anyone who finds your public IP
can sniff the stream. Only use for short-term demos, never in production.

#### Option C — SSH tunnel

```bash
# On your dev machine — forwards local port 9554 to camera via remote host
ssh -L 9554:192.168.1.100:554 user@remote-public-ip
```
Then use: `source: rtsp://admin:pass@localhost:9554/stream1`

#### Option D — RTSP relay with MediaMTX

Install [MediaMTX](https://github.com/bluenviern/mediamtx) on a machine at the
camera's site:

```yaml
# mediamtx.yml
paths:
  cam01:
    source: rtsp://admin:pass@192.168.1.100:554/stream1
```

Then expose MediaMTX over Tailscale/VPN/port-forward. Your dev machine connects
to the relay instead of the camera directly. Benefits: transcoding, bandwidth
control, reconnect handling.

#### Option E — Phone as a test camera

For quick testing without hardware:
- **Android**: Install "IP Webcam" → starts an RTSP server on your phone
- **iOS**: Install "RTSP Camera Server"
- Both devices must be on the same Wi-Fi
- Use the URL the app shows (e.g., `rtsp://192.168.1.42:8080/h264`)

### 1.3 How it connects to the project

No code changes needed. The entire streaming setup is config-driven:

```
cameras.yaml  →  ingestion/source_resolver.py  →  ingestion/rtsp_reader.py
                         ↓
              Falls back through: RTSP → file → HTTP → device:0–7
              Shows "NO INPUT" placeholder if all fail
              Retries every 5s in background
```

The source resolver (`ingestion/source_resolver.py`) already handles:
- RTSP URLs (any format)
- Local video files
- HTTP streams (MJPEG)
- USB webcams (`device:N`)
- Auto-fallback between all of the above

### 1.4 Common RTSP URL patterns by manufacturer

| Brand | Typical URL |
|---|---|
| Hikvision | `rtsp://admin:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://admin:pass@IP:554/h264Preview_01_main` |
| Axis | `rtsp://root:pass@IP/axis-media/media.amp` |
| Generic ONVIF | Check camera's web UI under Network → RTSP settings |

### 1.5 Troubleshooting streams

| Symptom | Likely cause | Fix |
|---|---|---|
| "NO INPUT" in preview | Wrong URL, firewall, or camera offline | Test URL with VLC first: Media → Open Network Stream |
| Frame drops / lag | Network bandwidth too low for full resolution | Lower `width`/`height` in `cameras.yaml` or use sub-stream URL |
| Green/corrupted frames | H.265 codec (OpenCV can't decode) | Switch camera to H.264 in its admin panel |
| Works in VLC but not here | Camera needs TCP transport | OpenCV defaults to TCP, should work — check if camera limits connections |

---

## 2. FastAPI Dashboard Backend

### 2.1 Overview

A REST API that serves alert data, camera status, and media files. FastAPI
and Uvicorn are already in `pyproject.toml` — no new dependencies needed.

### 2.2 File structure to create

```
dashboard/
├── __init__.py          (already exists, empty)
├── app.py               FastAPI application factory
├── routers/
│   ├── __init__.py
│   ├── alerts.py        GET /api/alerts, GET /api/alerts/{id}
│   ├── cameras.py       GET /api/cameras, GET /api/cameras/{id}/status
│   ├── clips.py         GET /api/clips/{path} — serve media files
│   └── ws.py            WebSocket /ws/alerts — real-time push
├── services/
│   ├── __init__.py
│   ├── alert_service.py     reads JSONL / JSON files (Phase 1) or DB (Phase 2)
│   └── camera_service.py    reads config + status board
├── models/
│   ├── __init__.py
│   └── responses.py     Pydantic response models for the API
└── main.py              Entry point: uvicorn.run(...)
```

### 2.3 Phase 1 — JSONL-backed (no database needed)

This phase reads directly from the files the pipeline already writes.

#### `dashboard/app.py`

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    app = FastAPI(title="Zaiguard Dashboard", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Next.js dev server
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from dashboard.routers import alerts, cameras, clips
    app.include_router(alerts.router, prefix="/api")
    app.include_router(cameras.router, prefix="/api")
    app.include_router(clips.router, prefix="/api")

    return app
```

#### Key API endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/alerts` | List alerts from `alert_log.jsonl` with pagination (`?page=1&size=20`), filters (`?camera_id=cam_01&severity=high&from=2026-07-01&to=2026-07-03`) |
| `GET` | `/api/alerts/{alert_id}` | Full alert JSON from `outputs/alerts/YYYY-MM-DD/{alert_id}.json` |
| `GET` | `/api/cameras` | List all cameras from config with current schedule status |
| `GET` | `/api/clips/{filename}` | Serve snapshot JPEGs and clip MP4s via `FileResponse` |
| `GET` | `/api/stats` | Aggregate stats: alerts per hour, per camera, severity breakdown |
| `WS`  | `/ws/alerts` | Real-time WebSocket push of new alerts as they fire |

#### `dashboard/routers/alerts.py` (core logic)

```python
import json
from pathlib import Path
from fastapi import APIRouter, Query

router = APIRouter(tags=["alerts"])

ALERT_LOG = Path("outputs/alerts/alert_log.jsonl")
ALERT_DIR = Path("outputs/alerts")

@router.get("/alerts")
def list_alerts(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    camera_id: str | None = None,
    severity: str | None = None,
):
    if not ALERT_LOG.exists():
        return {"alerts": [], "total": 0}

    lines = ALERT_LOG.read_text().strip().split("\n")
    alerts = [json.loads(line) for line in reversed(lines)]  # newest first

    # Filter
    if camera_id:
        alerts = [a for a in alerts if a.get("camera_id") == camera_id]
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]

    total = len(alerts)
    start = (page - 1) * size
    return {"alerts": alerts[start:start+size], "total": total, "page": page}

@router.get("/alerts/{alert_id}")
def get_alert(alert_id: str):
    # Search date directories for the full JSON
    for date_dir in sorted(ALERT_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        path = date_dir / f"{alert_id}.json"
        if path.exists():
            return json.loads(path.read_text())
    return {"error": "Alert not found"}, 404
```

#### Running the dashboard

```python
# dashboard/main.py
import uvicorn
from dashboard.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
uv run -m dashboard.main
# Dashboard at http://localhost:8000
# API docs at http://localhost:8000/docs (auto-generated by FastAPI)
```

#### Real-time alerts via WebSocket

```python
# dashboard/routers/ws.py
import asyncio
import json
from fastapi import APIRouter, WebSocket
from watchfiles import awatch  # pip install watchfiles

router = APIRouter()

@router.websocket("/ws/alerts")
async def alert_stream(ws: WebSocket):
    await ws.accept()
    alert_log = Path("outputs/alerts/alert_log.jsonl")
    last_size = alert_log.stat().st_size if alert_log.exists() else 0

    # Watch for file changes and push new lines
    async for _ in awatch(alert_log):
        current_size = alert_log.stat().st_size
        if current_size > last_size:
            with open(alert_log) as f:
                f.seek(last_size)
                new_lines = f.readlines()
            for line in new_lines:
                await ws.send_text(line.strip())
            last_size = current_size
```

### 2.4 Phase 2 — Database-backed (after storage backends)

Replace `alert_service.py` to query TimescaleDB instead of reading JSONL.
The API contract stays identical — the frontend doesn't change at all.

### 2.5 How it connects to the project

Two options for running the dashboard:

**Option A — Standalone process (recommended for development)**
```bash
# Terminal 1: Run the pipeline
uv run main.py

# Terminal 2: Run the dashboard
uv run -m dashboard.main
```

**Option B — Integrated into the pipeline**
Add to `core/pipeline.py`:
```python
if cfg.extensions.get("dashboard", {}).get("enabled"):
    import threading, uvicorn
    from dashboard.app import create_app
    dash_cfg = cfg.extensions["dashboard"]
    dash_app = create_app()
    threading.Thread(
        target=uvicorn.run,
        args=(dash_app,),
        kwargs={"host": dash_cfg["host"], "port": dash_cfg["port"]},
        daemon=True,
    ).start()
```

Then enable in `config/thresholds.yaml`:
```yaml
extensions:
  dashboard:
    enabled: true
    host: "0.0.0.0"
    port: 8000
```

---

## 3. Next.js Frontend

### 3.1 Overview

A modern React-based frontend that consumes the FastAPI backend. Provides
a polished UI for alert browsing, camera monitoring, and analytics.

### 3.2 Setup

```bash
# From the project root
cd dashboard
npx -y create-next-app@latest frontend --typescript --tailwind --app --src-dir --no-eslint --import-alias "@/*"
cd frontend
npm install
```

### 3.3 Pages to build

```
dashboard/frontend/src/app/
├── page.tsx                    Home / overview dashboard
├── alerts/
│   ├── page.tsx                Alert list with filters and search
│   └── [id]/page.tsx           Single alert detail view
├── cameras/
│   ├── page.tsx                Camera grid with live status
│   └── [id]/page.tsx           Single camera view + its alert history
├── analytics/
│   └── page.tsx                Charts: alerts over time, heatmaps
└── layout.tsx                  Sidebar navigation, dark theme
```

### 3.4 Key components

| Component | Purpose |
|---|---|
| `AlertCard` | Card showing severity badge, camera name, timestamp, thumbnail |
| `AlertTable` | Sortable/filterable table of alerts |
| `CameraGrid` | Grid of camera tiles showing status (active/restricted/offline) |
| `AlertTimeline` | Time-series chart of alert frequency (use `recharts` or `chart.js`) |
| `HeatmapView` | Hour × Camera heatmap of alert density |
| `ClipPlayer` | Video player for alert clips (HTML5 `<video>` for MP4s) |
| `LiveBadge` | Real-time alert count via WebSocket |
| `SeverityBadge` | Color-coded severity indicator (low/medium/high/critical) |

### 3.5 API integration

```typescript
// dashboard/frontend/src/lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function getAlerts(params?: {
  page?: number;
  size?: number;
  camera_id?: string;
  severity?: string;
}) {
  const query = new URLSearchParams(params as any).toString();
  const res = await fetch(`${API_BASE}/api/alerts?${query}`);
  return res.json();
}

export async function getAlert(id: string) {
  const res = await fetch(`${API_BASE}/api/alerts/${id}`);
  return res.json();
}

export function subscribeAlerts(onAlert: (alert: any) => void) {
  const ws = new WebSocket(`ws://localhost:8000/ws/alerts`);
  ws.onmessage = (e) => onAlert(JSON.parse(e.data));
  return ws;
}
```

### 3.6 Running

```bash
# Terminal 1: Pipeline
uv run main.py

# Terminal 2: FastAPI backend
uv run -m dashboard.main

# Terminal 3: Next.js frontend
cd dashboard/frontend && npm run dev
# Opens at http://localhost:3000
```

### 3.7 How it connects

```
┌─────────────┐     RTSP/USB      ┌──────────────────┐
│  IP Camera   │ ────────────────→ │  Pipeline         │
└─────────────┘                    │  (main.py)        │
                                   │                   │
                                   │  Writes:          │
                                   │  • alert_log.jsonl│
                                   │  • full JSONs     │
                                   │  • clip MP4s      │
                                   └───────┬───────────┘
                                           │ reads files
                                   ┌───────▼───────────┐
                                   │  FastAPI Backend   │
                                   │  (localhost:8000)  │
                                   │  /api/alerts       │
                                   │  /api/cameras      │
                                   │  /ws/alerts        │
                                   └───────┬───────────┘
                                           │ HTTP / WS
                                   ┌───────▼───────────┐
                                   │  Next.js Frontend  │
                                   │  (localhost:3000)  │
                                   └───────────────────┘
```

---

## 4. Full Working Alert System

### 4.1 Current state

The alert system already works end-to-end:

```
Classifier → AlertCandidate → Deduplicator → ConfirmedAlert → Alerter
                                                                  ↓
                                                    ┌─────────────┴──────────────┐
                                                    │ ✅ Log line (loguru)       │
                                                    │ ✅ Compact JSONL           │
                                                    │ ✅ Full JSON per alert     │
                                                    │ ✅ Snapshot JPEG           │
                                                    │ ✅ Clip MP4/JPEG sequence  │
                                                    │ ❌ Redis Stream            │
                                                    │ ❌ TimescaleDB             │
                                                    │ ❌ Qdrant vector store     │
                                                    └────────────────────────────┘
```

### 4.2 Adding Redis Streams (real-time push)

**Purpose:** Enables real-time alert broadcasting to the dashboard and any
other subscriber.

**Dependencies to add:**
```bash
uv add redis
```

**Create `storage/redis_stream.py`:**

```python
import json
import redis
from core.packets import ConfirmedAlert
from core.logger import log

class RedisAlertStream:
    def __init__(self, url: str = "redis://localhost:6379", stream: str = "alerts"):
        self.client = redis.from_url(url)
        self.stream = stream

    def publish(self, alert: ConfirmedAlert) -> None:
        """Publish alert to Redis Stream for real-time subscribers."""
        data = {
            "alert_id": alert.alert_id,
            "event_type": alert.event_type,
            "camera_id": alert.camera_id,
            "confidence": str(alert.confidence),
            "severity": alert.meta.severity,
            "ts_iso": alert.ts_iso,
            "snapshot_path": alert.snapshot_path or "",
            "clip_path": alert.clip_path or "",
        }
        try:
            self.client.xadd(self.stream, data, maxlen=10000)
        except Exception as exc:
            log.warning("Redis publish failed: {}", exc)
```

**Wire into `postproc/alerter.py`:**
```python
# In Alerter.__init__:
redis_cfg = config.extensions.get("storage", {}).get("redis", {})
if redis_cfg.get("enabled"):
    from storage.redis_stream import RedisAlertStream
    self._redis = RedisAlertStream(url=redis_cfg["url"])

# In Alerter._dispatch:
if hasattr(self, "_redis"):
    self._redis.publish(alert)
```

### 4.3 Adding TimescaleDB (structured time-series storage)

**Purpose:** Persistent, queryable storage for alerts with time-series
superpowers (automatic partitioning, downsampling, retention policies).

**Dependencies to add:**
```bash
uv add psycopg2-binary  # or asyncpg for async
```

**Create `storage/timescaledb.py`:**

```python
import json
import psycopg2
from core.packets import ConfirmedAlert
from core.logger import log

class TimescaleAlertStore:
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id     TEXT PRIMARY KEY,
                    ts           TIMESTAMPTZ NOT NULL,
                    event_type   TEXT NOT NULL,
                    camera_id    TEXT NOT NULL,
                    track_id     INTEGER,
                    confidence   DOUBLE PRECISION,
                    severity     TEXT,
                    camera_label TEXT,
                    clip_path    TEXT,
                    snapshot_path TEXT,
                    meta         JSONB,
                    embeddings   JSONB
                );
            """)
            # Convert to hypertable for time-series partitioning
            cur.execute("""
                SELECT create_hypertable('alerts', 'ts',
                    if_not_exists => TRUE,
                    migrate_data => TRUE);
            """)
            self.conn.commit()

    def upsert(self, alert: ConfirmedAlert) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO alerts (alert_id, ts, event_type, camera_id,
                    track_id, confidence, severity, camera_label,
                    clip_path, snapshot_path, meta, embeddings)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (alert_id) DO NOTHING
            """, (
                alert.alert_id, alert.ts_iso, alert.event_type,
                alert.camera_id, alert.track_id, alert.confidence,
                alert.meta.severity, alert.meta.camera_label,
                alert.clip_path, alert.snapshot_path,
                alert.meta.model_dump_json(),
                alert.embeddings.model_dump_json(),
            ))
            self.conn.commit()
```

### 4.4 Adding Qdrant (vector similarity search)

**Purpose:** "Find alerts similar to this one" — by appearance, location,
or time of day. Uses the embeddings already computed in `AlertEmbeddings`.

**Dependencies to add:**
```bash
uv add qdrant-client
```

**Create `storage/qdrant.py`:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, NamedVector
)
from core.packets import ConfirmedAlert
from core.logger import log

class QdrantAlertStore:
    def __init__(self, url: str, collection: str = "alerts"):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "appearance": VectorParams(size=128, distance=Distance.COSINE),
                    "spatial":    VectorParams(size=6,   distance=Distance.EUCLID),
                    "temporal":   VectorParams(size=4,   distance=Distance.COSINE),
                },
            )

    def upsert(self, alert: ConfirmedAlert) -> None:
        emb = alert.embeddings
        # Skip if no real embeddings
        if not emb.appearance_embedding:
            return
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(
                    id=alert.alert_id,
                    vector={
                        "appearance": emb.appearance_embedding,
                        "spatial":    emb.spatial_embedding,
                        "temporal":   emb.temporal_embedding,
                    },
                    payload={
                        "event_type": alert.event_type,
                        "camera_id":  alert.camera_id,
                        "severity":   alert.meta.severity,
                        "ts_iso":     alert.ts_iso,
                    },
                )],
            )
        except Exception as exc:
            log.warning("Qdrant upsert failed: {}", exc)
```

### 4.5 Wiring all storage backends

All three follow the same pattern — add to `Alerter._dispatch()`, gated
by config:

```python
# postproc/alerter.py — updated _dispatch method
def _dispatch(self, alert: ConfirmedAlert) -> None:
    self._dispatch_log(alert)
    self._dispatch_jsonl(alert)
    self._dispatch_full_json(alert)
    # Storage backends (enabled via config/thresholds.yaml extensions)
    if self._redis:
        self._redis.publish(alert)
    if self._timescale:
        self._timescale.upsert(alert)
    if self._qdrant:
        self._qdrant.upsert(alert)
```

Enable in `config/thresholds.yaml`:
```yaml
extensions:
  storage:
    timescaledb:
      enabled: true
      dsn: "postgresql://user:pass@localhost:5432/zaiguard"
    qdrant:
      enabled: true
      url: "http://localhost:6333"
      collection: alerts
    redis:
      enabled: true
      url: "redis://localhost:6379"
```

---

## 5. Remaining Pipelines

### 5.1 How classifiers work in this project

Every classifier follows the same pattern (see `docs/PIPELINE_CONTRACT.md`):

```
BaseClassifier subclass  →  receives TrackPacket  →  emits AlertCandidate
```

The infrastructure handles everything else (deduplication, alerting, storage).
You only write the detection logic.

### 5.2 `classifiers/violence.py`

**What it detects:** Physical violence between people (fighting, shoving, etc.)

**Stub already exists:** `ViolenceMeta` in `core/packets.py` (to be added)

**Implementation approach:**

1. **Subclass `BaseClassifier`** — follow the pattern in `classifiers/unauth_access.py`
2. **Detection logic** (pick one):
   - **Pose-based:** Use a lightweight pose model (MoveNet or MediaPipe) to detect
     aggressive body positions (raised fists, rapid arm movement). Requires a
     second ONNX model alongside the backbone.
   - **Motion energy:** Compute optical flow magnitude between consecutive frames.
     High motion energy concentrated between two nearby tracks = likely violence.
     Simpler, no extra model needed.
   - **Interaction graph:** For each pair of `person` tracks within a distance
     threshold, compute relative velocity and closing speed. Alert when two
     tracks are close + moving fast toward each other.
3. **Threshold engine:** Same sliding-window pattern as `unauth_access.py` —
   count how many recent frames show violence indicators, fire when ratio
   exceeds threshold.
4. **Register in pipeline:**
   ```python
   # core/pipeline.py
   from classifiers.violence import ViolenceClassifier
   self.runner.register(ViolenceClassifier(cfg.thresholds))
   ```
5. **Add config:**
   ```yaml
   # config/thresholds.yaml → thresholds section
   violence_min_frames: 5
   violence_score: 0.70
   violence_cooldown_s: 60.0
   violence_min_confidence: 0.60
   ```

### 5.3 `classifiers/dog_attack.py`

**What it detects:** A dog in aggressive proximity to a person.

**Implementation approach:**

1. **Backbone change:** The YOLOv8 backbone already detects `dog` (COCO class 16).
   Ensure `backbone.py` doesn't filter to `person`-only — it should pass all
   detected classes through to classifiers.
2. **Proximity heuristic:** For each frame, check if any `dog` track's bbox
   overlaps or is within N pixels of any `person` track's bbox. Use IoU or
   centroid distance.
3. **Optional — pose-based:** If a pose model is available, check if the dog's
   bounding box overlaps with the person's lower body (legs/ankles) keypoints.
4. **Same threshold engine pattern** as above.

### 5.4 `classifiers/road_accident.py`

**What it detects:** Vehicle collisions or near-misses.

**Implementation approach:**

1. **Backbone change:** Track vehicles (COCO classes: car=2, motorcycle=3,
   bus=5, truck=7). The backbone already can — just ensure class filtering
   passes them through.
2. **Trajectory history:** Maintain a per-track deque of recent centroid
   positions (last N frames). Compute velocity vector per track.
3. **Conflict prediction:** For each pair of vehicle tracks:
   - Compute closing velocity (are they getting closer?)
   - Project trajectories forward — do they intersect within T seconds?
   - Check for sudden speed changes (deceleration = braking = near-miss)
4. **Impact detection:** Sudden track disappearance (merged bboxes) or
   dramatic velocity change between frames = likely collision.
5. **Same threshold engine pattern**.

### 5.5 Registration pattern (all classifiers)

```python
# core/pipeline.py — _build() method
self.runner.register(UnauthAccessClassifier(cfg.schedules, cfg.thresholds))

# Add new ones here:
if cfg.extensions["classifiers"]["violence"]["enabled"]:
    from classifiers.violence import ViolenceClassifier
    self.runner.register(ViolenceClassifier(cfg.thresholds))

if cfg.extensions["classifiers"]["dog_attack"]["enabled"]:
    from classifiers.dog_attack import DogAttackClassifier
    self.runner.register(DogAttackClassifier(cfg.thresholds))

if cfg.extensions["classifiers"]["road_accident"]["enabled"]:
    from classifiers.road_accident import RoadAccidentClassifier
    self.runner.register(RoadAccidentClassifier(cfg.thresholds))
```

---

## 6. Federated Learning

### 6.1 Concept

Each camera node trains classifier heads locally on its own data. A central
server aggregates the updates (FedAvg). The backbone (YOLOv8 ONNX) stays
**frozen** — only lightweight classifier heads are federated.

### 6.2 Files to create

```
federated/
├── __init__.py       (exists, empty)
├── fl_client.py      Per-node local training + weight upload
└── fl_server.py      FedAvg aggregation server
```

### 6.3 Implementation approach

1. **`fl_client.py`**: After each confirmed alert, fine-tune the classifier
   head (a small MLP or logistic regression on top of embeddings) on the
   local alert data. Periodically send weight deltas to the server.
2. **`fl_server.py`**: FastAPI or gRPC server that receives weight updates
   from N clients, averages them (FedAvg), and pushes the updated weights
   back.
3. **Privacy:** Raw frames never leave the node — only model weight deltas
   are transmitted.

### 6.4 Libraries

- [Flower](https://flower.dev/) — production FL framework, handles FedAvg,
  communication, and client management out of the box
- Or roll your own with plain PyTorch + FastAPI if you want full control

---

## 7. Integration Map — How Everything Connects

```
                    ┌──────────────────────────────────────────────────┐
                    │              CONFIG LAYER                        │
                    │  cameras.yaml  schedules.yaml  thresholds.yaml   │
                    └──────────────┬───────────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────────┐
                    │           PIPELINE (main.py)                     │
                    │                                                  │
                    │  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
                    │  │ Ingest  │→ │ Backbone │→ │ Classifiers    │  │
                    │  │ (RTSP)  │  │ (YOLOv8) │  │ • unauth ✅    │  │
                    │  └─────────┘  └──────────┘  │ • violence ❌  │  │
                    │                              │ • dog_atk  ❌  │  │
                    │                              │ • accident ❌  │  │
                    │                              └───────┬────────┘  │
                    │                                      │           │
                    │  ┌──────────┐  ┌──────────┐  ┌──────▼────────┐  │
                    │  │ Preview  │← │ Dedup    │← │ Alert         │  │
                    │  │ (OpenCV) │  │          │  │ Candidates    │  │
                    │  └──────────┘  └────┬─────┘  └───────────────┘  │
                    │                     │                            │
                    │              ┌──────▼──────┐                    │
                    │              │   Alerter   │                    │
                    │              └──────┬──────┘                    │
                    └─────────────────────┼────────────────────────────┘
                                          │
                    ┌─────────────────────▼────────────────────────────┐
                    │              OUTPUT SINKS                        │
                    │                                                  │
                    │  ✅ Log line      ✅ JSONL       ✅ Full JSON    │
                    │  ✅ Snapshot      ✅ Clip MP4                    │
                    │  ❌ Redis Stream  ❌ TimescaleDB  ❌ Qdrant      │
                    └─────────────────────┬────────────────────────────┘
                                          │ reads files / DB
                    ┌─────────────────────▼────────────────────────────┐
                    │           DASHBOARD                              │
                    │  ❌ FastAPI backend  (localhost:8000)             │
                    │  ❌ Next.js frontend (localhost:3000)             │
                    └──────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────┐
                    │           FEDERATED LEARNING (future)            │
                    │  ❌ fl_client.py    ❌ fl_server.py               │
                    └──────────────────────────────────────────────────┘

    ✅ = implemented     ❌ = not yet built
```

### Dependency order for implementation

```
1. FastAPI Dashboard (Phase 1)     ← no dependencies, reads existing files
2. Redis Stream                    ← needs Redis running (docker)
3. TimescaleDB                     ← needs TimescaleDB running (docker)
4. Dashboard Phase 2               ← needs TimescaleDB
5. Qdrant                          ← needs Qdrant running (docker)
6. Next.js Frontend                ← needs FastAPI backend
7. Violence / Dog / Accident       ← independent, can be done anytime
8. Federated Learning              ← needs classifiers to have trainable heads
```

### Docker for infrastructure services

```bash
# Start all three services with one command
docker compose up -d

# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  timescaledb:
    image: timescale/timescaledb:latest-pg16
    ports: ["5432:5432"]
    environment:
      POSTGRES_PASSWORD: zaiguard
      POSTGRES_DB: zaiguard

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
```
