# Zaiguard — AI-Powered Surveillance System

A multi-stage, multi-threaded pipeline that watches one or more camera
feeds, detects people via YOLOv8, tracks them across frames, and fires
alerts when someone is detected on a camera outside its allowed access
hours.

This is the **first pipeline** of a larger surveillance system. The
architecture is built so additional event classifiers (violence, dog
attack, road accident) can be added without touching existing code — see
`docs/PIPELINE_CONTRACT.md` if you're implementing one of those.

---

## Project Status

| Component | Status |
|---|---|
| Multi-camera ingestion (RTSP / webcam / file / HTTP) | ✅ Complete |
| YOLOv8 ONNX backbone (GPU + CPU fallback) | ✅ Complete |
| ByteTrack person tracking | ✅ Complete |
| Spatial (6-d) + temporal (4-d) embeddings | ✅ Complete |
| Unauthorized access classifier | ✅ Complete |
| Deduplicator (two-gate: confidence + cooldown) | ✅ Complete |
| Alerter (log + JSONL + full JSON) | ✅ Complete |
| Live multi-camera preview (OpenCV) | ✅ Complete |
| Clip saving (MP4 default, JPEG sequence fallback) | ✅ Complete |
| Config-driven thresholds & clip settings | ✅ Complete |
| Auto-cleanup (retention policy) | ✅ Complete |
| Violence classifier | ❌ Not started |
| Dog attack classifier | ❌ Not started |
| Road accident classifier | ❌ Not started |
| Storage backends (TimescaleDB / Qdrant / Redis) | ❌ Not started |
| FastAPI dashboard backend | ❌ Not started |
| Next.js dashboard frontend | ❌ Not started |
| Federated learning | ❌ Not started |

---

## What this does right now

- Reads **any number of cameras** simultaneously (RTSP, webcam, video
  file, or HTTP stream — each with its own auto-fallback chain), minimum 1
- Runs YOLOv8 object detection via ONNX Runtime (GPU if available, CPU otherwise)
- Tracks people across frames with ByteTrack (stable IDs)
- Extracts rich embeddings per tracked person (128-d/512-d appearance, 6-d spatial, 4-d temporal) for future cross-camera re-identification and complex similarity search
- Compares detections against a per-camera time schedule — no zone
  drawing needed, the whole camera frame is the watched area
- Fires an alert when a person is seen on a **restricted** camera for
  enough consecutive frames
- Shows one live annotated preview window per camera (boxes, HUD,
  restricted banner, alert flash) — with a clear **NO INPUT** placeholder
  for any camera whose source can't be reached
- Saves alert snapshots + short clip sequences to disk
- Writes structured JSON/JSONL alert records (schema v1.3.0), including full embeddings and dashboard-ready metadata (severity, durations), ready for future storage integration (TimescaleDB / Qdrant)

---

## Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- At least one camera: USB webcam, RTSP stream, or a test video file —
  no upper limit on how many you can run at once
- (Optional) NVIDIA GPU + CUDA 12 + cuDNN 9 for GPU inference — falls
  back to CPU automatically if these aren't installed

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

This creates `.venv` and installs everything pinned in `pyproject.toml`
(opencv, onnxruntime, supervision, pydantic, loguru, fastapi, etc).

### 2. Get a detection model

You need a YOLOv8 ONNX model at the path configured in
`config/thresholds.yaml` (`backbone_model`). Two options:

**Option A — export one yourself:**
```bash
uv add ultralytics --dev
uv run tools/export_onnx.py --model yolov8n --size 640
```
This downloads YOLOv8n (~6MB) and writes `weights/backbone.onnx`.

**Option B — use an existing `.onnx` file** (e.g. `models/onnx/yolov8n.onnx`).
Just point `backbone_model` in `config/thresholds.yaml` at its path.

If no model is found, the pipeline runs in **PASSTHROUGH mode** — every
stage still runs, but zero detections are produced. Useful for testing
ingestion and the preview window without a model.

### 3. Appearance Embeddings Note

For the current integration, central semantic suppression is handled by description-based sentence embeddings generated inside the Alert Engine. The local ONNX Re-ID appearance embedding extractor is bypassed, and all tracked people default to using zero vectors (`[0.0] * 128`) for their appearance embedding stubs.

### 4. Configure your camera(s)

Copy the example and edit:
```bash
cp config/cameras.example.yaml config/cameras.yaml
```

Minimal working example (one webcam):
```yaml
cameras:
  - camera_id: cam_01
    label: Front entrance
    source: device:0          # or rtsp://user:pass@ip/stream, or a video file path
    enabled: true
    fps_cap: 25
    width: 1280
    height: 720
```

**There is no upper limit on the number of cameras** — add as many
entries as you have feeds for, each gets its own reader thread and its
own preview window. **A minimum of 1 enabled camera is required**; the
pipeline refuses to start with zero. See `docs/MULTI_CAMERA.md` for the
full rules, including what happens when a camera's source can't be reached.

`cameras.yaml` and `secrets.yaml` are gitignored — never commit real
credentials.

### 5. Configure access schedules

Edit `config/schedules.yaml`. Every camera you want monitored needs an
entry here, and the `camera_id` **must** match one in `cameras.yaml` or
config loading will fail validation.

```yaml
schedules:
  - camera_id: cam_01
    label: Front entrance
    restricted: true
    allowed:
      - start: 8     # 24h clock
        end: 20
        label: Business hours
```

Any detection outside the `allowed` windows triggers the threshold engine.
An empty `allowed: []` list means the camera is **always** restricted —
useful for testing or for areas nobody should ever enter (vaults, server rooms).

Cameras with no schedule entry at all are treated as unrestricted (never alert).

### 6. Tune thresholds (optional)

`config/thresholds.yaml` controls detection confidence, motion gating, the
re-identification model path, and the alert threshold engine (sliding
window size, score, cooldowns). Defaults work fine for a first run — see
inline comments for what each does.

---

## Running

```bash
uv run main.py                  # opens one live preview window per camera
uv run main.py --no-preview     # headless — for servers / SSH sessions
uv run main.py --log DEBUG      # verbose per-frame logging
uv run main.py --config myconf/ # use a different config directory
```

Press **Q** or **Esc** in any preview window to stop everything, or
Ctrl+C in the terminal.

### What you should see

On startup, with 3 cameras configured:
```
Config loaded | cameras=3 schedules=2
Pipeline running | cameras=3 classifiers=1
Preview active — press Q or Esc to quit
```

One window opens per enabled camera, each showing:
- Green boxes around detected people when that camera is in allowed hours
- Red boxes + a red "RESTRICTED ZONE" banner when it isn't
- A red full-screen flash for 2 seconds whenever an alert fires
- A drawn **"NO INPUT"** placeholder (camera icon, red slash, current
  status) if that camera's source couldn't be opened — other cameras with
  working sources keep running normally regardless

If literally every camera is disabled, the pipeline refuses to start at all:
```
ERROR  Failed to load config: No enabled cameras configured...
```

When an alert fires you'll see in the log:
```
ALERT unauth_access | cam=cam_01 track=3 time=02:17:43 conf=1.000 [7/7]
CONFIRMED | id=b22718d3  cam=cam_01  track=3  time=02:17:43  conf=1.000
ALERT FIRED | id=b22718d3  type=unauth_access  cam=cam_01  ...  clip=outputs\clips\b22718d3.jpg
```

---

## Output locations

```
outputs/
├── alerts/
│   ├── alert_log.jsonl              one compact line per alert, append-only
│   └── 2026-06-25/
│       └── <alert_id>.json          full alert record (embeddings + metadata)
└── clips/
    ├── <alert_id_prefix>.jpg        annotated snapshot at the moment of firing
    └── 2026-06-25/
        └── <alert_id_prefix>/
            └── clip.mp4             compressed clip (default, ~90% smaller than JPEG seq)

logs/
└── surveillance_YYYY-MM-DD.log      full debug log, rotated daily, 14-day retention
```

Clip format is configurable in `config/thresholds.yaml` — set `clip_format: jpeg_seq`
for the legacy per-frame JPEG sequence if needed.

---

## Project layout

```
core/            shared infrastructure — config loading, packet schemas,
                 logging, the thread/queue pipeline orchestrator
ingestion/       camera reading (N cameras), source fallback chain,
                 connection status tracking, motion-gated sampling
inference/       ONNX backbone + ByteTrack, re-ID feature extractor,
                 classifier runner
classifiers/     event detection logic — one file per event type
postproc/        deduplication, alert dispatch, live multi-camera
                 preview (with NO INPUT handling) + clip saving
storage/         (future) TimescaleDB / Qdrant / Redis adapters
federated/       (future) federated learning loop
dashboard/       (future) web dashboard
tools/           one-off scripts — ONNX export, re-ID model export
tests/           pytest unit tests
docs/            see "Documentation" below
```

---

## Documentation

| Document | What it covers |
|---|---|
| `docs/PIPELINE_CONTRACT.md` | The full contract for adding a new event classifier — packet schemas, embedding structure, metadata structure, threshold engine pattern, PR checklist |
| `docs/MULTI_CAMERA.md` | Minimum/maximum camera rules, NO INPUT semantics, per-camera vs. shared resources, scaling guidance |
| `docs/REMAINING_WORK.md` | Prioritized list of everything left to build, with what/why/how for each item |
| `docs/CHANGELOG.md` | Chronological record of major changes, each linking to the relevant doc above |
| `FUTURE_IMPLEMENTATION_GUIDE.md` | **How to build everything that's not yet implemented** — streaming sources (RTSP/remote cameras), FastAPI dashboard, Next.js frontend, storage backends (Redis/TimescaleDB/Qdrant), remaining classifiers (violence/dog attack/road accident), federated learning, with code samples and integration instructions |
| `remainingShikhar.md` | Current task tracker with ✅/❌ status for every component, priority order, and links to the implementation guide |

---

## Adding a new event classifier

See `docs/PIPELINE_CONTRACT.md` for the full contract. Short version:
subclass `BaseClassifier`, emit `AlertCandidate` objects with a typed
`AlertMeta` subclass, register it in `core/pipeline.py`, done.

---

## Future Roadmap

See `FUTURE_IMPLEMENTATION_GUIDE.md` for detailed implementation instructions
for each of the following:

1. **Dashboard (Phase 1)** — FastAPI backend + Next.js frontend reading
   existing JSONL/JSON alert files. No database needed. Gives a web UI for
   alert history, camera status, clip playback, and analytics.

2. **Storage backends** — Redis Streams (real-time push), TimescaleDB
   (time-series queries), Qdrant (vector similarity search on embeddings).
   Config stubs already exist in `thresholds.yaml` under `extensions`.

3. **New classifiers** — Violence detection, dog attack detection, road
   accident detection. All follow the same `BaseClassifier` pattern —
   see `docs/PIPELINE_CONTRACT.md`.

4. **Federated learning** — Per-camera local training of classifier heads
   with FedAvg aggregation. Backbone stays frozen.

---

## Troubleshooting

**`cameras=0` at startup**
`config/cameras.yaml` is missing, empty, or every camera has
`enabled: false`. The pipeline will refuse to start and tell you this
explicitly — copy from `cameras.example.yaml` and ensure at least one
camera has `enabled: true`.

**A camera's window shows "NO INPUT"**
That camera's source (RTSP/file/HTTP, then recordings folder, then device
scan 0-7) could not be opened on the most recent attempt. It keeps
retrying every 5 seconds in the background — other cameras are unaffected.
See `docs/MULTI_CAMERA.md` for the full fallback chain and the related
"stalled" case (connected but no recent frames).

**Config validation error: "Schedule references unknown camera_id"**
Every `camera_id` in `schedules.yaml` must also exist in `cameras.yaml`.
Keep both files in sync.

**`No ONNX model at '...' — PASSTHROUGH mode`**
Expected if you haven't exported/placed a detection model yet. The
pipeline still runs; classifiers just never fire. See Setup step 2.


**ONNX CUDA errors about missing DLLs (`cublasLt64_12.dll`, `cudnn64_9.dll`)**
You have `onnxruntime-gpu` installed but not the matching CUDA 12 + cuDNN 9
runtime. The pipeline catches this automatically and falls back to CPU —
the error lines from ONNX Runtime itself are harmless noise. To get GPU
inference, install CUDA 12 + cuDNN 9 matching your driver, or just stay on
CPU for a prototype.

**Webcam shows `[WARN] Failed to select stream 0` on Windows**
Harmless — it's the Media Foundation backend warning, the camera still opens.

**Nothing in `outputs/`**
Means no alert has fired yet. Either every camera is in allowed hours (no
violation), the sliding window hasn't filled yet (needs `unauth_min_frames`
consecutive detections), or no person has been detected at all. Run with
`--log DEBUG` to see per-frame, per-camera track counts.


**Always using `uv run`, never bare `python`**
`uv run main.py` guarantees you're using the locked dependency versions in
`.venv`. Running `python main.py` directly may silently use a different
interpreter or site-packages and produce confusing, inconsistent bugs.
