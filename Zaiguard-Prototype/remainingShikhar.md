# Remaining Work — Surveillance System

This document tracks everything left to build, organized by owner and
priority. Read `PIPELINE_CONTRACT.md` first if you're implementing a new
classifier — this document tells you *what* to build, that one tells you
*how*.

For detailed implementation guides on future work (dashboard, storage,
new pipelines, streaming setup), see **`FUTURE_IMPLEMENTATION_GUIDE.md`**.

---

## Current Project Status (as of July 2026)

### ✅ Completed

| Component | Status | Notes |
|---|---|---|
| Multi-camera ingestion | ✅ Done | RTSP, webcam, file, HTTP — with auto-fallback chain |
| Source resolver + fallback | ✅ Done | `ingestion/source_resolver.py` — tries RTSP → file → HTTP → device |
| YOLOv8 ONNX backbone | ✅ Done | GPU if available, CPU fallback, PASSTHROUGH if no model |
| ByteTrack tracking | ✅ Done | Stable track IDs across frames |
| Re-ID feature extractor | 🚫 Removed | Bypassed local ONNX extractor (Alert Engine generates 384-d semantic description embeddings) |
| Spatial + temporal embeddings | ✅ Done | 6-d spatial, 4-d temporal in every alert |
| `unauth_access` classifier | ✅ Done | Sliding window + schedule-based detection |
| Deduplicator (two-gate) | ✅ Done | Min confidence + cross-camera cooldown |
| Alerter (3 sinks) | ✅ Done | Log line, compact JSONL, full JSON per alert |
| Live preview (multi-camera) | ✅ Done | One window per camera, NO INPUT placeholder, alert flash |
| Clip saving (MP4 + JPEG) | ✅ Done | Config-driven `clip_format: mp4` (default) or `jpeg_seq` |
| Snapshot saving | ✅ Done | Annotated JPEG at moment of alert firing |
| Config-driven clip/preview constants | ✅ Done | `clip_fps`, `clip_pre_s`, `clip_post_s` in `thresholds.yaml` |
| Storage cleanup | ✅ Done | Auto-delete old clips/alerts based on `retention_days` |
| Schema v1.3.0 | ✅ Done | Severity, frame_shape, track_duration, dashboard-ready metadata |
| Unit tests: unauth_access | ✅ Done | `tests/test_unauth_access.py` |
| Unit tests: deduplicator | ✅ Done | `tests/test_deduplicator.py` |
| Unit tests: backbone | ✅ Done | `tests/test_backbone.py` |
| Unit tests: source_resolver | ✅ Done | `tests/test_source_resolver.py` |
| Unit tests: config_loader | ✅ Done | `tests/test_config_loader.py` |
| Unit tests: alert_engine_client | ✅ Done | `tests/test_alert_engine_client.py` |
| Unit tests: alerter | ✅ Done | `tests/test_alerter.py` |
| Unit tests: pipeline | ✅ Done | `tests/test_pipeline.py` |

### ❌ Not Yet Built

| Component | Priority | Notes |
|---|---|---|
| Real RTSP camera integration (1.2) | 🔴 High | Config change only — see `FUTURE_IMPLEMENTATION_GUIDE.md` §1 |
| Threshold tuning on real footage (1.3) | 🔴 High | Requires real camera footage to calibrate |
| FastAPI dashboard backend (2.6) | 🟡 Medium | See `FUTURE_IMPLEMENTATION_GUIDE.md` §2 |
| Next.js dashboard frontend (2.6) | 🟡 Medium | See `FUTURE_IMPLEMENTATION_GUIDE.md` §3 |
| Redis Stream storage (2.4) | 🟠 Low | See `FUTURE_IMPLEMENTATION_GUIDE.md` §4.2 |
| TimescaleDB storage (2.4) | 🟠 Low | See `FUTURE_IMPLEMENTATION_GUIDE.md` §4.3 |
| Qdrant vector storage (2.4) | 🟠 Low | See `FUTURE_IMPLEMENTATION_GUIDE.md` §4.4 |
| Violence classifier (2.1) | 🟠 Low | See `FUTURE_IMPLEMENTATION_GUIDE.md` §5.2 |
| Dog attack classifier (2.2) | 🟠 Low | See `FUTURE_IMPLEMENTATION_GUIDE.md` §5.3 |
| Road accident classifier (2.3) | 🟠 Low | See `FUTURE_IMPLEMENTATION_GUIDE.md` §5.4 |
| Federated learning (2.5) | ⚪ Future | See `FUTURE_IMPLEMENTATION_GUIDE.md` §6 |

---

## New Issues
1. ~~**Reduce clip storage footprint:** Currently saving ~25 frames per clip (5s at 5fps) as individual JPEGs. Need to switch default `clip_format` to `mp4` in config.~~ **✅ RESOLVED** — `clip_format: mp4` is now the default in `thresholds.yaml`, producing ~90% smaller clips.

## Part 1 — Finishing the unauth_access pipeline (current owner)

### ~~1.1 Verify end-to-end on the real project folder~~ — ⚠️ NEEDS VERIFICATION

**What:** Confirm `cameras=1` (not `0`) at startup and that `backbone.py`
matches the latest version (no `CUDAExecutionProvider selected` log line
before the try/except guard).

**Why it matters:** File-sync issues between chat sessions and local disk
caused two separate regressions already — an old `backbone.py` silently
reappeared, and `cameras.yaml` reverted to empty. Both are config/file
problems, not code bugs, but they block every other step.

**How:**
```powershell
# Confirm the file is current — should print nothing
Select-String -Path inference\backbone.py -Pattern "CUDAExecutionProvider selected"

# Confirm cameras loaded
uv run main.py --log DEBUG
# Look for: Config loaded | cameras=1 schedules=1
```

---

### 1.2 Move from webcam to real camera — ❌ NOT STARTED

**What:** Replace `source: device:0` in `config/cameras.yaml` with a real
RTSP URL once a physical camera is available.

**Why it matters:** `device:0` was only ever a development convenience.
RTSP has different failure modes (network drops, auth failures, codec
mismatches) that the fallback chain in `source_resolver.py` already
handles, but you should test against the real stream before demo day —
RTSP timing and resolution often differ from what you tested with locally.

**How:** See `FUTURE_IMPLEMENTATION_GUIDE.md` §1 for full details including
remote camera setup (Tailscale, SSH tunnels, port forwarding).

```yaml
cameras:
  - camera_id: cam_01
    label: Front entrance
    source: rtsp://admin:yourpassword@192.168.1.100:554/stream1
    enabled: true
    fps_cap: 25
    width: 1920      # match the camera's actual resolution
    height: 1080
```
Run with `--log DEBUG` and watch for `[cam_01] Resolving source: rtsp://...`
followed by `Source opened`. If it fails, the log will show exactly which
fallback tier kicked in and why.

---

### 1.3 Tune thresholds against real footage — ❌ NOT STARTED

**What:** Adjust `unauth_min_frames`, `unauth_score`, `unauth_cooldown_s`,
`unauth_global_cooldown_s`, `unauth_min_confidence` in `config/thresholds.yaml`
based on observed false-positive / false-negative rates.

**Why it matters:** Current defaults (`window=7, score=0.60`) were chosen
to fire reliably during testing, not calibrated against a real scenario.
A window of 7 frames at 5fps = 1.4 seconds — fine for someone walking into
frame, too slow if you need near-instant detection, too fast if your
camera has flickering lights causing spurious detections.

**How to calibrate:**
1. Run the pipeline pointed at real footage for an extended period (e.g. an hour) with no actual intrusions.
2. Count false alerts in `outputs/alerts/alert_log.jsonl`.
3. If too many false positives: raise `unauth_score` (e.g. 0.60 → 0.75) or raise `unauth_min_frames` (e.g. 7 → 10).
4. If real intrusions are missed: lower `unauth_score` or `unauth_min_frames`, or lower `backbone_conf` so weaker detections still count.
5. Repeat. There is no universal correct value — it depends on camera placement, lighting, and how much latency is acceptable.

---

### 1.4 Add missing unit tests — ✅ RESOLVED

**What:** Unit tests for `deduplicator.py` (`tests/test_deduplicator.py`) and `backbone.py` (`tests/test_backbone.py`) have been fully written and verified.

**Why it matters:** Ensures the deduplicator's two-gate filtering and the backbone's ONNX-to-PASSTHROUGH CPU/GPU fallback branching work exactly as expected and cannot be broken by subsequent config updates.

---

### ~~1.5 Move hardcoded preview/clip constants into config~~ — ✅ COMPLETED

`_CLIP_FPS`, `_CLIP_PRE_S`, `_CLIP_POST_S` are now config-driven fields
in `ThresholdConfig` (`core/config_loader.py`) and exposed in
`config/thresholds.yaml`. The `PreviewRenderer` reads them from
`config.thresholds` in `__init__`. Additionally, `clip_format`, 
`clip_jpeg_quality`, and `snapshot_jpeg_quality` were added.

---

## Part 2 — Other pipelines (different owners, per the architecture diagram)

These are out of scope for the unauth_access pipeline but documented here
so the full picture is visible. Each follows the exact same pattern —
see `PIPELINE_CONTRACT.md` Section 8–11 for the step-by-step recipe.

**For detailed implementation instructions, see `FUTURE_IMPLEMENTATION_GUIDE.md`.**

### 2.1 `classifiers/violence.py` — ❌ NOT STARTED
Stub metadata (`ViolenceMeta`) already exists in `core/packets.py`.
Needs: motion energy computation (optical flow or pose-based), an
interaction graph between nearby tracks, and the same threshold-engine
pattern as `unauth_access.py`.

### 2.2 `classifiers/dog_attack.py` — ❌ NOT STARTED
Stub metadata (`DogAttackMeta`) already exists. Needs: pose estimation or
proximity-based heuristic between `dog` and `person` class detections,
likely requiring the backbone to also output a pose-keypoints head or a
secondary pose model.

### 2.3 `classifiers/road_accident.py` — ❌ NOT STARTED
Stub metadata (`RoadAccidentMeta`) already exists. Needs: trajectory
history per vehicle track, a conflict-prediction heuristic (e.g. closing
velocity + intersecting paths), and an "impact zone" bounding box
computation.

### 2.4 `storage/timescaledb.py`, `storage/qdrant.py`, `storage/redis_stream.py` — ❌ NOT STARTED
All three are empty `__init__.py` stubs. The `ConfirmedAlert` schema
(`core/packets.py`) is already updated to `v1.3.0` and designed to map directly onto these. Qdrant will use the new 6-d spatial and 4-d temporal embeddings for advanced search.

### 2.5 `federated/` — ❌ NOT STARTED
No code exists. Will need a `fl_client.py` (per-camera-node local
training of classifier heads only, per the architecture diagram) and
`fl_server.py` (FedAvg aggregation). Backbone stays frozen — only
classifier heads are federated.

### 2.6 `dashboard/` — ❌ NOT STARTED
No code exists. Will need a FastAPI app (`fastapi`/`uvicorn` are already
in `pyproject.toml`) serving live feeds, alert history, and heatmaps, per
the architecture diagram's "Dashboard" box. Should read from
`outputs/alerts/alert_log.jsonl` initially (which already contains full metadata like `severity` and `frame_shape` as of schema v1.3.0), then from TimescaleDB once
Part 2.4 is done.

---

## Priority order if time is limited before a demo

1. **1.1 and 1.2** — these block everything; without them nothing runs against a real camera.
2. **1.3** — calibration is what makes the demo look credible instead of randomly firing or missing events.
3. **Dashboard (Phase 1)** — gives you a visual web UI to show during a demo, reads existing JSONL files.
4. **1.4** — cheap insurance against last-minute config changes breaking silently.
5. Everything else (storage backends, new classifiers, federated learning) can wait — they don't affect whether the current pipeline works, only how complete the overall system is.