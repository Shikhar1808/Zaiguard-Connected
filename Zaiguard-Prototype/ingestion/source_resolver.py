"""
ingestion/source_resolver.py

Resolves a camera source string to a working cv2.VideoCapture.

Fallback chain
--------------
1. Explicit device:N
2. RTSP (CAP_FFMPEG first, then CAP_ANY)
3. Video file on disk
4. HTTP/MJPEG stream
5. Any .mp4/.avi in recordings/
6. Device scan indices 0–7
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import cv2

from core.logger import log

_OPEN_TIMEOUT_S  = 5
_DEVICE_SCAN_RANGE = range(8)


def _try_open(source: str | int, label: str) -> Optional[cv2.VideoCapture]:
    try:
        cap      = cv2.VideoCapture(source)
        deadline = time.monotonic() + _OPEN_TIMEOUT_S
        while time.monotonic() < deadline:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    log.info("Source opened — {} ({})", label, source)
                    return cap
            time.sleep(0.1)
        cap.release()
    except Exception as exc:
        log.debug("Open attempt failed for {}: {}", label, exc)
    return None


def _is_rtsp(s: str)         -> bool: return s.lower().startswith("rtsp://")
def _is_http(s: str)         -> bool: return bool(re.match(r"https?://", s, re.I))
def _is_device(s: str)       -> bool: return bool(re.match(r"^device:\d+$", s, re.I))
def _is_video_file(s: str)   -> bool:
    p = Path(s)
    return p.exists() and p.suffix.lower() in {
        ".mp4", ".avi", ".mkv", ".mov", ".h264", ".h265", ".ts", ".flv", ".webm",
    }


def _scan_devices() -> tuple[Optional[cv2.VideoCapture], Optional[str]]:
    for idx in _DEVICE_SCAN_RANGE:
        cap = _try_open(idx, f"device:{idx}")
        if cap is not None:
            return cap, f"device:{idx}"
    return None, None


def resolve_source(
    source: str,
    camera_id: str = "unknown",
) -> tuple[Optional[cv2.VideoCapture], Optional[str]]:
    log.info("[{}] Resolving source: {}", camera_id, source)

    if _is_device(source):
        idx = int(source.split(":")[1])
        cap = _try_open(idx, source)
        if cap:
            return cap, source
        log.warning("[{}] Device {} unavailable", camera_id, source)

    elif _is_rtsp(source):
        for backend, name in [(cv2.CAP_FFMPEG, "FFMPEG"), (cv2.CAP_ANY, "ANY")]:
            cap = _try_open(source, f"RTSP/{name}")
            if cap:
                return cap, source
        log.warning("[{}] RTSP unreachable: {}", camera_id, source)

    elif _is_video_file(source):
        cap = _try_open(source, "file")
        if cap:
            return cap, source
        log.warning("[{}] Cannot open file: {}", camera_id, source)

    elif _is_http(source):
        cap = _try_open(source, "HTTP")
        if cap:
            return cap, source
        log.warning("[{}] HTTP stream unavailable: {}", camera_id, source)

    else:
        log.warning("[{}] Unrecognised source format: {}", camera_id, source)

    # Fallback A — recordings folder
    recordings = Path("recordings")
    if recordings.is_dir():
        candidates = sorted(recordings.glob("*.mp4")) + sorted(recordings.glob("*.avi"))
        if candidates:
            log.warning("[{}] Trying local recording as fallback: {}", camera_id, candidates[0])
            cap = _try_open(str(candidates[0]), "fallback-file")
            if cap:
                return cap, str(candidates[0])

    # Fallback B — device scan
    log.warning("[{}] Scanning local device cameras (indices 0–7)…", camera_id)
    cap, dev = _scan_devices()
    if cap:
        log.warning("[{}] Using device camera as last-resort fallback: {}", camera_id, dev)
        return cap, dev

    log.error("[{}] All source tiers exhausted — camera disabled", camera_id)
    return None, None