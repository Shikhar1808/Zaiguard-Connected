"""
postproc/preview.py

Live preview window — runs in the MAIN thread (OpenCV imshow requirement on Windows).

Multi-camera support
---------------------
One window per configured-and-enabled camera, regardless of count (no
upper limit). Cameras are known from config at construction time, so a
window exists and shows a placeholder even before the first frame
arrives or if the camera's source never resolves.

No-input handling
------------------
  - Zero enabled cameras at all  ->  a single "NO CAMERAS CONFIGURED"
    window is shown (this case is also rejected earlier by config
    validation in core/config_loader.py, but the renderer handles it
    defensively too).
  - A configured camera whose source could not be opened by any
    fallback tier (RTSP, file, HTTP, device scan)  ->  that camera's
    window shows a "NO INPUT" placeholder with the camera_id and last
    known status, instead of disappearing or freezing on a stale frame.

What it draws (per live camera)
--------------------------------
  - Bounding boxes + track ID for every detected person (green = allowed, red = restricted)
  - Top-left HUD: camera ID, time, schedule status, person count
  - Bottom bar: frame ID, track count
  - Red "RESTRICTED" banner when camera is in violation state
  - Alert flash overlay for 2 seconds after each confirmed alert

Clip saving
-----------
On each ConfirmedAlert, saves a rolling buffer of frames as a JPEG
sequence in outputs/clips/YYYY-MM-DD/<alert_id>/, plus a flat annotated
snapshot. The buffer is kept as a deque per camera — no extra thread needed.
"""

from __future__ import annotations

import queue
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.config_loader import AppConfig, CameraConfig
from core.logger import log
from core.packets import ConfirmedAlert, TrackPacket
from ingestion.rtsp_reader import CameraStatus

# ── colours (BGR) ─────────────────────────────────────────────────────────────
_GREEN  = (50, 205, 50)
_RED    = (30,  30, 220)
_ORANGE = (0,  165, 255)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_YELLOW = (0,   215, 255)
_GRAY   = (90,   90,  90)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX

# How long (seconds) without a frame before a "connected" camera is
# considered stalled and shown as NO INPUT, even if status_board still
# says "connected" (covers the gap between last good frame and the
# reader thread noticing the read failed).
_STALL_TIMEOUT_S = 8.0

_PLACEHOLDER_SIZE = (960, 540)   # (W, H) for windows with no live frame yet


class PreviewRenderer:
    """
    Called from the main thread. Call .update() in a tight loop.
    Returns False when a window is closed (q / Esc / X button).
    """

    def __init__(
        self,
        config: AppConfig,
        preview_queue: queue.Queue,
        alert_queue: queue.Queue,
        output_dir: str = "outputs",
        status_board: CameraStatus | None = None,
    ) -> None:
        self.config        = config
        self.preview_queue = preview_queue
        self.alert_queue   = alert_queue
        self.status_board  = status_board
        self._clip_dir      = Path(output_dir) / "clips"
        self._clip_dir.mkdir(parents=True, exist_ok=True)

        # Clip settings from config (replaces hardcoded constants)
        th = config.thresholds
        self._clip_fps      = th.clip_fps
        self._clip_pre_s    = th.clip_pre_s
        self._clip_post_s   = th.clip_post_s
        self._clip_format   = th.clip_format
        self._clip_jpeg_q   = th.clip_jpeg_quality
        self._snap_jpeg_q   = th.snapshot_jpeg_quality

        # Build schedule lookup
        self._schedules = {s.camera_id: s for s in config.schedules}

        # Every ENABLED camera known from config — windows are created
        # for all of these up front, even before any frame arrives.
        self._enabled_cameras: list[CameraConfig] = [
            c for c in config.cameras if c.enabled
        ]

        # Rolling frame buffer: camera_id -> deque of (timestamp, frame)
        buf_size = self._clip_fps * (self._clip_pre_s + self._clip_post_s)
        self._buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=buf_size)
        )

        # Alert flash: camera_id -> expiry timestamp
        self._flash_until: dict[str, float] = {}

        # Last rendered packet per camera, and when it arrived
        self._last_packet: dict[str, TrackPacket] = {}
        self._last_packet_time: dict[str, float] = {}

        self._windows: set[str] = set()
        self._start_time: float = time.monotonic()  # grace period for window init

    # ── schedule helpers ──────────────────────────────────────────────────────

    def _is_restricted(self, camera_id: str) -> bool:
        sched = self._schedules.get(camera_id)
        if sched is None or not sched.restricted:
            return False
        return not sched.is_allowed_now(datetime.now().hour)

    def _window_name(self, camera_id: str) -> str:
        return f"Surveillance - {camera_id}"

    # ── no-input / placeholder drawing ────────────────────────────────────────

    def _camera_label(self, camera_id: str) -> str:
        for c in self._enabled_cameras:
            if c.camera_id == camera_id:
                return c.label or camera_id
        return camera_id

    def _is_stalled(self, camera_id: str) -> bool:
        """True if we have never received a frame, or haven't recently."""
        last_t = self._last_packet_time.get(camera_id)
        if last_t is None:
            return True
        return (time.monotonic() - last_t) > _STALL_TIMEOUT_S

    def _draw_no_input(self, camera_id: str) -> np.ndarray:
        """Placeholder frame for a camera with no live video right now."""
        W, H = _PLACEHOLDER_SIZE
        frame = np.full((H, W, 3), 24, dtype=np.uint8)   # near-black background

        status = self.status_board.get(camera_id) if self.status_board else "connecting"
        status_text = {
            "connecting": "CONNECTING...",
            "failed":     "NO INPUT",
            "connected":  "NO INPUT",   # connected but stalled (no recent frame)
        }.get(status, "NO INPUT")
        status_col = {
            "connecting": _YELLOW,
            "failed":     _RED,
            "connected":  _ORANGE,
        }.get(status, _RED)

        # Camera icon — simple drawn rectangle + lens circle, no external assets
        cx, cy = W // 2, H // 2 - 40
        cv2.rectangle(frame, (cx - 70, cy - 40), (cx + 70, cy + 40), _GRAY, 3)
        cv2.circle(frame, (cx, cy), 28, _GRAY, 3)
        cv2.circle(frame, (cx, cy), 12, _GRAY, -1)
        # Diagonal "no signal" slash
        cv2.line(frame, (cx - 90, cy - 60), (cx + 90, cy + 60), _RED, 4)

        (tw, _), _ = cv2.getTextSize(status_text, _FONT, 1.1, 2)
        cv2.putText(frame, status_text, (W // 2 - tw // 2, cy + 100),
                    _FONT, 1.1, status_col, 2, cv2.LINE_AA)

        label = self._camera_label(camera_id)
        (lw, _), _ = cv2.getTextSize(label, _FONT, 0.7, 1)
        cv2.putText(frame, label, (W // 2 - lw // 2, cy + 140),
                    _FONT, 0.7, _WHITE, 1, cv2.LINE_AA)

        sub = "Checking RTSP / file / HTTP / device sources..."
        (sw, _), _ = cv2.getTextSize(sub, _FONT, 0.5, 1)
        cv2.putText(frame, sub, (W // 2 - sw // 2, cy + 170),
                    _FONT, 0.5, _GRAY, 1, cv2.LINE_AA)

        return frame

    def _draw_no_cameras_configured(self) -> np.ndarray:
        """Single global placeholder when zero enabled cameras exist."""
        W, H = _PLACEHOLDER_SIZE
        frame = np.full((H, W, 3), 24, dtype=np.uint8)

        title = "NO CAMERAS CONFIGURED"
        (tw, _), _ = cv2.getTextSize(title, _FONT, 1.0, 2)
        cv2.putText(frame, title, (W // 2 - tw // 2, H // 2 - 20),
                    _FONT, 1.0, _RED, 2, cv2.LINE_AA)

        sub = "Add at least one camera to config/cameras.yaml"
        (sw, _), _ = cv2.getTextSize(sub, _FONT, 0.55, 1)
        cv2.putText(frame, sub, (W // 2 - sw // 2, H // 2 + 20),
                    _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)
        return frame

    # ── drawing (live frame) ──────────────────────────────────────────────────

    def _draw(self, packet: TrackPacket) -> np.ndarray:
        frame      = packet.frame.copy()
        cam_id     = packet.camera_id
        H, W       = frame.shape[:2]
        now        = datetime.now()
        restricted = self._is_restricted(cam_id)
        flashing   = time.monotonic() < self._flash_until.get(cam_id, 0)

        # ── bounding boxes ────────────────────────────────────────────────────
        for track in packet.tracks:
            x1, y1, x2, y2 = track.bbox
            color = _RED if restricted else _GREEN
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{track.track_id} {track.class_label} {track.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)

        # ── HUD top-left ──────────────────────────────────────────────────────
        sched  = self._schedules.get(cam_id)
        s_text = sched.schedule_summary if sched else "unrestricted"
        status = "RESTRICTED" if restricted else "ALLOWED"
        s_col  = _RED if restricted else _GREEN

        hud_lines = [
            (f"CAM: {cam_id}",                           _WHITE),
            (f"TIME: {now.strftime('%H:%M:%S')}",        _WHITE),
            (f"STATUS: {status}",                        s_col),
            (f"SCHEDULE: {s_text}",                      _YELLOW),
            (f"PERSONS: {packet.scene_features.person_count}", _WHITE),
        ]
        for i, (text, col) in enumerate(hud_lines):
            y = 24 + i * 22
            cv2.putText(frame, text, (10, y), _FONT, 0.55, _BLACK, 3, cv2.LINE_AA)
            cv2.putText(frame, text, (10, y), _FONT, 0.55, col,   1, cv2.LINE_AA)

        # ── bottom bar ────────────────────────────────────────────────────────
        bar_text = f"frame={packet.frame_id}  tracks={len(packet.tracks)}"
        cv2.rectangle(frame, (0, H - 24), (W, H), _BLACK, -1)
        cv2.putText(frame, bar_text, (8, H - 7), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)

        # ── RESTRICTED banner ─────────────────────────────────────────────────
        if restricted:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, H // 2 - 28), (W, H // 2 + 28), _RED, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.putText(frame, "!! RESTRICTED ZONE - MONITORING ACTIVE !!",
                        (W // 2 - 260, H // 2 + 8), _FONT, 0.7, _WHITE, 2, cv2.LINE_AA)

        # ── alert flash ───────────────────────────────────────────────────────
        if flashing:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), _RED, -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.putText(frame, "ALERT FIRED", (W // 2 - 90, 60),
                        _FONT, 1.2, _RED, 3, cv2.LINE_AA)

        return frame

    # ── clip saving ───────────────────────────────────────────────────────────

    def _save_clip(self, alert: ConfirmedAlert) -> tuple[str | None, str | None]:
        """Save snapshot + clip (MP4 or JPEG sequence). Returns (snap_path, clip_path)."""
        buf = self._buffers.get(alert.camera_id)
        snap_path: str | None = None
        clip_path: str | None = None

        last_packet = self._last_packet.get(alert.camera_id)
        if last_packet is not None:
            snap_path = str(self._clip_dir / f"{alert.alert_id[:8]}.jpg")
            cv2.imwrite(
                snap_path, self._draw(last_packet),
                [cv2.IMWRITE_JPEG_QUALITY, self._snap_jpeg_q],
            )
            log.info("Snapshot saved: {}", snap_path)

        if buf:
            clip_dir = self._clip_dir / time.strftime("%Y-%m-%d") / alert.alert_id[:8]
            clip_dir.mkdir(parents=True, exist_ok=True)

            if self._clip_format == "mp4":
                # Write a single compressed MP4 — ~90% smaller than JPEG sequence
                clip_path = str(clip_dir / "clip.mp4")
                first_frame = buf[0][1]
                h, w = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(clip_path, fourcc, self._clip_fps, (w, h))
                for _, frame in buf:
                    writer.write(frame)
                writer.release()
                log.info("Clip MP4 saved: {} ({} frames)", clip_path, len(buf))
            else:
                # Legacy: JPEG sequence
                clip_path = str(clip_dir)
                for i, (ts, frame) in enumerate(buf):
                    cv2.imwrite(
                        str(clip_dir / f"{i:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self._clip_jpeg_q],
                    )
                log.info("Clip sequence saved: {} ({} frames)", clip_dir, len(buf))
        else:
            log.warning("No frame buffer for {} — snapshot only", alert.camera_id)

        return snap_path, clip_path

    # ── main update — call from main thread loop ──────────────────────────────

    def update(self) -> bool:
        """
        Drain queues, render, show windows.
        Returns False when user closes a window or presses Q/Esc.
        """
        # ── drain confirmed alerts ────────────────────────────────────────────
        while True:
            try:
                alert: ConfirmedAlert = self.alert_queue.get_nowait()
                self._flash_until[alert.camera_id] = time.monotonic() + 2.0
                snap_path, clip_path = self._save_clip(alert)
                if snap_path is not None:
                    object.__setattr__(alert, "snapshot_path", snap_path)
                if clip_path is not None:
                    object.__setattr__(alert, "clip_path", clip_path)
            except queue.Empty:
                break

        # ── drain preview frames ──────────────────────────────────────────────
        while True:
            try:
                packet: TrackPacket = self.preview_queue.get_nowait()
                cam_id = packet.camera_id
                self._last_packet[cam_id] = packet
                self._last_packet_time[cam_id] = time.monotonic()

                buf = self._buffers[cam_id]
                if not buf or (packet.timestamp - buf[-1][0]) >= (1.0 / self._clip_fps):
                    buf.append((packet.timestamp, packet.frame.copy()))

            except queue.Empty:
                break

        # ── render ────────────────────────────────────────────────────────────
        if not self._enabled_cameras:
            # No cameras at all (defensive — config_loader should already
            # have rejected this at startup, but handle it gracefully too).
            win = "Surveillance - NO CAMERAS"
            if win not in self._windows:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, *_PLACEHOLDER_SIZE)
                self._windows.add(win)
            cv2.imshow(win, self._draw_no_cameras_configured())
        else:
            # One window per enabled camera, in config order — live frame
            # if we have one and it's fresh, otherwise a NO INPUT placeholder.
            for cam in self._enabled_cameras:
                cam_id = cam.camera_id
                win = self._window_name(cam_id)
                if win not in self._windows:
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win, 960, 540)
                    self._windows.add(win)

                if cam_id in self._last_packet and not self._is_stalled(cam_id):
                    vis = self._draw(self._last_packet[cam_id])
                else:
                    vis = self._draw_no_input(cam_id)

                cv2.imshow(win, vis)

        # ── key / window close ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            cv2.destroyAllWindows()
            return False

        # Skip window-close detection during the first 3 seconds —
        # on Windows, WND_PROP_VISIBLE can briefly return 0 before the
        # window is fully initialised, causing an immediate shutdown.
        if (time.monotonic() - self._start_time) > 3.0:
            for win in list(self._windows):
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        cv2.destroyAllWindows()
                        return False
                except Exception:
                    pass

        return True

    def close(self) -> None:
        cv2.destroyAllWindows()