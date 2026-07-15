"""
classifiers/unauth_access.py

Camera-level unauthorized access classifier.

Core idea
---------
No polygons. No zone drawing. The entire camera field-of-view is the detection
area. The classifier asks one question per frame:

    "Is this camera currently in a restricted state AND is a person visible?"

A camera is in a restricted state when the current wall-clock hour does NOT
fall inside any of its configured allowed windows.

Examples
--------
Camera "server_room", allowed 08:00–18:00 on weekdays:
  → 02:17 any day:   restricted=True  — any detection → violation
  → 10:45 weekday:   restricted=False — detections ignored
  → 10:45 weekend:   restricted=True  — any detection → violation
  (weekday filtering not yet implemented but allowed list handles most cases)

Threshold engine (per (camera_id, track_id) pair)
--------------------------------------------------
  Step 1  Time gate   — is camera restricted right now?
          If not restricted → push False into window, nothing fires.

  Step 2  Class gate  — is this track a person (or other watched class)?
          Non-person classes (cars, bicycles) are ignored by default.

  Step 3  Sliding window  — deque of `unauth_min_frames` bools
          Slot = True if step 1 AND step 2 both hold.

  Step 4  Score  — frames_in_violation / frames_evaluated

  Step 5  Fire   — score >= unauth_score AND window full AND not in cooldown

Every decision is recorded in ThresholdVerdict for full auditability.
The alert metadata includes wall_time, hour_of_day, day_of_week, and
schedule_summary so the alert is self-explaining in any storage backend.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from datetime import datetime

from classifiers.base_classifier import BaseClassifier
from core.config_loader import CameraSchedule, ThresholdConfig
from core.logger import log
from core.packets import (
    SCHEMA_VERSION,
    AlertCandidate,
    AlertEmbeddings,
    AlertMeta,
    ThresholdVerdict,
    TrackBox,
    TrackPacket,
)

# Only these class labels are treated as potential intruders.
# Extend if your model detects other relevant classes (e.g. "motorcycle").
WATCHED_CLASSES = {"person"}


class UnauthAccessClassifier(BaseClassifier):
    event_type = "unauth_access"

    def __init__(
        self,
        schedules: list[CameraSchedule],
        thresholds: ThresholdConfig,
    ) -> None:
        super().__init__()
        # Index by camera_id for O(1) lookup
        self._schedules: dict[str, CameraSchedule] = {
            s.camera_id: s for s in schedules if s.restricted
        }
        self.cfg = thresholds

        # Sliding windows: (camera_id, track_id) → deque[bool]
        self._history: dict[tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.unauth_min_frames)
        )
        # Per-track last alert time
        self._last_alert: dict[tuple[str, int], float] = {}
        # Per-track first-seen timestamp for duration calculation (v1.3)
        self._track_first_seen: dict[tuple[str, int], tuple[float, int]] = {}

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        for cam_id, sched in self._schedules.items():
            log.info(
                "UnauthAccess watching | cam={} schedule={}",
                cam_id, sched.schedule_summary,
            )
        log.info(
            "UnauthAccessClassifier ready | cameras={} window={} score={:.2f} cooldown={:.0f}s",
            len(self._schedules), self.cfg.unauth_min_frames,
            self.cfg.unauth_score, self.cfg.unauth_cooldown_s,
        )
        self.ready.set()

    # ── gates ─────────────────────────────────────────────────────────────────

    def _is_camera_restricted(self, camera_id: str, hour: int) -> bool:
        """True = currently restricted (violation condition active)."""
        sched = self._schedules.get(camera_id)
        if sched is None:
            return False    # no schedule → unrestricted
        return not sched.is_allowed_now(hour)

    @staticmethod
    def _is_watched_class(track: TrackBox) -> bool:
        return track.class_label in WATCHED_CLASSES

    # ── threshold engine ──────────────────────────────────────────────────────

    def _evaluate(
        self,
        key: tuple[str, int],
        restricted: bool,
        is_watched: bool,
        schedule_summary: str,
    ) -> ThresholdVerdict:
        violation = restricted and is_watched
        self._history[key].append(violation)
        window = self._history[key]

        n_eval  = len(window)
        n_viol  = int(sum(window))
        score   = n_viol / n_eval if n_eval else 0.0
        passed  = (
            n_eval >= self.cfg.unauth_min_frames
            and score >= self.cfg.unauth_score
        )

        return ThresholdVerdict(
            window_size=self.cfg.unauth_min_frames,
            frames_in_violation=n_viol,
            frames_evaluated=n_eval,
            raw_score=round(score, 4),
            threshold=self.cfg.unauth_score,
            passed=passed,
            cooldown_active=self._in_cooldown(key),
            camera_restricted=restricted,
            schedule_summary=schedule_summary,
        )

    def _in_cooldown(self, key: tuple[str, int]) -> bool:
        last = self._last_alert.get(key)
        return last is not None and (time.time() - last) < self.cfg.unauth_cooldown_s

    # ── v1.3 helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _spatial_embedding_6d(track: TrackBox) -> list[float]:
        """6-d spatial: [cx_norm, cy_norm, w_norm, h_norm, area_ratio, aspect_ratio]."""
        cx, cy = track.centroid_norm
        x1, y1, x2, y2 = track.bbox_norm
        w = round(x2 - x1, 4)
        h = round(y2 - y1, 4)
        area_ratio = round(w * h, 6)   # fraction of frame area occupied
        return [cx, cy, w, h, area_ratio, track.aspect_ratio]

    @staticmethod
    def _temporal_embedding_4d(hour: int, dow_index: int) -> list[float]:
        """4-d cyclical: [sin(hour), cos(hour), sin(dow), cos(dow)]."""
        h_rad = 2 * math.pi * hour / 24.0
        d_rad = 2 * math.pi * dow_index / 7.0
        return [
            round(math.sin(h_rad), 6),
            round(math.cos(h_rad), 6),
            round(math.sin(d_rad), 6),
            round(math.cos(d_rad), 6),
        ]

    @staticmethod
    def _classify_severity(
        confidence: float,
        person_count: int,
        violation_ratio: float,
    ) -> str:
        """
        Auto-classify severity from context:
          critical:  high conf + multiple people + 100% violation
          high:      high conf + full violation window
          medium:    default
          low:       borderline score
        """
        if confidence >= 0.95 and person_count >= 3 and violation_ratio >= 1.0:
            return "critical"
        if confidence >= 0.85 and violation_ratio >= 1.0:
            return "high"
        if confidence < 0.70:
            return "low"
        return "medium"

    def _track_duration(
        self,
        key: tuple[str, int],
        current_time: float,
        current_frame: int,
    ) -> tuple[float, int]:
        """Return (duration_s, first_seen_frame) for a track."""
        if key not in self._track_first_seen:
            self._track_first_seen[key] = (current_time, current_frame)
        first_time, first_frame = self._track_first_seen[key]
        return round(current_time - first_time, 3), first_frame

    # ── main classify ─────────────────────────────────────────────────────────

    def classify(self, packet: TrackPacket) -> list[AlertCandidate]:
        cam_id   = packet.camera_id
        schedule = self._schedules.get(cam_id)

        # Skip entirely if this camera isn't being watched
        if schedule is None:
            return []

        now      = datetime.now()
        hour     = now.hour
        wall     = now.strftime("%H:%M:%S")
        dow      = now.strftime("%A")
        dow_idx  = now.weekday()              # 0=Monday … 6=Sunday
        sum_str  = schedule.schedule_summary
        restricted = not schedule.is_allowed_now(hour)

        # Frame dimensions from packet (v1.3)
        frame_h = packet.frame_shape[0] if packet.frame_shape else 0
        frame_w = packet.frame_shape[1] if len(packet.frame_shape) > 1 else 0

        alerts: list[AlertCandidate] = []

        for track in packet.tracks:
            if not self._is_watched_class(track):
                continue

            key = (cam_id, track.track_id)
            verdict = self._evaluate(key, restricted, True, sum_str)

            log.debug(
                "[{}] track={} restricted={} score={:.3f} passed={}",
                cam_id, track.track_id, restricted,
                verdict.raw_score, verdict.passed,
            )

            if not verdict.passed or verdict.cooldown_active:
                continue

            self._last_alert[key] = time.time()

            # v1.3: richer embeddings
            track_dur, first_frame = self._track_duration(
                key, packet.timestamp, packet.frame_id,
            )

            embeddings = AlertEmbeddings(
                appearance_embedding=track.appearance_embedding,
                spatial_embedding=self._spatial_embedding_6d(track),
                temporal_embedding=self._temporal_embedding_4d(hour, dow_idx),
                trajectory_embedding=[],
            )

            # v1.3: severity classification
            severity = self._classify_severity(
                confidence=verdict.raw_score,
                person_count=packet.scene_features.person_count,
                violation_ratio=verdict.raw_score,
            )

            meta = AlertMeta(
                # spatial
                bbox=track.bbox,
                bbox_norm=track.bbox_norm,
                centroid=track.centroid,
                centroid_norm=track.centroid_norm,
                area_px=track.area_px,
                aspect_ratio=track.aspect_ratio,
                # temporal
                wall_time=wall,
                hour_of_day=hour,
                day_of_week=dow,
                schedule_summary=sum_str,
                violation_history=list(self._history[key]),
                threshold_verdict=verdict,
                # scene
                scene=packet.scene_features,
                # track
                class_label=track.class_label,
                detection_score=track.score,
                # camera
                camera_id=cam_id,
                camera_label=schedule.label or cam_id,
                # v1.3 fields
                frame_width=frame_w,
                frame_height=frame_h,
                track_duration_s=track_dur,
                track_first_seen_frame=first_frame,
                severity=severity,
            )

            alert = AlertCandidate(
                schema_version=SCHEMA_VERSION,
                camera_id=cam_id,
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                event_type=self.event_type,
                track_id=track.track_id,
                confidence=round(verdict.raw_score, 4),
                zone_id=cam_id,
                embeddings=embeddings,
                meta=meta,
            )
            object.__setattr__(alert, "_frame", packet.frame)

            log.warning(
                "ALERT unauth_access | cam={} track={} time={} conf={:.3f} [{}/{}] severity={}",
                cam_id, track.track_id, wall,
                verdict.raw_score,
                verdict.frames_in_violation,
                verdict.frames_evaluated,
                severity,
            )
            alerts.append(alert)

        return alerts

    # ── maintenance ───────────────────────────────────────────────────────────

    def prune_stale_tracks(self, active_ids: set[int], camera_id: str) -> None:
        stale = [
            k for k in list(self._history)
            if k[0] == camera_id and k[1] not in active_ids
        ]
        for k in stale:
            self._history.pop(k, None)
            self._last_alert.pop(k, None)
            self._track_first_seen.pop(k, None)
        if stale:
            log.debug("Pruned {} stale track histories for {}", len(stale), camera_id)