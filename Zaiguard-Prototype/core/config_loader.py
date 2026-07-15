"""
core/config_loader.py

Pydantic v2 config models.

Design change: no zones/polygons.
The unit of access control is the camera itself.
Each camera has a schedule — a list of allowed time windows.
Any detection outside ALL windows is a violation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Time window — one contiguous allowed period
# ─────────────────────────────────────────────────────────────────────────────

class TimeWindow(BaseModel):
    """
    A single allowed access window.

    start / end are integers 0–23 (hour of day, 24-h clock).
    Overnight windows are supported: start=22, end=6 means 22:00–06:00.
    label is optional free text shown in alert meta.

    Examples
    --------
    Office hours only:  start=8,  end=18
    Overnight guard:    start=20, end=6
    Always open:        start=0,  end=23  (or just omit the camera from cameras.yaml)
    """
    start: int = Field(ge=0, le=23)
    end: int   = Field(ge=0, le=23)
    label: str = ""

    def is_active(self, hour: int) -> bool:
        """Return True if *hour* falls inside this allowed window."""
        if self.start <= self.end:
            return self.start <= hour <= self.end
        # Overnight wrap: e.g. 22 → 6
        return hour >= self.start or hour <= self.end

    @field_validator("start", "end", mode="before")
    @classmethod
    def coerce_int(cls, v: Any) -> int:
        return int(v)


# ─────────────────────────────────────────────────────────────────────────────
# Camera schedule — one entry per camera in cameras.yaml
# ─────────────────────────────────────────────────────────────────────────────

class CameraSchedule(BaseModel):
    """
    Access schedule for one camera.

    camera_id    must match an entry in cameras.yaml.
    allowed      list of TimeWindow. Any detection that does NOT fall inside
                 at least one window is a violation.
    restricted   false → classifier skips this camera entirely.

    If a camera has no schedule entry at all, it defaults to unrestricted
    (pipeline.py creates a pass-through CameraSchedule with restricted=False).
    """
    camera_id: str
    label: str = ""                          # human-readable camera name for alerts
    restricted: bool = True
    allowed: list[TimeWindow] = Field(default_factory=list)

    def is_allowed_now(self, hour: int) -> bool:
        """
        True  → current hour is inside at least one allowed window.
        False → violation (any detection should be flagged).
        An empty allowed list means "never allowed" (always restricted).
        """
        if not self.restricted:
            return True
        return any(w.is_active(hour) for w in self.allowed)

    @property
    def schedule_summary(self) -> str:
        """Human-readable summary for log lines."""
        if not self.restricted:
            return "unrestricted"
        if not self.allowed:
            return "always restricted"
        parts = [f"{w.start:02d}:00-{w.end:02d}:00" for w in self.allowed]
        return "allowed " + ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Camera hardware config
# ─────────────────────────────────────────────────────────────────────────────

class CameraConfig(BaseModel):
    camera_id: str
    source: str
    label: str = ""
    enabled: bool = True
    fps_cap: int = Field(default=25, ge=1, le=120)
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdConfig(BaseModel):
    # ── backbone ──────────────────────────────────────────────────────────────
    backbone_model: str = "weights/backbone.onnx"
    backbone_input_size: list[int] = Field(default=[640, 640])
    backbone_conf: float = Field(default=0.40, ge=0.0, le=1.0)
    backbone_nms_iou: float = Field(default=0.45, ge=0.0, le=1.0)

    # ── motion gate ───────────────────────────────────────────────────────────
    motion_threshold: float = Field(default=4.0, ge=0.0)
    motion_sample_fps: int = Field(default=5, ge=1, le=60)

    # ── unauth_access ─────────────────────────────────────────────────────────
    # Sliding window: how many recent frames to evaluate
    unauth_min_frames: int = Field(default=7, ge=1)
    # Fraction of window frames that must be violations to fire
    unauth_score: float = Field(default=0.60, ge=0.0, le=1.0)
    # Per-track cooldown: seconds before the same track can re-trigger
    unauth_cooldown_s: float = Field(default=30.0, ge=0.0)
    # Global cooldown: suppress same camera firing again within N seconds
    unauth_global_cooldown_s: float = Field(default=15.0, ge=0.0)
    # Second gate in deduplicator — independent of classifier score
    unauth_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)

    # ── clip / snapshot settings ─────────────────────────────────────────────────
    clip_fps:      int = Field(default=5,  ge=1, le=30)
    clip_pre_s:    int = Field(default=3,  ge=0)
    clip_post_s:   int = Field(default=2,  ge=0)
    clip_format:   str = "mp4"          # "mp4" or "jpeg_seq"
    clip_jpeg_quality: int = Field(default=70, ge=10, le=100)
    snapshot_jpeg_quality: int = Field(default=80, ge=10, le=100)

    @field_validator("backbone_input_size")
    @classmethod
    def input_size_valid(cls, v: list) -> list:
        if len(v) != 2 or not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("backbone_input_size must be [W, H] positive ints")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# App-level
# ─────────────────────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)
    schedules: list[CameraSchedule] = Field(default_factory=list)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    queue_maxsize: int = Field(default=32, ge=1)
    log_level: str = "INFO"
    alert_output_dir: str = "outputs"
    save_clips: bool = True
    retention_days: int = Field(default=30, ge=0)   # 0 = no auto-cleanup

    # ── Part 2 extension points ────────────────────────────────────────────────
    # Each key maps to a dict of sub-config for that module.
    # Part 2 owners enable their module by setting enabled=True.
    extensions: dict[str, Any] = Field(default_factory=lambda: {
        "classifiers": {
            "violence":      {"enabled": False, "model": "weights/violence.onnx"},
            "dog_attack":    {"enabled": False, "model": "weights/dog_attack.onnx"},
            "road_accident": {"enabled": False, "model": "weights/road_accident.onnx"},
        },
        "storage": {
            "timescaledb": {"enabled": False, "dsn": ""},
            "qdrant":      {"enabled": False, "url": "http://localhost:6333", "collection": "alerts"},
            "redis":       {"enabled": False, "url": "redis://localhost:6379"},
        },
        "federated": {"enabled": False, "server_url": ""},
        "dashboard":  {"enabled": False, "host": "0.0.0.0", "port": 8000},
    })

    @model_validator(mode="after")
    def at_least_one_enabled_camera(self) -> "AppConfig":
        enabled = [c for c in self.cameras if c.enabled]
        if not enabled:
            raise ValueError(
                "No enabled cameras configured. At least 1 camera is required "
                "in config/cameras.yaml (enabled: true). There is no upper limit "
                "on the number of cameras."
            )
        return self

    @model_validator(mode="after")
    def schedules_reference_valid_cameras(self) -> "AppConfig":
        cam_ids = {c.camera_id for c in self.cameras}
        for s in self.schedules:
            if s.camera_id not in cam_ids and cam_ids:
                raise ValueError(
                    f"Schedule references unknown camera_id '{s.camera_id}'. "
                    f"Known cameras: {sorted(cam_ids)}"
                )
        return self

    def schedule_for(self, camera_id: str) -> CameraSchedule:
        """Return the schedule for a camera, or an unrestricted default."""
        for s in self.schedules:
            if s.camera_id == camera_id:
                return s
        return CameraSchedule(camera_id=camera_id, restricted=False)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path = "config") -> AppConfig:
    base = Path(config_dir)

    raw_cameras   = _load_yaml(base / "cameras.yaml")
    raw_schedules = _load_yaml(base / "schedules.yaml")
    raw_thresh    = _load_yaml(base / "thresholds.yaml")

    cameras = [CameraConfig(**c) for c in raw_cameras.get("cameras", [])]

    schedules = []
    for s in raw_schedules.get("schedules", []):
        s = dict(s)
        raw_windows = s.pop("allowed", [])
        windows = [TimeWindow(**w) for w in raw_windows]
        schedules.append(CameraSchedule(allowed=windows, **s))

    thresh_raw = raw_thresh.get("thresholds", {})
    thresholds = ThresholdConfig(**thresh_raw) if thresh_raw else ThresholdConfig()
    app_raw    = raw_thresh.get("app", {})

    # Extensions live at top-level in thresholds.yaml (alongside app/thresholds)
    extensions_raw = raw_thresh.get("extensions", {})

    kwargs = {k: v for k, v in app_raw.items() if k in AppConfig.model_fields}
    if extensions_raw:
        kwargs["extensions"] = extensions_raw

    return AppConfig(
        cameras=cameras,
        schedules=schedules,
        thresholds=thresholds,
        **kwargs,
    )