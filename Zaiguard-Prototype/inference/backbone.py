"""
inference/backbone.py

Backbone inference thread.
Consumes FramePackets → ONNX detection → supervision ByteTrack → TrackPackets.

Fan-out
-------
Every processed TrackPacket is pushed to BOTH:
  - QueueBus.tracks         (classifier chain)
  - QueueBus.preview_frames (preview renderer — non-blocking, drop if full)

PASSTHROUGH mode
----------------
If no ONNX model is found at the configured path, this thread emits
TrackPackets with an empty tracks list. The rest of the pipeline keeps
running — classifiers simply never fire. This lets you test ingestion,
sampling, and the preview window without a model.
"""

from __future__ import annotations

import queue
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from core.config_loader import ThresholdConfig
from core.logger import log
from core.packets import FramePacket, SceneFeatures, TrackBox, TrackPacket

COCO_LABELS: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 16: "dog", 17: "cat",
}
_APPEARANCE_DIM = 128

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False
    log.warning("onnxruntime not installed — backbone in PASSTHROUGH mode")


def _build_session(model_path: str):
    """
    Build an ONNX Runtime session, preferring CUDA and falling back to CPU.
    Never raises — any failure (missing DLLs, missing API, missing model)
    results in either a CPU session or None (PASSTHROUGH mode).
    """
    if not _ORT_OK:
        return None
    if not Path(model_path).exists():
        log.warning("No ONNX model at '{}' — PASSTHROUGH mode", model_path)
        return None

    # get_available_providers exists on onnxruntime >= 1.10.
    # Some builds / odd installs may not expose it — guard and fall back.
    try:
        available = ort.get_available_providers()
    except AttributeError:
        available = []

    if "CUDAExecutionProvider" in available:
        try:
            sess = ort.InferenceSession(
                model_path,
                providers=[("CUDAExecutionProvider", {"device_id": 0}),
                           "CPUExecutionProvider"],
            )
            active = sess.get_providers()
            if "CUDAExecutionProvider" in active:
                log.info("ONNX loaded on GPU (CUDA) | {}", model_path)
            else:
                log.warning(
                    "ONNX: CUDA DLLs missing (cublasLt64_12 / cudnn64_9) — "
                    "running on CPU. Install CUDA 12 + cuDNN 9 to enable GPU."
                )
            return sess
        except Exception as exc:
            log.warning("ONNX CUDA init failed ({}) — falling back to CPU", exc)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    log.info("ONNX loaded on CPU | {}", model_path)
    return sess


def _preprocess(frame: np.ndarray, size: tuple) -> tuple[np.ndarray, float, float]:
    H, W = frame.shape[:2]
    resized = cv2.resize(frame, size)
    blob = resized.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, W / size[0], H / size[1]


def _postprocess_standard(boxes, scores, labels, sx, sy, conf_thr, nms_iou):
    mask = scores >= conf_thr
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    keep: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        nms_idx = cv2.dnn.NMSBoxes(
            boxes[idx].tolist(), scores[idx].tolist(), conf_thr, nms_iou,
        )
        if len(nms_idx):
            keep.extend(idx[np.array(nms_idx).flatten()])
    return boxes[keep], scores[keep], labels[keep].astype(int)


def _postprocess_yolov8(raw, sx, sy, conf_thr, nms_iou):
    pred = raw[0].T
    xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    labels = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(labels)), labels]
    boxes = np.zeros_like(xywh)
    boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) * sx
    boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) * sy
    boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2) * sx
    boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2) * sy
    return _postprocess_standard(boxes, scores, labels.astype(float),
                                  sx, sy, conf_thr, nms_iou)


def _to_track_boxes(
    det: sv.Detections,
    W: int,
    H: int,
    embeddings: list[list[float]] | None = None,
) -> list[TrackBox]:
    """
    embeddings, if provided, must be the same length and order as det —
    i.e. one appearance_embedding per detection, already computed by
    FeatureExtractor.extract_batch(). Falls back to zero vectors if not
    provided (keeps this function usable without the extractor wired).
    """
    if det.tracker_id is None or len(det) == 0:
        return []
    result = []
    for i in range(len(det)):
        x1, y1, x2, y2 = (int(v) for v in det.xyxy[i])
        score  = float(det.confidence[i]) if det.confidence is not None else 0.0
        cls_id = int(det.class_id[i]) if det.class_id is not None else 0
        tid    = int(det.tracker_id[i])
        w_b = max(x2 - x1, 1)
        h_b = max(y2 - y1, 1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        emb = embeddings[i] if embeddings is not None else [0.0] * _APPEARANCE_DIM
        result.append(TrackBox(
            track_id=tid,
            bbox=[x1, y1, x2, y2],
            bbox_norm=[round(x1/W, 4), round(y1/H, 4), round(x2/W, 4), round(y2/H, 4)],
            score=round(score, 4),
            class_id=cls_id,
            class_label=COCO_LABELS.get(cls_id, str(cls_id)),
            centroid=[cx, cy],
            centroid_norm=[round(cx/W, 4), round(cy/H, 4)],
            area_px=w_b * h_b,
            aspect_ratio=round(w_b / h_b, 3),
            appearance_embedding=emb,
        ))
    return result


def _build_scene(tracks: list[TrackBox], H: int, W: int) -> SceneFeatures:
    n = sum(1 for t in tracks if t.class_label == "person")
    return SceneFeatures(
        track_count=len(tracks),
        person_count=n,
        crowd_density=round(n / max(H * W, 1) * 1_000_000, 4),
    )


class BackboneInference(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
        thresholds: ThresholdConfig,
        stop_event: threading.Event,
        preview_queue: queue.Queue | None = None,
    ) -> None:
        super().__init__(name="backbone-inference", daemon=True)
        self.in_queue      = in_queue
        self.out_queue     = out_queue
        self.preview_queue = preview_queue
        self.cfg           = thresholds
        self.stop_event    = stop_event
        self._session      = None
        self._input_wh     = tuple(thresholds.backbone_input_size)
        self._trackers: dict[str, sv.ByteTrack] = {}

    def _get_tracker(self, camera_id: str) -> sv.ByteTrack:
        if camera_id not in self._trackers:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self._trackers[camera_id] = sv.ByteTrack(
                    track_activation_threshold=self.cfg.backbone_conf,
                    lost_track_buffer=30,
                    minimum_matching_threshold=0.8,
                    frame_rate=self.cfg.motion_sample_fps,
                )
        return self._trackers[camera_id]

    def _run_model(self, frame: np.ndarray):
        blob, sx, sy = _preprocess(frame, self._input_wh)
        inputs = {self._session.get_inputs()[0].name: blob}
        outputs = self._session.run(None, inputs)
        if len(outputs) == 1 and outputs[0].ndim == 3:
            return _postprocess_yolov8(outputs[0], sx, sy,
                                        self.cfg.backbone_conf, self.cfg.backbone_nms_iou)
        return _postprocess_standard(outputs[0], outputs[1], outputs[2],
                                      sx, sy, self.cfg.backbone_conf, self.cfg.backbone_nms_iou)

    def run(self) -> None:
        self._session = _build_session(self.cfg.backbone_model)
        mode = "ONNX" if self._session else "PASSTHROUGH"
        log.info("BackboneInference started | mode={} input={}", mode, self._input_wh)

        # Embedding creation feature disabled for current integration

        while not self.stop_event.is_set():
            try:
                packet: FramePacket = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            cam_id  = packet.camera_id
            frame   = packet.frame
            H, W    = frame.shape[:2]
            tracker = self._get_tracker(cam_id)

            t0 = time.monotonic()
            if self._session is not None:
                boxes, scores, labels = self._run_model(frame)
            else:
                boxes  = np.empty((0, 4), dtype=np.float32)
                scores = np.empty(0, dtype=np.float32)
                labels = np.empty(0, dtype=int)

            if len(boxes):
                det = sv.Detections(
                    xyxy=boxes.astype(np.float32),
                    confidence=scores.astype(np.float32),
                    class_id=labels.astype(int),
                )
                det = tracker.update_with_detections(det)
            else:
                det = sv.Detections.empty()

            # One embedding per surviving tracked detection.
            # Embedding creation feature disabled (retains zero vectors default).
            embeddings = None
            tracks = _to_track_boxes(det, W, H, embeddings)
            ms     = (time.monotonic() - t0) * 1000
            log.debug("[{}] {:.1f}ms  {} tracks", cam_id, ms, len(tracks))

            out = TrackPacket(
                camera_id=cam_id,
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                frame=frame,
                frame_shape=[H, W, frame.shape[2]],
                tracks=tracks,
                scene_features=_build_scene(tracks, H, W),
            )

            try:
                self.out_queue.put_nowait(out)
            except queue.Full:
                log.debug("[{}] tracks queue full — dropped", cam_id)

            if self.preview_queue is not None:
                try:
                    self.preview_queue.put_nowait(out)
                except queue.Full:
                    pass

        log.info("BackboneInference stopped")