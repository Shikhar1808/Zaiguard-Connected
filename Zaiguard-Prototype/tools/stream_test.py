"""
tools/stream_test.py

Quick diagnostic tool to test RTSP / HTTP / phone camera connections
before running the full pipeline.

Usage:
    uv run tools/stream_test.py                           # test all cameras in cameras.yaml
    uv run tools/stream_test.py rtsp://localhost:8554/cam_01  # test a single URL
    uv run tools/stream_test.py --phone 192.168.1.42      # test Android IP Webcam at this IP
    uv run tools/stream_test.py --mediamtx                # check MediaMTX API status

Shows: connection status, resolution, FPS, codec info, and a preview frame.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


def test_stream(url: str, label: str = "", timeout: float = 10.0) -> dict:
    """Try to connect to a stream and read one frame. Returns diagnostic info."""
    result = {
        "url": url,
        "label": label,
        "status": "failed",
        "resolution": None,
        "fps": None,
        "codec": None,
        "latency_ms": None,
        "error": None,
    }

    t0 = time.monotonic()
    try:
        cap = cv2.VideoCapture(url)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    elapsed = (time.monotonic() - t0) * 1000
                    h, w = frame.shape[:2]
                    result.update({
                        "status": "connected",
                        "resolution": f"{w}x{h}",
                        "fps": cap.get(cv2.CAP_PROP_FPS) or "unknown",
                        "codec": _get_codec(cap),
                        "latency_ms": round(elapsed, 1),
                    })
                    cap.release()
                    return result
            time.sleep(0.1)

        cap.release()
        result["error"] = f"Timeout after {timeout}s — stream opened but no frames received"

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _get_codec(cap: cv2.VideoCapture) -> str:
    """Try to extract the video codec FourCC."""
    try:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc > 0:
            return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    except Exception:
        pass
    return "unknown"


def test_mediamtx_api(host: str = "localhost", port: int = 9997) -> None:
    """Check MediaMTX API for active streams."""
    import urllib.request

    url = f"http://{host}:{port}/v3/paths/list"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            items = data.get("items", [])
            print(f"\nMediaMTX Status ({host}:{port})")
            print(f"   Active paths: {len(items)}")
            for item in items:
                name = item.get("name", "?")
                ready = item.get("ready", False)
                source_type = item.get("source", {}).get("type", "?")
                readers = len(item.get("readers", []))
                status = "[READY]" if ready else "[NOT READY]"
                print(f"   {status} /{name}  (source: {source_type}, readers: {readers})")
    except Exception as exc:
        print(f"\n[FAIL] Cannot reach MediaMTX API at {host}:{port}: {exc}")
        print("  Make sure MediaMTX is running:")
        print("  docker compose -f tools/docker-compose.streams.yaml up -d")


def test_phone(ip: str) -> None:
    """Test common phone camera RTSP URLs."""
    print(f"\nTesting phone camera at {ip}...")

    # Common phone RTSP app URLs
    candidates = [
        (f"rtsp://{ip}:8080/h264",       "IP Webcam (Android) — H.264"),
        (f"rtsp://{ip}:8080/h264_opus",   "IP Webcam (Android) — H.264+audio"),
        (f"rtsp://{ip}:8554/live",        "RTSP Camera Server (iOS)"),
        (f"rtsp://{ip}:554/stream1",      "Generic RTSP"),
        (f"http://{ip}:8080/video",       "IP Webcam (Android) — MJPEG"),
    ]

    for url, label in candidates:
        result = test_stream(url, label, timeout=5.0)
        status = "[OK]" if result["status"] == "connected" else "[FAIL]"
        if result["status"] == "connected":
            print(f"  {status} {label}")
            print(f"    URL: {url}")
            print(f"    Resolution: {result['resolution']}")
            print(f"    FPS: {result['fps']}")
            print(f"    Latency: {result['latency_ms']}ms")
            return
        else:
            print(f"  {status} {label} — {result.get('error', 'timeout')}")

    print(f"\n  No working stream found at {ip}")
    print("  Make sure the camera app is running and streaming.")


def load_cameras_from_config(config_dir: str = "config") -> list[dict]:
    """Load camera entries from cameras.yaml."""
    import yaml

    path = Path(config_dir) / "cameras.yaml"
    if not path.exists():
        print(f"[FAIL] Config not found: {path}")
        return []

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return data.get("cameras", [])


def main():
    parser = argparse.ArgumentParser(description="Test camera stream connectivity")
    parser.add_argument("url", nargs="?", help="Single URL to test")
    parser.add_argument("--phone", metavar="IP", help="Test phone camera at IP address")
    parser.add_argument("--mediamtx", action="store_true", help="Check MediaMTX API status")
    parser.add_argument("--config", default="config", help="Config directory (default: config)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout in seconds")
    args = parser.parse_args()

    if args.mediamtx:
        test_mediamtx_api()
        return

    if args.phone:
        test_phone(args.phone)
        return

    if args.url:
        # Test a single URL
        print(f"\nTesting: {args.url}")
        result = test_stream(args.url, timeout=args.timeout)
        if result["status"] == "connected":
            print(f"  [OK] Connected!")
            print(f"    Resolution: {result['resolution']}")
            print(f"    FPS:        {result['fps']}")
            print(f"    Codec:      {result['codec']}")
            print(f"    Latency:    {result['latency_ms']}ms")
        else:
            print(f"  [FAIL] Failed: {result.get('error', 'unknown')}")
        return

    # Test all cameras from config
    cameras = load_cameras_from_config(args.config)
    if not cameras:
        print("No cameras found. Provide a URL or check config/cameras.yaml")
        return

    print(f"\nTesting {len(cameras)} camera(s) from {args.config}/cameras.yaml\n")
    print(f"{'Camera':<15} {'Status':<12} {'Resolution':<12} {'FPS':<8} {'Latency':<10} {'Source'}")
    print("-" * 90)

    for cam in cameras:
        cam_id = cam.get("camera_id", "?")
        source = cam.get("source", "?")
        enabled = cam.get("enabled", True)

        if not enabled:
            print(f"{cam_id:<15} {'DISABLED':<12} {'-':<12} {'-':<8} {'-':<10} {source}")
            continue

        result = test_stream(source, cam_id, timeout=args.timeout)
        status = "[OK]" if result["status"] == "connected" else "[FAIL]"
        res = result.get("resolution") or "-"
        fps = str(result.get("fps") or "-")
        lat = f"{result['latency_ms']}ms" if result.get("latency_ms") else "-"

        print(f"{cam_id:<15} {status:<12} {res:<12} {fps:<8} {lat:<10} {source}")

        if result["status"] != "connected" and result.get("error"):
            print(f"{'':>15}  |- {result['error'][:60]}")

    print()


if __name__ == "__main__":
    main()
