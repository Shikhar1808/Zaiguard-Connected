#!/usr/bin/env python3
"""
ZaiGuard Alert Engine — Comprehensive Traffic Simulator & Load Tester
=====================================================================
Simulates detection events from upstream AI classifiers to exercise and
stress-test all 5 layers of the ZaiGuard Alert Engine.

OPERATING MODES:
  1. realistic     : Background campus traffic with occasional high-confidence spikes.
  2. burst         : High-rate incident burst on 1 camera to demonstrate Layer 3 Redis Dedup.
  3. escalation    : Medium alert followed by a critical spike to demonstrate Layer 3 Escalation.
  4. feedback-loop : Alert creation -> Operator Dismissal -> Subsequent suppression (Layer 4A/4B).
  5. stress        : High-concurrency benchmark measuring RPS, p95/p99 latency, and gate drop ratios.

USAGE:
  python scripts/simulate_traffic.py --mode realistic --duration 15
  python scripts/simulate_traffic.py --mode burst
  python scripts/simulate_traffic.py --mode escalation
  python scripts/simulate_traffic.py --mode feedback-loop
  python scripts/simulate_traffic.py --mode stress --requests 200 --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

# Ensure project root is on sys.path when run from command line
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ANSI Color codes for rich console output
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BOLD = "\033[1m"
COLOR_MAGENTA = "\033[95m"

CAMERAS = [
    {"camera_id": "CAM_ENTRANCE_01", "zone_id": "entrance", "zone_label": "Main Campus Gate"},
    {"camera_id": "CAM_PARKING_02",  "zone_id": "parking",  "zone_label": "North Parking Lot"},
    {"camera_id": "CAM_RESTRICTED_03", "zone_id": "restricted_high", "zone_label": "Server Room Hall"},
    {"camera_id": "CAM_HALLWAY_04",  "zone_id": "public_high", "zone_label": "Student Union Atrium"},
    {"camera_id": "CAM_LOADING_05",  "zone_id": "default", "zone_label": "Logistics Dock B"},
]

PIPELINES = ["fire", "violence", "dog_attack", "trespassing", "accident"]


# Enforce UTF-8 stdout on Windows terminal if possible
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def print_header(title: str) -> None:
    print(f"\n{COLOR_BOLD}{COLOR_CYAN}{'='*70}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_CYAN}  {title}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_CYAN}{'='*70}{COLOR_RESET}\n")


async def get_client(base_url: str, force_in_memory: bool = False) -> httpx.AsyncClient:
    """
    Returns an AsyncClient connected to either a running server or an in-memory ASGI app.
    """
    if not force_in_memory:
        try:
            client = httpx.AsyncClient(base_url=base_url, timeout=5.0)
            resp = await client.get("/health")
            if resp.status_code == 200:
                print(f"{COLOR_GREEN}[+] Connected to live ZaiGuard server at {base_url}{COLOR_RESET}")
                return client
            await client.aclose()
        except Exception:
            pass

    print(f"{COLOR_YELLOW}[*] Live server at {base_url} not detected. Booting in-memory engine...{COLOR_RESET}")
    from main import app
    from config.settings import settings
    from qdrant_client import AsyncQdrantClient
    from layers.suppression.semantic import ensure_qdrant_collections
    try:
        qclient = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        await ensure_qdrant_collections(qclient)
        await qclient.close()
    except Exception:
        pass
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test", timeout=10.0)


async def send_event(
    client: httpx.AsyncClient,
    pipeline: str,
    confidence: float,
    camera: dict[str, str],
    features: dict[str, Any] | None = None,
    silent: bool = False,
) -> dict[str, Any]:
    payload = {
        "pipeline": pipeline,
        "raw_confidence": round(confidence, 4),
        "camera_id": camera["camera_id"],
        "zone_id": camera["zone_id"],
        "zone_label": camera["zone_label"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_features": features or {"simulated": True, "metric": round(random.random() * 10, 2)},
    }

    t0 = time.perf_counter()
    resp = await client.post("/events", json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if resp.status_code != 200:
        if not silent:
            print(f"{COLOR_RED}[-] POST /events failed ({resp.status_code}): {resp.text}{COLOR_RESET}")
        return {"status_code": resp.status_code, "passed": False, "latency_ms": latency_ms}

    data = resp.json()
    alert = data.get("alert")

    if not silent:
        cam_str = f"{camera['camera_id']} ({camera['zone_label']})"
        if alert:
            tier_color = COLOR_RED if alert["tier"] == "CRITICAL" else (COLOR_YELLOW if alert["tier"] == "HIGH" else COLOR_GREEN)
            print(
                f"[{COLOR_BOLD}{tier_color}ALERT CREATED{COLOR_RESET}] {pipeline.upper():11} | "
                f"Conf: {confidence:.2f} -> {alert['effective_conf']:.2f} | "
                f"Tier: {tier_color}{alert['tier']:8}{COLOR_RESET} | "
                f"AlertID: {alert['alert_id'][:8]}... | ({latency_ms:.1f}ms)"
            )
        else:
            print(
                f"[{COLOR_MAGENTA}DROPPED     {COLOR_RESET}] {pipeline.upper():11} | "
                f"Conf: {confidence:.2f} | Cam: {cam_str:<25} | ({latency_ms:.1f}ms)"
            )

    return {"status_code": 200, "passed": alert is not None, "alert": alert, "latency_ms": latency_ms}


async def mode_realistic(client: httpx.AsyncClient, duration_sec: int) -> None:
    print_header("REALISTIC CAMPUS TRAFFIC SIMULATION")
    print(f"Simulating continuous multi-camera traffic for {duration_sec} seconds...\n")

    start_time = time.time()
    passed_count = 0
    dropped_count = 0

    while time.time() - start_time < duration_sec:
        cam = random.choice(CAMERAS)
        pipe = random.choice(PIPELINES)

        # 75% low/noise detections (dropped by Layer 1), 25% true incidents (passed)
        if random.random() < 0.75:
            conf = random.uniform(0.40, 0.67)
        else:
            conf = random.uniform(0.72, 0.96)

        res = await send_event(client, pipe, conf, cam)
        if res.get("passed"):
            passed_count += 1
        else:
            dropped_count += 1

        await asyncio.sleep(random.uniform(0.4, 1.2))

    print(f"\n{COLOR_BOLD}Simulation Complete:{COLOR_RESET}")
    print(f"  Total Events Processed : {passed_count + dropped_count}")
    print(f"  Alerts Created (Passed): {COLOR_GREEN}{passed_count}{COLOR_RESET}")
    print(f"  Events Dropped (Gates) : {COLOR_MAGENTA}{dropped_count}{COLOR_RESET}")


async def mode_burst(client: httpx.AsyncClient) -> None:
    print_header("LAYER 3 BURST DEDUPLICATION DEMO")
    cam = CAMERAS[3]  # Student Union Atrium
    pipe = "violence"

    print(f"Scenario: 10 rapid violence detections occur on {cam['camera_id']} within 2 seconds.")
    print("Expected: First event clears Gate 1 and creates an alert. Remaining 9 are dropped by Layer 3 (Redis Dedup).\n")

    alerts = 0
    drops = 0

    for i in range(1, 11):
        conf = random.uniform(0.92, 0.96)
        print(f"Event #{i:02d} -> ", end="")
        res = await send_event(client, pipe, conf, cam)
        if res.get("passed"):
            alerts += 1
        else:
            drops += 1
        await asyncio.sleep(0.15)

    print(f"\n{COLOR_BOLD}Burst Summary:{COLOR_RESET}")
    print(f"  Sent: 10 | Alerts: {COLOR_GREEN}{alerts}{COLOR_RESET} | Dropped as duplicate: {COLOR_MAGENTA}{drops}{COLOR_RESET}")


async def mode_escalation(client: httpx.AsyncClient) -> None:
    print_header("LAYER 3 ESCALATION DEMO")
    cam = CAMERAS[2]  # Restricted Area
    pipe = "trespassing"

    print(f"Step 1: Person spotted in restricted zone at confidence 0.79.")
    res1 = await send_event(client, pipe, 0.79, cam)

    print(f"\nWaiting 2 seconds during active incident window...\n")
    await asyncio.sleep(2.0)

    print(f"Step 2: Same camera reports trespassing at confidence 0.80 (delta +0.01 < 0.15).")
    print("Expected: Dropped as duplicate.")
    res2 = await send_event(client, pipe, 0.80, cam)

    print(f"\nStep 3: Sudden jump in confidence to 0.96 (delta +0.17 >= escalation threshold 0.15)!")
    print("Expected: Escalation passed through as a new alert despite active window.")
    res3 = await send_event(client, pipe, 0.96, cam)

    print(f"\n{COLOR_BOLD}Escalation Summary:{COLOR_RESET}")
    print(f"  Initial Alert  : {'PASSED' if res1.get('passed') else 'FAILED'}")
    print(f"  Minor Update   : {'DROPPED (Dedup)' if not res2.get('passed') else 'UNEXPECTED PASS'}")
    print(f"  Major Spike    : {'PASSED (Escalated)' if res3.get('passed') else 'FAILED'}")


async def mode_feedback_loop(client: httpx.AsyncClient) -> None:
    print_header("LAYER 4 FALSE-POSITIVE FEEDBACK & SUPPRESSION DEMO")
    cam = {"camera_id": f"CAM_SIM_FB_{uuid.uuid4().hex[:6]}", "zone_id": "default", "zone_label": "Simulated Hall"}
    pipe = "accident"
    features = {"forklift_speed": 4.5, "pallet_angle": 12.0}

    print(f"Step 1: Event enters pipeline and creates initial alert.")
    res1 = await send_event(client, pipe, 0.85, cam, features=features)
    alert = res1.get("alert")
    if not alert:
        print(f"{COLOR_RED}Failed to create initial alert!{COLOR_RESET}")
        return

    alert_id = alert["alert_id"]
    print(f"\nStep 2: Operator reviews dashboard and flags Alert {alert_id[:8]}... as FALSE POSITIVE [Dismiss].")
    fb_resp = await client.post("/feedback", json={"alert_id": alert_id, "action": "dismiss", "permanent": True})
    print(f"  Feedback Response: {fb_resp.status_code} ({fb_resp.json().get('status')})")

    print(f"\nWaiting 1 second for Qdrant / Postgres indexing...\n")
    await asyncio.sleep(1.0)

    print(f"Step 3: Identical false detection occurs again 5 seconds later.")
    print("Expected: Layer 4A/4B intercepts and suppresses the event automatically.")
    res2 = await send_event(client, pipe, 0.85, cam, features=features)

    print(f"\n{COLOR_BOLD}Feedback Loop Summary:{COLOR_RESET}")
    print(f"  Initial Event    : PASSED -> Alert created")
    print(f"  Operator Action  : DISMISSED (Rule created)")
    print(f"  Subsequent Event : {'SUPPRESSED BY LAYER 4' if not res2.get('passed') else 'UNEXPECTED PASS'}")


async def mode_stress(client: httpx.AsyncClient, num_requests: int, concurrency: int) -> None:
    print_header("BENCHMARK & STRESS LOAD TEST")
    print(f"Launching {num_requests} concurrent requests across {concurrency} async workers...\n")

    queue: asyncio.Queue[tuple[str, float, dict[str, str]]] = asyncio.Queue()
    for _ in range(num_requests):
        cam = random.choice(CAMERAS)
        pipe = random.choice(PIPELINES)
        conf = random.uniform(0.50, 0.95)
        queue.put_nowait((pipe, conf, cam))

    latencies: list[float] = []
    passed = 0
    dropped = 0
    errors = 0

    async def worker():
        nonlocal passed, dropped, errors
        while not queue.empty():
            try:
                pipe, conf, cam = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            res = await send_event(client, pipe, conf, cam, silent=True)
            lat = res.get("latency_ms", 0.0)
            latencies.append(lat)
            if res.get("status_code") == 200:
                if res.get("passed"):
                    passed += 1
                else:
                    dropped += 1
            else:
                errors += 1
            queue.task_done()

    t0 = time.perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    total_time = time.perf_counter() - t0

    latencies.sort()
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
    rps = num_requests / total_time if total_time > 0 else 0

    print(f"{COLOR_BOLD}{COLOR_GREEN}LOAD TEST RESULTS:{COLOR_RESET}")
    print(f"  Total Duration   : {total_time:.2f} seconds")
    print(f"  Throughput (RPS) : {COLOR_BOLD}{rps:.1f} req/sec{COLOR_RESET}")
    print(f"  Passed Alerts    : {passed} ({passed/num_requests*100:.1f}%)")
    print(f"  Dropped Events   : {dropped} ({dropped/num_requests*100:.1f}%)")
    print(f"  HTTP Errors      : {errors}")
    print(f"\n{COLOR_BOLD}LATENCY DISTRIBUTION:{COLOR_RESET}")
    print(f"  Average Latency  : {avg_lat:.2f} ms")
    print(f"  Median (p50)     : {p50:.2f} ms")
    print(f"  95th Percentile  : {p95:.2f} ms")
    print(f"  99th Percentile  : {p99:.2f} ms")


async def main() -> None:
    parser = argparse.ArgumentParser(description="ZaiGuard Traffic Simulator & Load Tester")
    parser.add_argument(
        "--mode",
        choices=["realistic", "burst", "escalation", "feedback-loop", "stress"],
        default="realistic",
        help="Simulation mode to execute",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of target server")
    parser.add_argument("--in-memory", action="store_true", help="Force running in-memory ASGI app directly")
    parser.add_argument("--duration", type=int, default=12, help="Duration in seconds for realistic mode")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests for stress mode")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level for stress mode")

    args = parser.parse_args()

    client = await get_client(args.url, force_in_memory=args.in_memory)
    try:
        if args.mode == "realistic":
            await mode_realistic(client, args.duration)
        elif args.mode == "burst":
            await mode_burst(client)
        elif args.mode == "escalation":
            await mode_escalation(client)
        elif args.mode == "feedback-loop":
            await mode_feedback_loop(client)
        elif args.mode == "stress":
            await mode_stress(client, args.requests, args.concurrency)
    finally:
        await client.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    asyncio.run(main())
