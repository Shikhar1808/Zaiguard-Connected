"""
tools/download_test_videos.py

Downloads free sample surveillance-style videos for multi-camera testing.
Places them in recordings/ with the names expected by mediamtx.yml.

Usage:
    uv run tools/download_test_videos.py

These are short Creative Commons / public domain clips. Replace with your
own footage for realistic testing.
"""

import subprocess
import sys
from pathlib import Path

RECORDINGS_DIR = Path("recordings")

# Free sample videos from Pexels (direct download links, CC0 license)
# These are surveillance-style footage with people walking around.
VIDEOS = {
    "entrance.mp4": "https://www.pexels.com/download/video/3129671/?fps=25.0&h=1080&w=1920",
    "server_room.mp4": "https://www.pexels.com/download/video/854669/?fps=25.0&h=1080&w=1920",
    "parking.mp4": "https://www.pexels.com/download/video/2103099/?fps=25.0&h=720&w=1280",
}

# Alternative: use ffmpeg to generate synthetic test patterns with moving objects
SYNTHETIC_COMMAND = """
ffmpeg -f lavfi -i "testsrc2=size=1920x1080:rate=25" \
       -f lavfi -i "sine=frequency=1000:sample_rate=44100" \
       -t 60 -c:v libx264 -preset fast -pix_fmt yuv420p \
       -c:a aac {output}
""".strip()


def generate_synthetic(name: str, width: int, height: int, duration: int = 60) -> bool:
    """Generate a synthetic test video using ffmpeg (no download needed)."""
    output = RECORDINGS_DIR / name
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate=25",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            print(f"  [OK] Generated {name} ({width}x{height}, {duration}s)")
            return True
        else:
            print(f"  [FAIL] ffmpeg failed for {name}: {result.stderr.decode()[:200]}")
            return False
    except FileNotFoundError:
        print("  [FAIL] ffmpeg not found — install it or download videos manually")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] ffmpeg timed out for {name}")
        return False


def main():
    RECORDINGS_DIR.mkdir(exist_ok=True)

    existing = [f.name for f in RECORDINGS_DIR.glob("*.mp4")]
    needed = ["entrance.mp4", "server_room.mp4", "parking.mp4"]
    missing = [n for n in needed if n not in existing]

    if not missing:
        print("All test videos already exist in recordings/:")
        for n in needed:
            print(f"  [OK] {n}")
        return

    print(f"Missing {len(missing)} test video(s). Generating synthetic test patterns...")
    print("(These use ffmpeg's built-in test source — no download required)\n")

    specs = {
        "entrance.mp4":    (1920, 1080, 60),
        "server_room.mp4": (1920, 1080, 60),
        "parking.mp4":     (1280, 720,  60),
    }

    success = 0
    for name in missing:
        w, h, dur = specs[name]
        if generate_synthetic(name, w, h, dur):
            success += 1

    print(f"\nGenerated {success}/{len(missing)} videos in recordings/")

    if success < len(missing):
        print("\nAlternative: manually place any .mp4 files in recordings/")
        print("and rename them to match mediamtx.yml expectations:")
        for n in missing:
            print(f"  recordings/{n}")
        print("\nOr download sample footage from:")
        print("  https://www.pexels.com/search/videos/surveillance/")
        print("  https://www.pexels.com/search/videos/security%20camera/")
        print("  https://pixabay.com/videos/search/cctv/")


if __name__ == "__main__":
    main()
