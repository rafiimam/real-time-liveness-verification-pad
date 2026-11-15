"""Integration test for the full liveness verification pipeline.

Tests the local decision API endpoints end-to-end with a sample video.
Run with: python scripts/test_integration.py [--video path/to/test.mp4]
"""

import argparse
import sys
import time

import cv2
import httpx
import numpy as np

LOCAL_API = "http://localhost:8001"
GPU_API = "http://localhost:8000"


def generate_test_video(output_path: str = "test_video.mp4", duration_sec: int = 10, fps: int = 30):
    """Generate a synthetic test video with a colored rectangle (no real face)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    for i in range(duration_sec * fps):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simulate a "face-like" region
        cv2.rectangle(frame, (220, 140), (420, 380), (180, 140, 100), -1)
        cv2.circle(frame, (290, 230), 15, (255, 255, 255), -1)
        cv2.circle(frame, (350, 230), 15, (255, 255, 255), -1)
        cv2.ellipse(frame, (320, 310), (40, 20), 0, 0, 180, (200, 100, 100), 2)
        writer.write(frame)

    writer.release()
    return output_path


def test_gpu_health():
    """Test GPU server health endpoint."""
    print("[1/5] Testing GPU server health...")
    try:
        resp = httpx.get(f"{GPU_API}/health", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        print(f"  OK: {data}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    return True


def test_local_health():
    """Test local API health endpoint."""
    print("[2/5] Testing local API health...")
    try:
        resp = httpx.get(f"{LOCAL_API}/health", timeout=10)
        assert resp.status_code == 200
        print(f"  OK: {resp.json()}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    return True


def test_session_create():
    """Test session creation."""
    print("[3/5] Testing session creation...")
    try:
        resp = httpx.post(f"{LOCAL_API}/session/create", json={
            "account_id": "TEST_ACC_001",
            "transaction_id": "TXN_12345",
            "transaction_amount": 50000.0,
        }, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "nonce" in data
        print(f"  OK: session_id={data['session_id']}")
        return data["session_id"]
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def test_stream_upload(session_id: str, video_path: str):
    """Test video stream upload and processing."""
    print("[4/5] Testing stream upload...")
    try:
        with open(video_path, "rb") as f:
            resp = httpx.post(
                f"{LOCAL_API}/session/{session_id}/stream",
                files={"video": ("test.mp4", f, "video/mp4")},
                timeout=120,
            )
        assert resp.status_code == 200
        data = resp.json()
        print(f"  OK: {data}")
        return data
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def test_result(session_id: str):
    """Test result retrieval."""
    print("[5/5] Testing result retrieval...")
    try:
        resp = httpx.get(f"{LOCAL_API}/session/{session_id}/result", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        print(f"  OK: {data}")
        return data
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def run_threshold_calibration(video_dir: str):
    """Run multiple test videos and report score distributions for threshold tuning."""
    print("\n=== Threshold Calibration ===")
    print(f"Video directory: {video_dir}")

    import os
    results = {"real": [], "spoof": []}

    for category in ["real", "spoof"]:
        cat_dir = os.path.join(video_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"  Skipping {category}/ (not found)")
            continue

        for video_file in os.listdir(cat_dir):
            if not video_file.endswith((".mp4", ".avi")):
                continue

            video_path = os.path.join(cat_dir, video_file)
            session_id = test_session_create()
            if not session_id:
                continue

            result = test_stream_upload(session_id, video_path)
            if result:
                results[category].append({
                    "file": video_file,
                    "spatial": result.get("spatial_liveness_score", 0),
                    "temporal": result.get("temporal_liveness_score", 0),
                    "face_match": result.get("face_match_confidence", 0),
                    "approved": result.get("is_approved", False),
                })

    print("\n--- Score Distribution ---")
    for category, scores in results.items():
        if not scores:
            continue
        spatial_scores = [s["spatial"] for s in scores]
        temporal_scores = [s["temporal"] for s in scores]
        print(f"\n{category.upper()} ({len(scores)} videos):")
        print(f"  Spatial liveness:  mean={np.mean(spatial_scores):.3f}, "
              f"min={np.min(spatial_scores):.3f}, max={np.max(spatial_scores):.3f}")
        print(f"  Temporal liveness: mean={np.mean(temporal_scores):.3f}, "
              f"min={np.min(temporal_scores):.3f}, max={np.max(temporal_scores):.3f}")

    print("\n--- Recommended Thresholds ---")
    print("Adjust in local-api/app/services/orchestrator.py:")
    print("  SPATIAL_LIVENESS_THRESHOLD = 0.5  (spoof score below this = live)")
    print("  TEMPORAL_LIVENESS_THRESHOLD = 0.90 (liveness above this = live)")
    print("  IDENTITY_THRESHOLD = 0.85 (face match confidence)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Path to test video (generates synthetic if not provided)")
    parser.add_argument("--calibrate", default=None, help="Directory with real/ and spoof/ video subdirs")
    args = parser.parse_args()

    if args.calibrate:
        run_threshold_calibration(args.calibrate)
        sys.exit(0)

    print("=== Liveness Verification Integration Test ===\n")

    video_path = args.video
    if not video_path:
        print("No video provided, generating synthetic test video...")
        video_path = generate_test_video()

    gpu_ok = test_gpu_health()
    local_ok = test_local_health()

    if not local_ok:
        print("\nLocal API not running. Start with: cd local-api && uvicorn app.main:app --port 8001")
        sys.exit(1)

    session_id = test_session_create()
    if not session_id:
        sys.exit(1)

    if gpu_ok:
        result = test_stream_upload(session_id, video_path)
        time.sleep(1)
        test_result(session_id)
    else:
        print("\nGPU server not reachable — skipping stream test.")
        print("Start GPU server: ssh to localhost, cd gpu-server && uvicorn app.main:app --port 8000")

    print("\n=== Test Complete ===")
