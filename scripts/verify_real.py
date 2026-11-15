"""CLI: verify a real photo + 10s video against the local decision API.

Usage:
  python scripts/verify_real.py --photo path\\face.jpg --video path\\clip.mp4
  python scripts/verify_real.py --photo face.jpg --video clip.mp4 --account ACC_123
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

DEFAULT_API = "http://127.0.0.1:8001"


def main():
    parser = argparse.ArgumentParser(description="Upload photo + video for liveness verification")
    parser.add_argument("--photo", required=True, help="Path to account-holder reference photo")
    parser.add_argument("--video", required=True, help="Path to ~10 second video")
    parser.add_argument("--account", default="REAL_USER", help="Account ID")
    parser.add_argument("--api", default=DEFAULT_API, help="Local decision API base URL")
    args = parser.parse_args()

    photo = Path(args.photo)
    video = Path(args.video)
    if not photo.exists():
        print(f"Photo not found: {photo}")
        sys.exit(1)
    if not video.exists():
        print(f"Video not found: {video}")
        sys.exit(1)

    print(f"Photo : {photo}")
    print(f"Video : {video}")
    print(f"Account: {args.account}")
    print("Uploading to /verify ...")

    with photo.open("rb") as pf, video.open("rb") as vf:
        files = {
            "photo": (photo.name, pf, "image/jpeg"),
            "video": (video.name, vf, "video/mp4"),
        }
        data = {"account_id": args.account}
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{args.api}/verify", files=files, data=data)

    print(f"HTTP {resp.status_code}")
    try:
        body = resp.json()
        print(json.dumps(body, indent=2, default=str))
    except Exception:
        print(resp.text)
        sys.exit(1)

    if resp.status_code != 200:
        sys.exit(1)

    approved = body.get("is_approved")
    print("\nRESULT:", "APPROVED" if approved else "REJECTED")


if __name__ == "__main__":
    main()
