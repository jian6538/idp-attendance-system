"""
download_weights.py
===================
Downloads the AntiSpoofing_bin_1.5_128.onnx model from
hairymax/Face-AntiSpoofing (GitHub) into the anti_spoof_weights/ folder.

Source: https://github.com/hairymax/Face-AntiSpoofing

Usage:
    python download_weights.py

If the download fails (network issues), use the manual git method:
    git clone https://github.com/hairymax/Face-AntiSpoofing.git /tmp/hairymax_as
    mkdir -p anti_spoof_weights
    cp /tmp/hairymax_as/saved_models/AntiSpoofing_bin_1.5_128.onnx anti_spoof_weights/
    rm -rf /tmp/hairymax_as
"""

from __future__ import annotations

import os
import sys
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anti_spoof_weights")
MODEL_NAME  = "AntiSpoofing_bin_1.5_128.onnx"
DEST_PATH   = os.path.join(WEIGHTS_DIR, MODEL_NAME)

# Direct URL to the ONNX model in the hairymax repo
MODEL_URL = (
    "https://github.com/hairymax/Face-AntiSpoofing/raw/main/saved_models/"
    "AntiSpoofing_bin_1.5_128.onnx"
)

EXPECTED_MIN_KB = 400   # sanity check — file should be at least 400 KB


def _progress(count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        print(f"\r  Downloaded {count * block_size // 1024} KB...", end="")
        return
    pct = min(100, count * block_size * 100 // total_size)
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    print(f"\r  [{bar}] {pct}%", end="", flush=True)


def download() -> bool:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.isfile(DEST_PATH):
        size_kb = os.path.getsize(DEST_PATH) // 1024
        print(f"[✓] Already present: {DEST_PATH}  ({size_kb} KB)")
        return True

    print(f"Downloading {MODEL_NAME} ...")
    print(f"  Source : {MODEL_URL}")
    print(f"  Dest   : {DEST_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, DEST_PATH, _progress)
        print()  # newline after progress bar
    except Exception as e:
        print(f"\n[✗] Download failed: {e}")
        if os.path.isfile(DEST_PATH):
            os.remove(DEST_PATH)
        print()
        print("  Use the manual git method instead:")
        print()
        print("    git clone https://github.com/hairymax/Face-AntiSpoofing.git /tmp/hairymax_as")
        print("    mkdir -p anti_spoof_weights")
        print("    cp /tmp/hairymax_as/saved_models/AntiSpoofing_bin_1.5_128.onnx anti_spoof_weights/")
        print("    rm -rf /tmp/hairymax_as")
        return False

    size_kb = os.path.getsize(DEST_PATH) // 1024
    print(f"[✓] Saved to {DEST_PATH}  ({size_kb} KB)")

    if size_kb < EXPECTED_MIN_KB:
        print("WARNING: file seems too small — may be corrupted. Delete and re-run.")
        return False

    return True


if __name__ == "__main__":
    sys.exit(0 if download() else 1)
