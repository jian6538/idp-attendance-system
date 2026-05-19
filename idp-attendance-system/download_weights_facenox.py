"""
download_weights_facenox.py
===========================
Downloads the facenox MiniFASNetV2-SE quantized ONNX model (~600 KB).

    python download_weights_facenox.py
"""

from __future__ import annotations
import os, sys, urllib.request

SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anti_spoof_weights")
SAVE_PATH = os.path.join(SAVE_DIR, "facenox_quantized.onnx")
URL       = "https://github.com/facenox/face-antispoof-onnx/raw/main/models/best_model_quantized.onnx"

def _progress(count, block_size, total_size):
    pct = min(100, int(count * block_size * 100 / max(total_size, 1)))
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    sys.stdout.write(f"\r  [{bar}] {pct}%"); sys.stdout.flush()

def download() -> bool:
    os.makedirs(SAVE_DIR, exist_ok=True)
    if os.path.isfile(SAVE_PATH):
        print(f"Already downloaded: {SAVE_PATH}"); return True
    print(f"Downloading facenox_quantized.onnx ...\n  {URL}")
    try:
        urllib.request.urlretrieve(URL, SAVE_PATH, _progress); print()
    except Exception as e:
        print(f"\nERROR: {e}")
        if os.path.isfile(SAVE_PATH): os.remove(SAVE_PATH)
        print(f"Manual: download from {URL}\nSave to: {SAVE_PATH}")
        return False
    size_kb = os.path.getsize(SAVE_PATH) / 1024
    print(f"✓ Saved ({size_kb:.0f} KB): {SAVE_PATH}")
    return size_kb > 100

if __name__ == "__main__":
    sys.exit(0 if download() else 1)
