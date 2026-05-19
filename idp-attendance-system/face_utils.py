"""
face_utils.py
=============
Core face-related helpers built around facenet-pytorch.

- MTCNN performs face detection on each frame.
- InceptionResnetV1 (pretrained on VGGFace2) produces a 512-D embedding per
  detected face.
- identify_face() compares a query embedding against the known database using
  cosine similarity.
- check_liveness() is a thin wrapper around anti_spoof.is_live_face().

Everything runs on CPU so the code works on both a developer laptop and a
Raspberry Pi 5.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from anti_spoof import check_liveness as _check_liveness, load_anti_spoof_model, DEFAULT_LIVENESS_THRESHOLD

# Explicit CPU device — do NOT switch to CUDA; Pi 5 has no GPU.
DEVICE = torch.device("cpu")

# Detection confidence floor. MTCNN returns (boxes, probs); we drop anything
# below this value to avoid triggering the recognizer on garbage crops.
MIN_DETECTION_PROB = 0.92

# Cosine-similarity decision threshold for "this is student X".
DEFAULT_MATCH_THRESHOLD = 0.82


def load_mtcnn() -> MTCNN:
    """Return a ready-to-use MTCNN detector configured for FaceNet input.

    image_size=160 matches InceptionResnetV1's expected input.
    margin=20 keeps a little padding so the crop is not glued to the chin/forehead.
    keep_all=True so we can see ALL faces and then pick the largest one ourselves.
    """
    return MTCNN(
        image_size=160,
        margin=20,
        keep_all=True,
        post_process=True,
        device=DEVICE,
    )


def load_facenet() -> InceptionResnetV1:
    """Return a VGGFace2-pretrained InceptionResnetV1, eval mode, on CPU."""
    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    # Disable gradient tracking globally for this model to save memory/CPU.
    for p in model.parameters():
        p.requires_grad = False
    return model


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """OpenCV gives us BGR, MTCNN wants RGB."""
    return frame_bgr[:, :, ::-1].copy()


def detect_faces(frame_bgr: np.ndarray, mtcnn: MTCNN) -> List[Dict]:
    """Detect faces in an OpenCV BGR frame.

    Returns a list of dicts:
        {
            "box":         [x1, y1, x2, y2]  # ints, clamped to frame
            "prob":        float,
            "face_tensor": torch.Tensor of shape (3, 160, 160)
        }
    Only detections with prob >= MIN_DETECTION_PROB are returned.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    rgb = _bgr_to_rgb(frame_bgr)
    h, w = rgb.shape[:2]

    # MTCNN.detect -> (boxes, probs).  MTCNN.__call__ -> aligned face tensors.
    try:
        boxes, probs = mtcnn.detect(rgb)
        face_tensors = mtcnn(rgb)
    except Exception:
        # MTCNN can occasionally throw on malformed frames; treat as no faces.
        return []

    if boxes is None or probs is None or face_tensors is None:
        return []

    # When keep_all=True and there are N faces, face_tensors is (N, 3, 160, 160).
    if face_tensors.ndim == 3:
        face_tensors = face_tensors.unsqueeze(0)

    results: List[Dict] = []
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob is None or prob < MIN_DETECTION_PROB:
            continue
        if i >= face_tensors.shape[0]:
            continue
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        results.append(
            {
                "box": [x1, y1, x2, y2],
                "prob": float(prob),
                "face_tensor": face_tensors[i].detach(),
            }
        )
    return results


def get_embedding(face_tensor: torch.Tensor, facenet: InceptionResnetV1) -> np.ndarray:
    """Convert a (3, 160, 160) face tensor into a 512-D L2-normalized float32 vector."""
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(DEVICE)
    with torch.no_grad():
        emb = facenet(face_tensor)  # (1, 512)
    emb = emb.cpu().numpy().astype(np.float32).squeeze(0)
    # L2 normalise so cosine-sim == dot product and magnitudes don't skew scores.
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def load_all_embeddings(db_path: str, embeddings_dir: str) -> List[Dict]:
    """Read students.json + each referenced .npy file into memory.

    Gracefully returns [] if the DB or directory is missing / corrupt.
    """
    known: List[Dict] = []

    if not os.path.isfile(db_path):
        return known
    if not os.path.isdir(embeddings_dir):
        return known

    try:
        with open(db_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupted or unreadable DB — start clean rather than crash the app.
        return known

    if not isinstance(records, list):
        return known

    for rec in records:
        try:
            name = rec["name"]
            matrix_number = rec["matrix_number"]
            emb_path = rec.get("embedding_path") or os.path.join(
                embeddings_dir, f"{matrix_number}.npy"
            )
        except (KeyError, TypeError):
            continue

        if not os.path.isfile(emb_path):
            continue

        try:
            emb = np.load(emb_path).astype(np.float32)
        except (OSError, ValueError):
            continue

        # Ensure normalization (in case the stored average wasn't re-normalized).
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n

        known.append(
            {
                "name": name,
                "matrix_number": matrix_number,
                "embedding": emb,
            }
        )

    return known


def identify_face(
    embedding: np.ndarray,
    known_embeddings: List[Dict],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Optional[Dict]:
    """Return the best-matching student dict, or None if below threshold.

    Uses cosine similarity on already-normalized embeddings.
    Returned dict adds a "score" key with the similarity value.
    """
    if embedding is None or len(known_embeddings) == 0:
        return None

    query = embedding.astype(np.float32)
    qn = np.linalg.norm(query)
    if qn > 0:
        query = query / qn

    best = None
    best_score = -1.0

    for rec in known_embeddings:
        ref = rec["embedding"]
        score = float(np.dot(query, ref))
        if score > best_score:
            best_score = score
            best = rec

    if best is None or best_score < threshold:
        return None

    return {
        "name": best["name"],
        "matrix_number": best["matrix_number"],
        "score": best_score,
    }


def check_liveness(
    frame_bgr: np.ndarray,
    box: List[int],
    anti_spoof_model,
    threshold: float = DEFAULT_LIVENESS_THRESHOLD,
) -> tuple:
    """Return (is_live: bool, real_prob: float).

    Wraps anti_spoof.check_liveness() so callers only need to import face_utils.
    If anti_spoof_model is None this returns (True, 1.0) — no liveness gate.
    """
    return _check_liveness(frame_bgr, box, anti_spoof_model, threshold)
