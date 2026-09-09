"""
dfx.perturbations — robustness suite (reviewer objection: compression/perturbation).

Runs on EXISTING checkpoints. No retraining. Apply to test frames only.

Design follows DeeperForensics-1.0's protocol shape: a fixed set of distortion
types at 5 discrete intensity levels, so results form a clean table/curve.
All functions: uint8 HWC BGR (OpenCV convention) in, same out.
"""
from __future__ import annotations
import cv2
import numpy as np

LEVELS = [1, 2, 3, 4, 5]


# ---------------- individual distortions ----------------
def jpeg_compress(img, level):
    q = {1: 75, 2: 60, 3: 45, 4: 30, 5: 15}[level]
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img.copy()


def gaussian_blur(img, level):
    k = {1: 3, 2: 5, 3: 7, 4: 9, 5: 13}[level]
    return cv2.GaussianBlur(img, (k, k), 0)


def gaussian_noise(img, level, seed=None):
    sigma = {1: 3, 2: 6, 3: 10, 4: 15, 5: 22}[level]
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def downscale(img, level):
    """Resolution loss: shrink then restore to original size."""
    f = {1: 0.85, 2: 0.7, 3: 0.55, 4: 0.4, 5: 0.25}[level]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(int(w * f), 8), max(int(h * f), 8)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def change_contrast(img, level):
    a = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.55, 5: 0.4}[level]
    return np.clip(img.astype(np.float32) * a + 128 * (1 - a), 0, 255).astype(np.uint8)


def change_saturation(img, level):
    s = {1: 0.8, 2: 0.6, 3: 0.45, 4: 0.3, 5: 0.15}[level]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= s
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def local_block(img, level, seed=None):
    """Occlusion by random black blocks (packet-loss / sticker analogue)."""
    n = {1: 1, 2: 2, 3: 4, 4: 6, 5: 9}[level]
    rng = np.random.default_rng(seed)
    out = img.copy(); h, w = img.shape[:2]
    bs = max(min(h, w) // 10, 4)
    for _ in range(n):
        y = rng.integers(0, max(h - bs, 1)); x = rng.integers(0, max(w - bs, 1))
        out[y:y + bs, x:x + bs] = 0
    return out


PERTURBATIONS = {
    "jpeg": jpeg_compress,
    "blur": gaussian_blur,
    "noise": gaussian_noise,
    "downscale": downscale,
    "contrast": change_contrast,
    "saturation": change_saturation,
    "block": local_block,
}


def apply_perturbation(img, name, level, seed=None):
    if name in ("clean", None):
        return img.copy()
    fn = PERTURBATIONS[name]
    try:
        return fn(img, level, seed=seed)
    except TypeError:
        return fn(img, level)


def perturb_clip(frames, name, level, seed=None):
    """Apply the SAME distortion consistently across every frame of a clip.

    Important: use one fixed seed per clip. Re-randomising per frame injects
    artificial temporal noise and unfairly penalises temporal models.
    """
    return [apply_perturbation(f, name, level, seed=seed) for f in frames]


def robustness_grid(names=None, levels=None, include_clean=True):
    """Enumerate the (name, level) cells of the robustness table."""
    names = names or list(PERTURBATIONS.keys())
    levels = levels or LEVELS
    grid = [("clean", 0)] if include_clean else []
    grid += [(n, l) for n in names for l in levels]
    return grid
