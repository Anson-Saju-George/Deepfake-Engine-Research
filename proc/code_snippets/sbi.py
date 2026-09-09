"""
dfx.sbi — Self-Blended Images (Shiohara & Yamasaki, CVPR 2022) style augmentation.

WHY THIS MATTERS FOR YOUR PAPER
-------------------------------
Detectors trained on real/fake pairs learn *generator fingerprints* and collapse
when the generator changes (your cross-dataset problem). SBI instead synthesises
pseudo-fakes from REAL images only, by blending a face with a mildly perturbed
copy of ITSELF. The only thing the model can key on is the *blending boundary* -
a manipulation-agnostic artifact - which is why SBI-class methods post the single
largest cross-dataset jump in the literature.

Practical consequences:
  * needs NO fake data at training time (train on FF++ real videos alone)
  * cheap, frame-level, CPU-side -> fits your single-GPU budget
  * pairs naturally with the "frame-level artifacts dominate" finding

Inputs/outputs: uint8 HWC BGR (OpenCV), aligned face crops recommended.
"""
from __future__ import annotations
import cv2
import numpy as np


# ======================================================================
# 1. Source/Target generator - create statistical inconsistency
# ======================================================================
def _rgb_shift(img, rng, m=20):
    sh = rng.integers(-m, m + 1, 3)
    return np.clip(img.astype(np.int16) + sh[None, None, :], 0, 255).astype(np.uint8)


def _hsv_shift(img, rng, h=10, s=25, v=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-h, h + 1)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.integers(-s, s + 1), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.integers(-v, v + 1), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _brightness_contrast(img, rng, b=0.15, c=0.15):
    alpha = 1.0 + rng.uniform(-c, c)
    beta = 255.0 * rng.uniform(-b, b)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _sharpen(img, rng):
    amt = rng.uniform(0.3, 1.2)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return np.clip(cv2.addWeighted(img, 1 + amt, blur, -amt, 0), 0, 255).astype(np.uint8)


def _resize_jitter(img, rng, lo=0.35, hi=0.9):
    """Frequency-domain inconsistency: downscale then restore."""
    h, w = img.shape[:2]
    f = rng.uniform(lo, hi)
    small = cv2.resize(img, (max(int(w * f), 8), max(int(h * f), 8)), interpolation=cv2.INTER_AREA)
    interp = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
    return cv2.resize(small, (w, h), interpolation=int(interp))


def source_transform(img, rng, p=0.5):
    """Apply a random subset of colour/frequency transforms."""
    out = img.copy()
    for fn in (_rgb_shift, _hsv_shift, _brightness_contrast, _sharpen):
        if rng.random() < p:
            out = fn(out, rng)
    if rng.random() < p:
        out = _resize_jitter(out, rng)
    return out


# ======================================================================
# 2. Mask generator - face-shaped region with a soft, deformed boundary
# ======================================================================
def _hull_mask(shape, landmarks):
    m = np.zeros(shape[:2], np.uint8)
    pts = cv2.convexHull(np.asarray(landmarks, np.int32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(m, pts, 255)
    return m


def _ellipse_mask(shape, rng, scale=(0.62, 0.78)):
    """Fallback when no landmarks: ellipse over the centre of an aligned crop."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    cx = int(w * (0.5 + rng.uniform(-0.03, 0.03)))
    cy = int(h * (0.52 + rng.uniform(-0.03, 0.03)))
    ax = int(w * rng.uniform(*scale) / 2)
    ay = int(h * rng.uniform(scale[0] + 0.05, scale[1] + 0.10) / 2)
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return m


def _elastic(mask, rng, alpha=None, sigma=None):
    """Elastic deformation so the blend boundary is never a clean geometric edge."""
    h, w = mask.shape[:2]
    alpha = alpha if alpha is not None else rng.uniform(8, 26)
    sigma = sigma if sigma is not None else rng.uniform(5, 10)
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def make_blend_mask(shape, rng, landmarks=None, blur_scale=1.0):
    """Return float32 mask in [0,1] with a soft, irregular boundary."""
    m = _hull_mask(shape, landmarks) if landmarks is not None else _ellipse_mask(shape, rng)

    # random shrink/grow so the seam does not always sit on the face contour
    k = int(max(rng.integers(1, 12), 1))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.erode(m, ker) if rng.random() < 0.5 else cv2.dilate(m, ker)

    m = _elastic(m, rng)

    # soft boundary: this IS the artifact the detector must learn
    bk = int(rng.integers(3, 26)) * 2 + 1
    bk = max(int(bk * blur_scale) | 1, 3)
    m = cv2.GaussianBlur(m, (bk, bk), 0).astype(np.float32) / 255.0

    # random global opacity
    return np.clip(m * rng.uniform(0.55, 1.0), 0, 1)


# ======================================================================
# 3. Blend
# ======================================================================
def blend(src, tgt, mask3):
    return np.clip(src.astype(np.float32) * mask3 +
                   tgt.astype(np.float32) * (1.0 - mask3), 0, 255).astype(np.uint8)


def self_blend(img, landmarks=None, rng=None, return_mask=False):
    """Create ONE pseudo-fake from a single real face crop."""
    rng = rng or np.random.default_rng()
    src, tgt = img.copy(), img.copy()
    if rng.random() < 0.5:
        src = source_transform(src, rng)
    else:
        tgt = source_transform(tgt, rng)
    mask = make_blend_mask(img.shape, rng, landmarks)
    out = blend(src, tgt, mask[..., None])
    return (out, mask) if return_mask else out


def self_blend_clip(frames, landmarks_seq=None, seed=None, jitter=0.0):
    """Temporally CONSISTENT SBI across a clip.

    One mask + one transform for the whole clip, so a temporal model cannot
    trivially detect per-frame randomness instead of the blending artifact.
    Set jitter>0 to add small per-frame mask wobble (more realistic, harder).
    """
    rng = np.random.default_rng(seed)
    base_shape = frames[0].shape
    lm0 = landmarks_seq[0] if landmarks_seq is not None else None
    mask = make_blend_mask(base_shape, rng, lm0)
    flip = rng.random() < 0.5
    tstate = rng.integers(0, 2**31 - 1)

    out = []
    for i, f in enumerate(frames):
        r = np.random.default_rng(tstate)          # identical transform each frame
        src, tgt = f.copy(), f.copy()
        if flip: src = source_transform(src, r)
        else:    tgt = source_transform(tgt, r)
        m = mask
        if jitter > 0:
            jr = np.random.default_rng(int(tstate) + i)
            m = np.clip(_elastic(( mask * 255).astype(np.uint8), jr,
                                 alpha=jitter * 10, sigma=8).astype(np.float32) / 255.0, 0, 1)
        out.append(blend(src, tgt, m[..., None]))
    return out


# ======================================================================
# 4. Dataset-side helper
# ======================================================================
class SBIWrapper:
    """Wrap a REAL-ONLY dataset and emit balanced real/pseudo-fake pairs.

    Usage:
        base = YourRealOnlyFrameDataset(...)   # returns (frames, meta)
        ds   = SBIWrapper(base, p_fake=0.5)
        # ds[i] -> (frames, label)  label 1 == pseudo-fake

    Train on FF++ REAL videos only, then evaluate on real fakes.
    """
    def __init__(self, base, p_fake=0.5, seed=42, clip_mode=True, jitter=0.0):
        self.base, self.p_fake, self.clip_mode, self.jitter = base, p_fake, clip_mode, jitter
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        frames = item[0] if isinstance(item, (tuple, list)) else item
        rng = np.random.default_rng(self.seed + i)
        if rng.random() >= self.p_fake:
            return frames, 0
        if self.clip_mode and isinstance(frames, (list, tuple)):
            return self_blend_clip(frames, seed=int(rng.integers(0, 2**31 - 1)),
                                   jitter=self.jitter), 1
        return self_blend(frames, rng=rng), 1
