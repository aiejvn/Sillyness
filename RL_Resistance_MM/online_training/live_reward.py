"""live_reward.py — Stateful per-frame reward extractor for online training.

Wraps the existing data_labelling OCR/pixel extractors and computes a scalar
reward for each live frame.  Maintains state across frames for:
  - Time burn deduplication (same popup value = already counted)
  - Bio efficiency delta (needs the previous frame's bio energy reading)
  - EMA normalization (replaces the offline episode-level mean/std)

OCR (Tesseract) is slow (~50-200 ms/call).  To keep the main training loop
at ~10 Hz, OCR extraction is offloaded to a background ThreadPoolExecutor.
The reward returned for frame N is computed from frame N-1's OCR result
(1-frame lag).  Pixel-based extraction (survivor health/infection, camera)
stays synchronous since it runs in <5 ms.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# ── sys.path setup ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "data_labelling"))
sys.path.insert(0, str(_ROOT / "modeling"))

from schemas import (
    TIME_BURN_POPUP_REGION,
    BIO_ENERGY_REGION,
    SURVIVOR_HEALTH_BAR_REGIONS,
    SURVIVOR_FULL_ICON_REGIONS,
    CAMERA_ICON_REGION,
)
from time_burn import crop_time_region, ocr_time_value, parse_delta
from bio_energy import crop_region as crop_bio_region, ocr_bio_value, parse_bio_value
from survivor_debuffs import crop_region as crop_surv_region, classify_health, classify_infection
from camera_uptime import crop_region as crop_cam_region, classify_camera_status
from reward import RewardWeights, compute_reward

logger = logging.getLogger(__name__)


@dataclass
class LiveFrameLabels:
    """All extracted labels for a single live frame."""
    time_burn_delta: int | None      # signed int, or None if no popup
    bio_energy: int | None           # current value, or None if OCR failed
    s1_health: str; s1_infection: str
    s2_health: str; s2_infection: str
    s3_health: str; s3_infection: str
    s4_health: str; s4_infection: str
    camera_status: str               # "online" / "disabled" / "neutral"


def _extract_ocr_labels(pil_image: Image.Image) -> tuple[int | None, int | None]:
    """Run slow OCR extractors.  Intended to run in a background thread.

    Returns:
        (time_burn_delta, bio_energy)  — both may be None on OCR failure.
    """
    # ── Time burn ────────────────────────────────────────────────────────────
    cropped_tb = crop_time_region(pil_image, TIME_BURN_POPUP_REGION)
    raw_text, sign = ocr_time_value(cropped_tb)
    time_burn_delta: int | None = None
    if sign != 0 and raw_text:
        time_burn_delta = parse_delta(raw_text, sign)

    # ── Bio energy ───────────────────────────────────────────────────────────
    cropped_bio = crop_bio_region(pil_image, BIO_ENERGY_REGION)
    raw_bio = ocr_bio_value(cropped_bio)
    bio_energy: int | None = parse_bio_value(raw_bio)

    return time_burn_delta, bio_energy


def _extract_pixel_labels(pil_image: Image.Image) -> tuple[list[tuple[str, str]], str]:
    """Run fast pixel-based extractors (no OCR).

    Returns:
        survivor_statuses: list of (health_status, infection_level) for s1..s4
        camera_status:     "online" / "disabled" / "neutral"
    """
    survivor_statuses: list[tuple[str, str]] = []
    for sid in range(1, 5):
        hb_crop = crop_surv_region(pil_image, SURVIVOR_HEALTH_BAR_REGIONS[sid])
        fi_crop = crop_surv_region(pil_image, SURVIVOR_FULL_ICON_REGIONS[sid])
        health = classify_health(hb_crop)["health_status"]
        infection = classify_infection(fi_crop)["infection_level"]
        survivor_statuses.append((health, infection))

    cam_crop = crop_cam_region(pil_image, CAMERA_ICON_REGION)
    camera_status = classify_camera_status(cam_crop)["camera_status"]

    return survivor_statuses, camera_status


class LiveRewardExtractor:
    """Stateful per-frame reward extractor.

    Usage::

        extractor = LiveRewardExtractor(weights=OnlineConfig().reward_weights)
        for frame in capture_loop():
            reward, labels = extractor.extract(frame)
            ...

    At episode boundaries (game restart / session end) call ``reset()`` to
    clear stateful tracking.
    """

    def __init__(self, weights: RewardWeights | None = None, max_reward: float = 10.0):
        self.weights = weights or RewardWeights()
        self.max_reward = max_reward

        # ── Stateful tracking ─────────────────────────────────────────────
        self._prev_bio_energy: float | None = None
        # Last seen time-burn popup value — re-appearance of the same value
        # means the popup is still showing and should NOT re-fire the reward.
        self._prev_time_burn_delta: int | None = None

        # ── Online EMA normalization ───────────────────────────────────────
        # Replaces offline episode-level mean/std with a running estimate.
        self._ema_alpha: float = 0.01
        self._ema_mean: float = 0.0
        self._ema_sq: float = 0.0

        # ── Background OCR thread ─────────────────────────────────────────
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._ocr_future: Future | None = None   # result of previous frame's OCR
        # The pending OCR result (time_burn_delta, bio_energy) from 1 frame ago
        self._pending_ocr: tuple[int | None, int | None] = (None, None)

    def extract(self, pil_image: Image.Image) -> tuple[float, LiveFrameLabels]:
        """Extract reward and labels for the current frame.

        OCR results lag by 1 frame (background thread).
        Pixel-based results (survivor/camera) are synchronous.

        Returns:
            (reward, labels)
        """
        # ── Collect last frame's OCR result ──────────────────────────────
        if self._ocr_future is not None and self._ocr_future.done():
            try:
                self._pending_ocr = self._ocr_future.result()
            except Exception as e:
                logger.warning(f"OCR thread error: {e}")
                self._pending_ocr = (None, None)
            self._ocr_future = None

        # ── Submit OCR for the CURRENT frame (runs in background) ────────
        if self._ocr_future is None:
            self._ocr_future = self._executor.submit(_extract_ocr_labels, pil_image)

        # ── Synchronous pixel extraction (fast) ───────────────────────────
        survivor_statuses, camera_status = _extract_pixel_labels(pil_image)

        # ── Apply deduplication to time-burn delta ────────────────────────
        raw_tb_delta, bio_energy = self._pending_ocr
        if raw_tb_delta is not None and raw_tb_delta == self._prev_time_burn_delta:
            # Same popup still on screen — don't re-count the reward
            time_burn_delta: int | None = None
        else:
            time_burn_delta = raw_tb_delta
            self._prev_time_burn_delta = raw_tb_delta  # update (None clears it too)

        # ── Build labels dataclass ────────────────────────────────────────
        s = survivor_statuses  # s[0] = (health, infection) for survivor 1, etc.
        labels = LiveFrameLabels(
            time_burn_delta=time_burn_delta,
            bio_energy=bio_energy,
            s1_health=s[0][0], s1_infection=s[0][1],
            s2_health=s[1][0], s2_infection=s[1][1],
            s3_health=s[2][0], s3_infection=s[2][1],
            s4_health=s[3][0], s4_infection=s[3][1],
            camera_status=camera_status,
        )

        # ── Compute scalar reward ─────────────────────────────────────────
        row = {
            "time_burn_delta": labels.time_burn_delta,
            "bio_energy":      labels.bio_energy,
            "s1_health":       labels.s1_health,   "s1_infection": labels.s1_infection,
            "s2_health":       labels.s2_health,   "s2_infection": labels.s2_infection,
            "s3_health":       labels.s3_health,   "s3_infection": labels.s3_infection,
            "s4_health":       labels.s4_health,   "s4_infection": labels.s4_infection,
            "camera_status":   labels.camera_status,
        }
        prev_row = {"bio_energy": self._prev_bio_energy} if self._prev_bio_energy is not None else None

        raw_reward = compute_reward(row, prev_row=prev_row, weights=self.weights, apply_relu=True)

        # ── Update state for next frame ───────────────────────────────────
        if bio_energy is not None:
            self._prev_bio_energy = float(bio_energy)

        # ── EMA normalization + clip ──────────────────────────────────────
        reward = self._normalize(raw_reward)
        reward = float(np.clip(reward, -self.max_reward, self.max_reward))

        return reward, labels

    def _normalize(self, reward: float) -> float:
        self._ema_mean = (1 - self._ema_alpha) * self._ema_mean + self._ema_alpha * reward
        self._ema_sq   = (1 - self._ema_alpha) * self._ema_sq   + self._ema_alpha * reward ** 2
        variance = max(self._ema_sq - self._ema_mean ** 2, 1e-8)
        return (reward - self._ema_mean) / (variance ** 0.5)

    def reset(self) -> None:
        """Clear stateful tracking.  Call at episode boundaries."""
        self._prev_bio_energy = None
        self._prev_time_burn_delta = None
        self._pending_ocr = (None, None)
        # Drain pending OCR future — don't wait on it
        if self._ocr_future is not None and not self._ocr_future.done():
            self._ocr_future.cancel()
        self._ocr_future = None

    def shutdown(self) -> None:
        """Cleanly shut down the background OCR thread."""
        self._executor.shutdown(wait=False)
