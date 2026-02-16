import os
import glob
import logging

import cv2
import numpy as np
from PIL import Image

from schemas import RegionConfig, CameraStatusReading, CAMERA_ICON_REGION

logger = logging.getLogger(__name__)

# Exact disabled-camera red in RGB
DISABLED_RED = (174, 17, 9)
DISABLED_RED_TOLERANCE = 30  # per-channel Euclidean distance threshold


def crop_region(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop a region from a full-resolution frame."""
    return image.crop(region.box)


def classify_camera_status(
    cropped_camera_icon: Image.Image,
    white_threshold: int = 180,
    white_sat_ceil: int = 50,
) -> dict:
    """Classify camera status from the camera icon region.

    Red detection uses exact RGB color matching against DISABLED_RED (174, 17, 9).
    White detection uses HSV (low saturation, high value).

    Args:
        cropped_camera_icon: The cropped camera icon region.
        white_threshold: Minimum value (0-255) for white pixels.
        white_sat_ceil: Maximum saturation (0-255) for white pixels.

    Returns:
        Dict with "red", "white" proportions and "camera_status".
        camera_status is "disabled" (red), "online" (white), or "neutral" (neither).
    """
    img = np.array(cropped_camera_icon)
    total_pixels = img.shape[0] * img.shape[1]

    # Red detection: RGB Euclidean distance from the exact disabled-camera color
    r, g, b = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), img[:, :, 2].astype(np.float32)
    dist_sq = (
        (r - DISABLED_RED[0]) ** 2
        + (g - DISABLED_RED[1]) ** 2
        + (b - DISABLED_RED[2]) ** 2
    )
    red_mask = dist_sq <= DISABLED_RED_TOLERANCE ** 2

    # White detection: low saturation + high brightness in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    white_mask = (s < white_sat_ceil) & (v >= white_threshold)

    counts = {
        "red": int(np.sum(red_mask)),
        "white": int(np.sum(white_mask)),
    }

    # Proportion of total pixels (not just "interesting" pixels)
    props = {k: counts[k] / total_pixels for k in counts}

    # Classification logic:
    # - If enough pixels match disabled red → disabled
    # - If enough white pixels → online (camera icon is small on dark bg, so ~10-20% is typical)
    # - Otherwise → neutral
    if props["red"] > 0.05:
        camera_status = "disabled"
    elif props["white"] >= 0.03:
        camera_status = "online"
    else:
        camera_status = "neutral"

    return {**props, "camera_status": camera_status}


def extract_camera_uptime(
    frames_dir: str,
    camera_region: RegionConfig = CAMERA_ICON_REGION,
) -> list[CameraStatusReading]:
    """Iterate over saved frame JPEGs, classify camera status.

    Returns all camera status readings for all frames (no deduplication).

    Args:
        frames_dir: Path to the directory containing frame_NNNNNN.jpg files.
        camera_region: The camera icon region.

    Returns:
        List of CameraStatusReading (one per frame).
    """
    pattern = os.path.join(frames_dir, "frame_*.jpg")
    frame_paths = sorted(glob.glob(pattern))

    if not frame_paths:
        logger.warning("No frame JPEGs found in %s", frames_dir)
        return []

    logger.info("Processing %d frames from %s", len(frame_paths), frames_dir)

    readings: list[CameraStatusReading] = []

    for i, path in enumerate(frame_paths):
        basename = os.path.basename(path)
        frame_number = int(basename.replace("frame_", "").replace(".jpg", ""))

        image = Image.open(path)
        camera_icon = crop_region(image, camera_region)

        result = classify_camera_status(camera_icon)

        readings.append(CameraStatusReading(
            frame_number=frame_number,
            red=result["red"],
            white=result["white"],
            camera_status=result["camera_status"],
        ))
        logger.info(
            "Frame %06d: camera=%s (r=%.2f w=%.2f)",
            frame_number, result["camera_status"],
            result["red"], result["white"],
        )

        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(frame_paths))

    logger.info("Extraction complete: %d readings", len(readings))
    return readings
