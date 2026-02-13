import os
import glob
import logging

import cv2
import numpy as np
from PIL import Image

from schemas import RegionConfig, CameraStatusReading, CAMERA_ICON_REGION

logger = logging.getLogger(__name__)

# HSV hue ranges (OpenCV uses 0-179 for hue)
HUE_RANGES = {
    "red_low": (0, 5),
    "red_high": (170, 179),
}

# Debug visualization colors (BGR)
DEBUG_COLORS = {
    "red": (0, 0, 255),
    "white": (255, 255, 255),
}


def crop_region(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop a region from a full-resolution frame."""
    return image.crop(region.box)


def classify_camera_status(
    cropped_camera_icon: Image.Image,
    sat_floor: int = 50,
    val_floor: int = 50,
    white_threshold: int = 180,
) -> dict:
    """Classify camera status from the camera icon region.

    Args:
        cropped_camera_icon: The cropped camera icon region.
        sat_floor: Minimum saturation (0-255) to count a chromatic pixel.
        val_floor: Minimum value/brightness (0-255) to count a chromatic pixel.
        white_threshold: Minimum value for white pixels (high V, low S).

    Returns:
        Dict with "red", "white" proportions and "camera_status".
        camera_status is "disabled" (red), "online" (white), or "neutral" (neither).
    """
    img = np.array(cropped_camera_icon)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Chromatic pixels: high saturation + high value
    chromatic = (s >= sat_floor) & (v >= val_floor)

    # White pixels: low saturation + very high value (bright but unsaturated)
    white_mask = (s < sat_floor) & (v >= white_threshold)

    red_mask = chromatic & (
        ((h >= HUE_RANGES["red_low"][0]) & (h <= HUE_RANGES["red_low"][1]))
        | ((h >= HUE_RANGES["red_high"][0]) & (h <= HUE_RANGES["red_high"][1]))
    )

    counts = {
        "red": int(np.sum(red_mask)),
        "white": int(np.sum(white_mask)),
    }

    # Total counted pixels (chromatic + white)
    total_pixels = int(np.sum(chromatic | white_mask))

    if total_pixels == 0:
        # No bright pixels → neutral (not viewing camera, dark icon)
        return {"red": 0.0, "white": 0.0, "camera_status": "neutral"}

    props = {k: counts[k] / total_pixels for k in counts}

    # Classification logic:
    # - If predominantly red → disabled
    # - If predominantly white → online (active camera shows white)
    # - Otherwise → neutral
    if props["red"] > 0.4:
        camera_status = "disabled"
    elif props["white"] > 0.4:
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
