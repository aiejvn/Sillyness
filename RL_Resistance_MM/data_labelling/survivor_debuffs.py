import os
import glob
import logging

import cv2
import numpy as np
from PIL import Image

from schemas import RegionConfig, SurvivorStatusReading, SURVIVOR_HEALTH_BAR_REGIONS, SURVIVOR_FULL_ICON_REGIONS

logger = logging.getLogger(__name__)

# HSV hue ranges (OpenCV uses 0-179 for hue)
# Each tuple is (low_hue, high_hue)
HUE_RANGES = {
    "red_low": (0, 5),
    "red_high": (170, 179),
    "yellow": (6, 35),  # Includes orange (10-20) for debuff borders
    "green": (36, 85),
    "purple": (120, 160),
}

# Debug visualization colors (BGR)
DEBUG_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "purple": (255, 0, 128),
}


def crop_region(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop a region from a full-resolution frame."""
    return image.crop(region.box)


def classify_health(
    cropped_health_bar: Image.Image,
    sat_floor: int = 50,
    val_floor: int = 50,
) -> dict:
    """Classify health status from the health bar strip.

    Args:
        cropped_health_bar: The cropped health bar region (narrow vertical strip).
        sat_floor: Minimum saturation (0-255) to count a pixel.
        val_floor: Minimum value/brightness (0-255) to count a pixel.

    Returns:
        Dict with "red", "yellow", "green" proportions and "health_status".
    """
    img = np.array(cropped_health_bar)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    chromatic = (s >= sat_floor) & (v >= val_floor)

    red_mask = chromatic & (
        ((h >= HUE_RANGES["red_low"][0]) & (h <= HUE_RANGES["red_low"][1]))
        | ((h >= HUE_RANGES["red_high"][0]) & (h <= HUE_RANGES["red_high"][1]))
    )
    yellow_mask = chromatic & (h >= HUE_RANGES["yellow"][0]) & (h <= HUE_RANGES["yellow"][1])
    green_mask = chromatic & (h >= HUE_RANGES["green"][0]) & (h <= HUE_RANGES["green"][1])

    counts = {
        "red": int(np.sum(red_mask)),
        "yellow": int(np.sum(yellow_mask)),
        "green": int(np.sum(green_mask)),
    }
    total = sum(counts.values())

    if total == 0:
        return {"red": 0.0, "yellow": 0.0, "green": 0.0, "health_status": "green"}

    props = {k: counts[k] / total for k in counts}
    health_status = max(counts, key=counts.get)

    return {**props, "health_status": health_status}


def classify_infection(
    cropped_full_icon: Image.Image,
    sat_floor: int = 50,
    val_floor: int = 50,
) -> dict:
    """Classify infection level from the full portrait icon.

    Args:
        cropped_full_icon: The cropped full icon region (entire portrait).
        sat_floor: Minimum saturation (0-255) to count a pixel.
        val_floor: Minimum value/brightness (0-255) to count a pixel.

    Returns:
        Dict with "purple" proportion and "infection_level".
    """
    img = np.array(cropped_full_icon)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    chromatic = (s >= sat_floor) & (v >= val_floor)

    purple_mask = chromatic & (h >= HUE_RANGES["purple"][0]) & (h <= HUE_RANGES["purple"][1])
    purple_count = int(np.sum(purple_mask))
    total_chromatic = int(np.sum(chromatic))

    if total_chromatic == 0:
        return {"purple": 0.0, "infection_level": "none"}

    purple_prop = purple_count / total_chromatic

    if purple_prop < 0.05:
        infection_level = "none"
    elif purple_prop < 0.25:
        infection_level = "low"
    elif purple_prop < 0.50:
        infection_level = "medium"
    else:
        infection_level = "high"

    return {"purple": purple_prop, "infection_level": infection_level}


def extract_survivor_debuffs(
    frames_dir: str,
    health_bar_regions: dict[int, RegionConfig] = SURVIVOR_HEALTH_BAR_REGIONS,
    full_icon_regions: dict[int, RegionConfig] = SURVIVOR_FULL_ICON_REGIONS,
) -> list[SurvivorStatusReading]:
    """Iterate over saved frame JPEGs, classify survivor health and infection status.

    Returns all survivor status readings for all frames (no deduplication).
    Each frame produces 4 readings (one per survivor).

    Args:
        frames_dir: Path to the directory containing frame_NNNNNN.jpg files.
        health_bar_regions: Mapping of survivor_id to health bar region (narrow strip).
        full_icon_regions: Mapping of survivor_id to full icon region (for infection).

    Returns:
        List of SurvivorStatusReading (4 per frame).
    """
    pattern = os.path.join(frames_dir, "frame_*.jpg")
    frame_paths = sorted(glob.glob(pattern))

    if not frame_paths:
        logger.warning("No frame JPEGs found in %s", frames_dir)
        return []

    logger.info("Processing %d frames from %s", len(frame_paths), frames_dir)

    readings: list[SurvivorStatusReading] = []

    for i, path in enumerate(frame_paths):
        basename = os.path.basename(path)
        frame_number = int(basename.replace("frame_", "").replace(".jpg", ""))

        image = Image.open(path)

        for sid in sorted(health_bar_regions.keys()):
            # Crop both regions
            health_bar = crop_region(image, health_bar_regions[sid])
            full_icon = crop_region(image, full_icon_regions[sid])

            # Classify separately
            health_result = classify_health(health_bar)
            infection_result = classify_infection(full_icon)

            readings.append(SurvivorStatusReading(
                frame_number=frame_number,
                survivor_id=sid,
                red=health_result["red"],
                yellow=health_result["yellow"],
                green=health_result["green"],
                purple=infection_result["purple"],
                health_status=health_result["health_status"],
                infection_level=infection_result["infection_level"],
            ))
            logger.info(
                "Frame %06d S%d: health=%s infection=%s (r=%.2f y=%.2f g=%.2f p=%.2f)",
                frame_number, sid,
                health_result["health_status"], infection_result["infection_level"],
                health_result["red"], health_result["yellow"], health_result["green"], infection_result["purple"],
            )

        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(frame_paths))

    logger.info("Extraction complete: %d readings (%d frames × 4 survivors)", len(readings), len(frame_paths))
    return readings
