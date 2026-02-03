import os
import re
import glob
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

from schemas import RegionConfig, TimeBurnEvent, TIME_BURN_POPUP_REGION

logger = logging.getLogger(__name__)


def crop_time_region(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop the OCR region from a full-resolution frame."""
    return image.crop(region.box)


def ocr_time_value(cropped_image: Image.Image) -> str:
    """Preprocess a cropped popup region and OCR it.

    Returns the raw OCR text (e.g. "-15", "+10", or "" if nothing detected).
    """
    img = np.array(cropped_image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Binary threshold — the popup text is bright on a coloured background
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Tesseract config: single line, allow digits and +/- signs
    config = "--psm 7 -c tessedit_char_whitelist=0123456789+-"
    text = pytesseract.image_to_string(
        thresh, config=config
    ).strip()

    return text


def parse_delta(raw_text: str) -> int | None:
    """Try to parse a signed integer from OCR text like '-15', '+10', etc.

    Returns the integer value or None if unparseable.
    """
    match = re.search(r"([+-]?\d+)", raw_text)
    if match:
        return int(match.group(1))
    return None


def extract_time_burn(
    frames_dir: str,
    region: RegionConfig = TIME_BURN_POPUP_REGION,
) -> list[TimeBurnEvent]:
    """Iterate over saved frame JPEGs, OCR the popup region, and return
    a deduplicated list of time-burn events.

    Args:
        frames_dir: Path to the directory containing frame_NNNNNN.jpg files.
        region: The pixel region to crop for the popup.

    Returns:
        List of TimeBurnEvent for each detected popup change.
    """
    pattern = os.path.join(frames_dir, "frame_*.jpg")
    frame_paths = sorted(glob.glob(pattern))

    if not frame_paths:
        logger.warning("No frame JPEGs found in %s", frames_dir)
        return []

    logger.info("Processing %d frames from %s", len(frame_paths), frames_dir)

    events: list[TimeBurnEvent] = []
    prev_delta = None  # track last seen delta to deduplicate

    for i, path in enumerate(frame_paths):
        # Extract frame number from filename
        basename = os.path.basename(path)
        frame_number = int(basename.replace("frame_", "").replace(".jpg", ""))

        image = Image.open(path)
        cropped = crop_time_region(image, region)
        raw_text = ocr_time_value(cropped)

        if not raw_text:
            # No popup visible — reset so next popup is treated as new
            if prev_delta is not None:
                prev_delta = None
            continue

        delta = parse_delta(raw_text)
        if delta is None:
            continue

        # Deduplicate: only emit when the popup value changes
        if delta != prev_delta:
            events.append(TimeBurnEvent(
                frame_number=frame_number,
                delta=delta,
                raw_text=raw_text,
            ))
            logger.info(
                "Frame %06d: detected popup %s (delta=%d)",
                frame_number, raw_text, delta,
            )
            prev_delta = delta

        # Progress logging
        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(frame_paths))

    logger.info("Extraction complete: %d events found", len(events))
    return events
