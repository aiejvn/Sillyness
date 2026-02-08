import os
import re
import glob
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

from schemas import RegionConfig, BioEnergyReading, BIO_ENERGY_REGION

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Py Torch\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

logger = logging.getLogger(__name__)


def crop_region(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop the OCR region from a full-resolution frame."""
    return image.crop(region.box)


def ocr_bio_value(
    cropped_image: Image.Image,
    equalization: str | None = None,
    thresholding: str | None = None,
    threshold_value: int = 100,
    scale_factor: int = 4,
    invert: bool = True,
    morph_clean: bool = True,
    debug_path: str | None = None,
) -> str:
    """Preprocess a cropped bio energy region and OCR it.

    The bio energy counter is red text on a dark HUD background.
    We isolate the red channel before thresholding to avoid noise from
    non-red elements. No sign detection needed — the value is always a
    positive integer.

    Args:
        cropped_image: The cropped bio energy region.
        equalization: Optional equalization method. Options: "clahe", "hist", None.
        thresholding: Optional thresholding method. Options: "otsu", "adaptive", None (fixed).
        threshold_value: Fixed threshold value when not using otsu/adaptive.
        scale_factor: Scale up image before OCR. Default 4 (region is small: 64x76).
        invert: If True, invert threshold to get black text on white background.
        morph_clean: If True, apply morphological operations to clean up noise.
        debug_path: If provided, save the preprocessed image to this path.

    Returns the raw OCR'd text (digits).
    """
    img = np.array(cropped_image)

    # Save raw cropped image before any transformations
    if debug_path:
        raw_path = debug_path.replace("_thresh.png", "_raw.png")
        cv2.imwrite(raw_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Isolate red channel — bio energy text is red (high R, low G/B) on dark background
    red = img[:, :, 0]

    # Scale up for better OCR (region is only 64x76, Tesseract needs larger images)
    if scale_factor > 1:
        red = cv2.resize(red, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply equalization if specified
    if equalization == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        red = clahe.apply(red)
    elif equalization == "hist":
        red = cv2.equalizeHist(red)

    # Thresholding on the red channel
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if thresholding == "otsu":
        _, thresh = cv2.threshold(red, 0, 255, thresh_type + cv2.THRESH_OTSU)
    elif thresholding == "adaptive":
        adaptive_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        thresh = cv2.adaptiveThreshold(
            red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, adaptive_type, 11, 2
        )
    else:
        _, thresh = cv2.threshold(red, threshold_value, 255, thresh_type)

    # Morphological operations to clean up noise
    if morph_clean:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Save debug image if requested
    if debug_path:
        cv2.imwrite(debug_path, thresh)

    # Tesseract config: uniform block (PSM 6), digits only
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config).strip()

    return text


def parse_bio_value(raw_text: str) -> int | None:
    """Parse the numeric bio energy value from OCR text.

    Returns the positive integer value or None if unparseable.
    """
    match = re.search(r"(\d+)", raw_text)
    if match:
        return int(match.group(1))
    return None


def extract_bio_energy(
    frames_dir: str,
    region: RegionConfig = BIO_ENERGY_REGION,
) -> list[BioEnergyReading]:
    """Iterate over saved frame JPEGs, OCR the bio energy region, and return
    a deduplicated list of bio energy readings.

    Only emits a new reading when the value changes (bio energy is discontinuous,
    changing in discrete steps as bio is spent or regenerated).

    Args:
        frames_dir: Path to the directory containing frame_NNNNNN.jpg files.
        region: The pixel region to crop for the bio counter.

    Returns:
        List of BioEnergyReading for each detected value change.
    """
    pattern = os.path.join(frames_dir, "frame_*.jpg")
    frame_paths = sorted(glob.glob(pattern))

    if not frame_paths:
        logger.warning("No frame JPEGs found in %s", frames_dir)
        return []

    logger.info("Processing %d frames from %s", len(frame_paths), frames_dir)

    readings: list[BioEnergyReading] = []
    prev_value = None  # track last seen value to deduplicate

    for i, path in enumerate(frame_paths):
        basename = os.path.basename(path)
        frame_number = int(basename.replace("frame_", "").replace(".jpg", ""))

        image = Image.open(path)
        cropped = crop_region(image, region)
        raw_text = ocr_bio_value(cropped)

        if not raw_text:
            continue

        value = parse_bio_value(raw_text)
        if value is None:
            continue

        # Deduplicate: only emit when the bio energy value changes
        if value != prev_value:
            readings.append(BioEnergyReading(
                frame_number=frame_number,
                value=value,
                raw_text=raw_text,
            ))
            logger.info(
                "Frame %06d: bio energy = %d (raw='%s')",
                frame_number, value, raw_text,
            )
            prev_value = value

        # Progress logging
        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(frame_paths))

    logger.info("Extraction complete: %d readings found", len(readings))
    return readings
