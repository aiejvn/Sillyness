import os
import re
import glob
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

from schemas import RegionConfig, TimeBurnEvent, TIME_BURN_POPUP_REGION

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Py Torch\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

logger = logging.getLogger(__name__)


def crop_time_region(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop the OCR region from a full-resolution frame."""
    return image.crop(region.box)

# TODO: Curate small validation set of time gain/loss images, start testing various methods of OCR

def detect_sign_from_color(cropped_image: Image.Image) -> int:
    """Detect whether the popup indicates time burn (-) or time gain (+) based on background color.

    Red background = time burn (negative)
    Blue background = time gain (positive)

    Returns -1 for burn, +1 for gain, or 0 if undetermined.
    """
    img = np.array(cropped_image)

    # Count red vs blue pixels
    r, b = img[:, :, 0], img[:, :, 2]

    # A pixel is "red" if red channel dominates blue
    red_pixels = np.sum((r > b + 30) & (r > 100))
    # A pixel is "blue" if blue channel dominates red
    blue_pixels = np.sum((b > r + 30) & (b > 100))

    # Red-dominant = time burn (negative), Blue-dominant = time gain (positive)
    if red_pixels > blue_pixels * 1.5:
        return -1
    elif blue_pixels > red_pixels * 1.5:
        return 1
    return 0


def ocr_time_value(
    cropped_image: Image.Image,
    equalization: str | None = None,
    thresholding: str | None = None,
    threshold_value: int = 180,
    scale_factor: int = 3,
    invert: bool = True,
    morph_clean: bool = True,
    debug_path: str | None = None,
) -> tuple[str, int]:
    """Preprocess a cropped popup region and OCR it.

    Args:
        cropped_image: The cropped popup region.
        equalization: Optional equalization method. Options: "clahe", "hist", None (default).
        thresholding: Optional thresholding method. Options: "otsu", "adaptive", None (default=fixed).
        threshold_value: Fixed threshold value when not using otsu/adaptive.
        scale_factor: Scale up image by this factor before OCR (Tesseract works better with larger images).
        invert: If True, invert threshold to get black text on white background.
        morph_clean: If True, apply morphological operations to clean up noise.
        debug_path: If provided, save the preprocessed image to this path.

    Returns a tuple of (raw_ocr_text, sign) where sign is -1, +1, or 0.
    """
    img = np.array(cropped_image)

    # Detect sign from background color before converting to grayscale
    sign = detect_sign_from_color(cropped_image)

    # Use HSV Value channel instead of standard grayscale
    # This captures brightness regardless of color, helping with colored text on colored backgrounds
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = hsv[:, :, 2]  # Value channel = brightness

    # Scale up for better OCR (Tesseract struggles with small text)
    if scale_factor > 1:
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply equalization if specified
    if equalization == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    elif equalization == "hist":
        gray = cv2.equalizeHist(gray)

    # Thresholding — the popup text is bright on a coloured background
    # Use THRESH_BINARY_INV if invert=True to get black text on white background (Tesseract prefers this)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if thresholding == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    elif thresholding == "adaptive":
        adaptive_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, adaptive_type, 11, 2
        )
    else:
        # Default: fixed threshold at user-provided value
        _, thresh = cv2.threshold(gray, threshold_value, 255, thresh_type)

    # Morphological operations to clean up noise
    if morph_clean:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Close small gaps in digits
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Remove small noise specks
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Save debug image if requested
    if debug_path:
        cv2.imwrite(debug_path, thresh)

    # Tesseract config: single text line (PSM 7), digits only
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config).strip()

    return text, sign

# Note: OCR just has issues recognizing the digits.
def parse_delta(raw_text: str, sign: int) -> int | None:
    """Parse the numeric value from OCR text and apply the sign.

    Args:
        raw_text: OCR'd digits (e.g. "15", "10")
        sign: -1 for time burn, +1 for time gain

    Returns the signed integer value or None if unparseable.
    """
    match = re.search(r"(\d+)", raw_text)
    if match and sign != 0:
        return sign * int(match.group(1))
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
        raw_text, sign = ocr_time_value(cropped)

        if not raw_text or sign == 0:
            # No popup visible or can't determine sign — reset so next popup is treated as new
            if prev_delta is not None:
                prev_delta = None
            continue

        delta = parse_delta(raw_text, sign)
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
