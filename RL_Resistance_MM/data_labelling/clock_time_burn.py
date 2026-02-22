"""Clock-based time burn detection.

Instead of OCR-ing the time burn popup, this module reads the main game clock
("MM SS" at top-center) every frame and detects anomalies in the countdown rate.

At 60 FPS the clock should tick down by 1 second every ~60 frames. Any deviation
from this expected rate indicates:
  - Time burn: clock drops by more than 1 second between ticks
  - Time gain: clock increases (e.g. area clear bonus +60s)

Output is written to separate files from the popup-based time_burn analysis.
"""

import os
import re
import glob
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

from schemas import RegionConfig, ClockReading, ClockTimeBurnEvent, MAIN_CLOCK_REGION

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Py Torch\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

logger = logging.getLogger(__name__)


def crop_region(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop a region from a full-resolution frame."""
    return image.crop(region.box)


def ocr_clock_value(
    cropped_clock: Image.Image,
    scale_factor: int = 3,
    debug_path: str | None = None,
) -> str:
    """OCR the main clock region to extract "MM SS" text.

    The clock shows near-white digits on a translucent black overlay.
    Preprocessing pipeline:
      1. Crop to the bounding box of the translucent overlay
      2. Convert to grayscale
      3. Median filter (3x3) to remove salt-and-pepper noise
      4. Otsu threshold to isolate bright digits
      5. Morphological close to fill small gaps in digits
      6. Scale up, pad, and feed to Tesseract with digit whitelist

    Returns:
        Raw OCR text (e.g. "03 52", "0352", "348").
    """
    img = np.array(cropped_clock)

    # 1. Crop to the bounding box of the translucent overlay.
    #    The overlay is darker than the surrounding game scene, so look for
    #    pixels below a darkness threshold to find its extent.
    gray_full = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark_mask = gray_full < 120  # translucent overlay pixels are darker
    coords = np.argwhere(dark_mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img[y0:y1, x0:x1]

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. Median filter (3×3) to remove noise from the translucent overlay
    gray = cv2.medianBlur(gray, 3)

    # 4. Threshold — Otsu automatically picks the best split for bright digits
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 5. Morphological close to fill small gaps within digit strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Scale up for better OCR
    if scale_factor > 1:
        thresh = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Pad with white border so digits don't touch image edges
    pad = 20
    thresh = cv2.copyMakeBorder(thresh, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    if debug_path:
        cv2.imwrite(debug_path, thresh)

    # Tesseract config: single line (PSM 7), digits and space only
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config).strip()

    return text


def parse_clock_text(raw_text: str) -> int | None:
    """Parse clock OCR text into total seconds.

    Handles formats:
      "03 52" or "0352" → 3*60 + 52 = 232
      "348"            → might be "3:48" = 228 (3-digit: first digit=min, rest=sec)
      "52"             → 52 seconds (under 1 minute)

    Returns total seconds, or None if unparseable.
    """
    # Remove all whitespace
    digits = re.sub(r"\s+", "", raw_text)

    if not digits or not digits.isdigit():
        return None

    if len(digits) == 4:
        # "MMSS" format: e.g. "0352" → 03:52
        minutes = int(digits[:2])
        seconds = int(digits[2:])
    elif len(digits) == 3:
        # "MSS" format: e.g. "348" → 3:48
        minutes = int(digits[0])
        seconds = int(digits[1:])
    elif len(digits) == 2:
        # "SS" format: under 1 minute
        minutes = 0
        seconds = int(digits)
    elif len(digits) == 1:
        # Single digit seconds
        minutes = 0
        seconds = int(digits)
    else:
        return None

    # Sanity check: seconds should be 0-59, minutes 0-99
    if seconds > 59 or minutes > 99:
        return None

    return minutes * 60 + seconds


def extract_clock_readings(
    frames_dir: str,
    region: RegionConfig = MAIN_CLOCK_REGION,
    debug_dir: str | None = None,
) -> list[ClockReading]:
    """OCR the main clock from every frame, return deduplicated readings.

    Only emits a new reading when the clock value changes.
    """
    pattern = os.path.join(frames_dir, "frame_*.jpg")
    frame_paths = sorted(glob.glob(pattern))

    if not frame_paths:
        logger.warning("No frame JPEGs found in %s", frames_dir)
        return []

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    logger.info("Processing %d frames from %s", len(frame_paths), frames_dir)

    readings: list[ClockReading] = []
    prev_seconds = None

    for i, path in enumerate(frame_paths):
        basename = os.path.basename(path)
        frame_number = int(basename.replace("frame_", "").replace(".jpg", ""))

        image = Image.open(path)
        cropped = crop_region(image, region)

        debug_path = None
        if debug_dir:
            stem = basename.replace(".jpg", "")
            debug_path = os.path.join(debug_dir, f"{stem}_clock_thresh.png")

        raw_text = ocr_clock_value(cropped, debug_path=debug_path)

        if not raw_text:
            continue

        seconds = parse_clock_text(raw_text)
        if seconds is None:
            continue

        if seconds != prev_seconds:
            readings.append(ClockReading(
                frame_number=frame_number,
                clock_seconds=seconds,
                raw_text=raw_text,
            ))
            logger.info(
                "Frame %06d: clock = %d:%02d (%ds) raw='%s'",
                frame_number, seconds // 60, seconds % 60, seconds, raw_text,
            )
            prev_seconds = seconds

        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(frame_paths))

    logger.info("Clock extraction complete: %d readings", len(readings))
    return readings


def detect_time_burn_events(
    readings: list[ClockReading],
    fps: int = 60,
    anomaly_threshold: float = 0.5,
) -> list[ClockTimeBurnEvent]:
    """Analyze consecutive clock readings to detect time burn/gain events.

    Uses the frame gap between readings to account for normal clock ticking.
    At 60 FPS, each real second ≈ 60 frames. The clock should decrease by
    1 second per real second. Any faster decrease = burn, any increase = gain.

    For each pair of consecutive readings:
      elapsed_seconds = frame_gap / fps  (real time that passed)
      delta = clock_change              (how much the clock actually changed)
      anomaly = delta + elapsed_seconds  (deviation from expected)

    Examples at 60 FPS:
      60-frame gap, delta=-1  → anomaly= -1+1.0 =  0.0  (normal tick)
      30-frame gap, delta=-1  → anomaly= -1+0.5 = -0.5  (burned 0.5s extra)
       2-frame gap, delta=-15 → anomaly=-15+0.03=-14.97  (burned ~15s)
      60-frame gap, delta=+59 → anomaly=+59+1.0 =+60.0  (gained 60s)

    Args:
        readings: Deduplicated clock readings.
        fps: Capture framerate. Default: 60.
        anomaly_threshold: Minimum |anomaly| to report as an event. Default: 0.5.
    """
    if len(readings) < 2:
        return []

    events: list[ClockTimeBurnEvent] = []

    for i in range(1, len(readings)):
        prev = readings[i - 1]
        curr = readings[i]

        delta = curr.clock_seconds - prev.clock_seconds
        frame_gap = curr.frame_number - prev.frame_number
        elapsed_seconds = frame_gap / fps

        # anomaly: how many seconds the clock deviated from expected
        # expected clock change = -elapsed_seconds (ticks down at 1s/s)
        # anomaly = delta - (-elapsed_seconds) = delta + elapsed_seconds
        anomaly = delta + elapsed_seconds

        if abs(anomaly) >= anomaly_threshold:
            events.append(ClockTimeBurnEvent(
                frame_number=curr.frame_number,
                clock_seconds=curr.clock_seconds,
                delta=delta,
                frame_gap=frame_gap,
                elapsed_seconds=round(elapsed_seconds, 2),
                anomaly=round(anomaly, 2),
            ))
            logger.info(
                "Frame %06d: clock %d→%d (delta=%d, gap=%d frames, %.1fs real, anomaly=%+.1f) %s",
                curr.frame_number,
                prev.clock_seconds, curr.clock_seconds,
                delta, frame_gap, elapsed_seconds, anomaly,
                "BURN" if anomaly < 0 else "GAIN",
            )

    logger.info("Detected %d time burn/gain events", len(events))
    return events


def cleanup_spike_deltas(
    events: list[ClockTimeBurnEvent],
    spike_threshold: int = 70,
) -> list[ClockTimeBurnEvent]:
    """Fix spurious large deltas caused by OCR misreads.

    When |delta| >= spike_threshold, the clock value at that frame is likely
    garbage. Replace the delta with -(t_spike + t_{spike+1}), where t_spike
    is the clock_seconds at the spike frame and t_{spike+1} is the
    clock_seconds at the next event.

    Events at the very end of the list with no successor are left unchanged.
    """
    cleaned = list(events)

    for i, ev in enumerate(cleaned):
        if abs(ev.delta) < spike_threshold:
            continue

        if i + 1 >= len(cleaned):
            logger.warning(
                "Frame %06d: spike delta=%d but no successor event to correct with, skipping",
                ev.frame_number, ev.delta,
            )
            continue

        t_spike = ev.clock_seconds
        t_next = cleaned[i + 1].clock_seconds
        corrected_delta = -(t_spike + t_next)

        logger.info(
            "Frame %06d: correcting spike delta=%d → %d  (t_spike=%d, t_next=%d)",
            ev.frame_number, ev.delta, corrected_delta, t_spike, t_next,
        )

        cleaned[i] = ClockTimeBurnEvent(
            frame_number=ev.frame_number,
            clock_seconds=ev.clock_seconds,
            delta=corrected_delta,
            frame_gap=ev.frame_gap,
            elapsed_seconds=ev.elapsed_seconds,
            anomaly=ev.anomaly,
        )

    return cleaned
