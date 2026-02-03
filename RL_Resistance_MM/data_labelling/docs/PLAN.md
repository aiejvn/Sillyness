# Plan: Implement Time Burn Extraction

## Goal
Build the first data labelling extractor: read captured frame JPEGs, crop the timer region, OCR it, and track time burn (decrements) across frames.

## Approach

### Observation from Screenshots
The main game timer is displayed as "MM SS" at top-center of the screen (e.g. "04 57"). Sometimes there is a blue popup to its right showing "+x s" or "-x s". The spec defines this time-burn OCR region at **(1146, 68, 179x81)** — this captures the seconds portion of the timer. By tracking when this value decrements between frames, we measure time burn. If it increments, survivors gained time (punishment).

### Files to Create/Modify

1. **`data_labelling/time_burn.py`** — Core extractor module:
   - `crop_time_region(image, region)` — crop the OCR region from a frame
   - `ocr_time_value(cropped_image)` — preprocess (grayscale, threshold, invert) and OCR with Tesseract to get an integer
   - `extract_time_burn(frames_dir)` — iterate over saved frame JPEGs in order, OCR each, compute deltas, deduplicate repeated values across adjacent frames, output a list of time-burn events with timestamps

2. **`data_labelling/schemas.py`** — Add data structures:
   - `TimeBurnEvent` dataclass: `frame_number`, `timestamp`, `old_value`, `new_value`, `delta` (negative = time burned, positive = time gained)
   - `RegionConfig` with the pixel coordinates from the spec

3. **`data_labelling/run_time_burn.py`** — CLI entry point:
   - Point at a session directory (e.g. `re_resistance_captures/streamrolled/`)
   - Run extraction, print results, save to JSON

### OCR Strategy
- Use **pytesseract** (Tesseract OCR) — well-suited for numeric HUD text
- Preprocessing pipeline: crop → grayscale → binary threshold → optional invert (white text on dark background) → pass to Tesseract with `--psm 7` (single line) and digit whitelist
- Fallback: if Tesseract struggles with the game font, we can switch to template matching or a small digit classifier later

### Deduplication
The timer value persists across many frames (60 FPS, timer changes every ~1 second). Only emit a `TimeBurnEvent` when the OCR'd value actually changes from one frame to the next.

### Dependencies
- `pytesseract` + Tesseract-OCR installed on system
- `Pillow` (already in use)
- `opencv-python` (already in use as cv2)

## Verification
1. Run `run_time_burn.py` against the `streamrolled` session (has clear gameplay HUD frames)
2. Confirm OCR reads the timer values correctly from the cropped region
3. Confirm time-burn events are detected when the timer decrements
4. Visual spot-check: open a few frame JPEGs and compare OCR output to what's on screen
