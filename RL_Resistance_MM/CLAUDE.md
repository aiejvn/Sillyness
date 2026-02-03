# RL Resistance MM — Project Memory Bank

## Project Overview
Reinforcement learning project for the game "Resistance" (Monster Madness / RE Resistance-style asymmetric multiplayer). The goal is to train a model to play the game by capturing gameplay data, labelling it, and using it for RL training.

## Architecture
- **input_capture/**: Python scripts for recording gameplay sessions (screen frames, audio, keyboard/mouse input). Uses `main.py` as the controller, with separate modules for screen, audio, and input capture. Frames saved as `frame_NNNNNN.jpg` in a `screens/` subdirectory per session.
- **input_capture/re_resistance_captures/**: Captured session data. Sessions include `streamrolled`, `won_in_area2`, `high-rankers_MM_lost`, `survivors_threw_won_at_end`.
- **data_labelling/**: Extractors that process captured frames to produce structured labels for RL training.

## Capture Format
- Frames: `frame_NNNNNN.jpg` (sequential, zero-padded 6 digits) in `<session>/screens/`
- Raw frame data (JSON with base64 image + audio + input): `<session>/frames/raw/frame_NNNNNN.json`
- Resolution: 1920x1080, 60 FPS capture target

## Key Game HUD Elements
- **Main timer**: "MM SS" format at top-center of screen (e.g. "03 52")
- **Time burn/gain popup**: Appears to the right of the main timer. Shows "-Xs Sec." (red, time burned — good for the model) or "+Xs Sec." (time gained — bad for the model). OCR region defined at pixel coords **(1146, 68, 179x81)**.

## Dependencies
- pytesseract + Tesseract-OCR (system install)
- Pillow, opencv-python, numpy

---

## Session Log

### 2026-02-02 — Time Burn Extraction Implementation

**Discussion:**
- Clarified that the goal is to OCR the time burn/gain popup (e.g. "-15 Sec."), not track the main countdown timer directly.
- Confirmed frame naming is sequential (`frame_000000.jpg`, `frame_000001.jpg`, ...).
- Verified OCR region coordinates by viewing `won_in_area2/screens/frame_007273.jpg` which shows a "-15 Sec." popup.
- Tesseract is installed on the system.

**Implemented (data_labelling/):**
1. **schemas.py** — `RegionConfig` dataclass (x, y, width, height with a `box` property for PIL cropping), `TimeBurnEvent` dataclass (frame_number, delta, raw_text), and `TIME_BURN_POPUP_REGION` constant set to (1146, 68, 179x81).
2. **time_burn.py** — Core extraction module:
   - `crop_time_region()` — crops the popup region from a frame using PIL.
   - `ocr_time_value()` — preprocesses (grayscale, binary threshold at 180) and runs Tesseract with `--psm 7` and digit+sign whitelist. Returns raw text.
   - `parse_delta()` — regex parses a signed integer from OCR text.
   - `extract_time_burn()` — iterates all frame JPEGs in order, OCRs each, deduplicates (only emits an event when the popup value changes or appears/disappears), returns list of `TimeBurnEvent`.
3. **run_time_burn.py** — CLI entry point. Takes a `frames_dir` argument, runs extraction, prints summary, saves results to JSON.

**Not yet done:**
- OCR has not been verified end-to-end against real frames yet. The threshold value (180) and Tesseract config may need tuning depending on actual popup colours/contrast.
- May need to handle edge cases: partial popups during fade-in/fade-out, OCR misreads on stylized game font.
- Plan mentions possible fallback to template matching if Tesseract struggles with the game font.
