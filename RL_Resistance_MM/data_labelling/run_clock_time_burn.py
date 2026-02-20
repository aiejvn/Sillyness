"""CLI entry point for clock-based time burn detection.

Reads the main game clock ("MM SS") from every frame and detects time burn/gain
events by comparing consecutive clock values.

Usage:
    python run_clock_time_burn.py <session_screens_dir> [--output results.json]
    python run_clock_time_burn.py <examples_dir> --validate

Example:
    python run_clock_time_burn.py ../input_capture/re_resistance_captures/won_in_area2/screens
    python run_clock_time_burn.py examples/clock --validate --debug-dir examples/clock/debug
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict

from PIL import Image

from schemas import MAIN_CLOCK_REGION
from clock_time_burn import (
    crop_region, ocr_clock_value, parse_clock_text,
    extract_clock_readings, detect_time_burn_events,
)


def run_validation(
    frames_dir: str,
    scale_factor: int = 3,
    threshold_value: int = 180,
    debug_dir: str | None = None,
):
    """Validate clock OCR against a map.csv file.

    map.csv columns: name, expected_seconds
    """
    map_path = os.path.join(frames_dir, "map.csv")
    if not os.path.exists(map_path):
        print(f"Error: map.csv not found in {frames_dir}")
        return

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"Scale: {scale_factor}x, Threshold: {threshold_value}")
    if debug_dir:
        print(f"Debug images: {debug_dir}")
    print()

    correct = 0
    wrong = 0
    errors = []

    with open(map_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            expected_seconds = int(row["expected_seconds"])

            img_path = os.path.join(frames_dir, name)
            if not os.path.exists(img_path):
                print(f"  Warning: {name} not found, skipping")
                continue

            image = Image.open(img_path)
            cropped = crop_region(image, MAIN_CLOCK_REGION)

            debug_path = None
            if debug_dir:
                stem = name.replace(".jpg", "")
                cropped.save(os.path.join(debug_dir, f"{stem}_clock_crop.png"))
                debug_path = os.path.join(debug_dir, f"{stem}_clock_thresh.png")

            raw_text = ocr_clock_value(
                cropped,
                scale_factor=scale_factor,
                threshold_value=threshold_value,
                debug_path=debug_path,
            )
            seconds = parse_clock_text(raw_text)

            match = seconds == expected_seconds

            if match:
                correct += 1
            else:
                wrong += 1
                errors.append((name, expected_seconds, seconds, raw_text))

            status = "✓" if match else "✗"
            display_time = f"{seconds // 60}:{seconds % 60:02d}" if seconds is not None else "None"
            expected_time = f"{expected_seconds // 60}:{expected_seconds % 60:02d}"

            print(f"  {status} {name}: got {display_time} ({seconds}s) "
                  f"exp {expected_time} ({expected_seconds}s) raw='{raw_text}'")

    total = correct + wrong
    if total == 0:
        print("No frames to validate.")
        return

    print(f"\n{'='*60}")
    print(f"Correct: {correct}/{total} ({100 * correct / total:.1f}%)")

    if errors:
        print(f"\nErrors:")
        for name, exp, got, raw in errors:
            print(f"  {name}: expected {exp}s, got {got}s (raw='{raw}')")


def main():
    parser = argparse.ArgumentParser(
        description="Detect time burn/gain events by reading the main game clock.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_clock_time_burn.py ../input_capture/re_resistance_captures/won_in_area2/screens
  python run_clock_time_burn.py examples/clock --validate --debug-dir examples/clock/debug

Output files (separate from popup-based time_burn):
  clock_readings.json       Deduplicated clock readings (frame, seconds, raw_text)
  clock_time_burn_events.json  Detected anomalies (burns/gains with delta and anomaly)
        """
    )
    parser.add_argument(
        "frames_dir",
        help="Path to the session screens directory containing frame_NNNNNN.jpg files.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for output JSONs. Defaults to frames_dir.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation mode: compare OCR against map.csv in frames_dir.",
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=3,
        help="Scale factor for image before OCR. Default: 3.",
    )
    parser.add_argument(
        "--threshold-value", "-tv",
        type=int,
        default=180,
        help="Binary threshold for clock digits. Default: 180.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Capture framerate for frame-gap calculations. Default: 60.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Save cropped/thresholded clock images for debugging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.validate:
        run_validation(
            args.frames_dir,
            scale_factor=args.scale,
            threshold_value=args.threshold_value,
            debug_dir=args.debug_dir,
        )
        return

    # Full extraction
    readings = extract_clock_readings(args.frames_dir)
    events = detect_time_burn_events(readings, fps=args.fps)

    output_dir = args.output_dir or args.frames_dir
    os.makedirs(output_dir, exist_ok=True)

    # Print clock readings summary
    print(f"\nClock readings: {len(readings)} distinct values\n")
    for r in readings:
        m, s = r.clock_seconds // 60, r.clock_seconds % 60
        print(f"  frame {r.frame_number:06d}: {m}:{s:02d} ({r.clock_seconds}s) raw='{r.raw_text}'")

    # Print events
    print(f"\nTime burn/gain events: {len(events)}\n")
    for ev in events:
        m, s = ev.clock_seconds // 60, ev.clock_seconds % 60
        label = "BURN" if ev.anomaly < 0 else "GAIN"
        print(f"  frame {ev.frame_number:06d}: {m}:{s:02d} delta={ev.delta:+d} "
              f"gap={ev.frame_gap}f ({ev.elapsed_seconds:.1f}s real) "
              f"anomaly={ev.anomaly:+.1f} ({label})")

    # Save to JSON (separate filenames from popup-based analysis)
    readings_path = os.path.join(output_dir, "clock_readings.json")
    with open(readings_path, "w") as f:
        json.dump([asdict(r) for r in readings], f, indent=2)
    print(f"\nClock readings saved to {readings_path}")

    events_path = os.path.join(output_dir, "clock_time_burn_events.json")
    with open(events_path, "w") as f:
        json.dump([asdict(ev) for ev in events], f, indent=2)
    print(f"Time burn events saved to {events_path}")


if __name__ == "__main__":
    main()
