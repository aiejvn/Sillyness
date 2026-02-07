"""CLI entry point for time-burn extraction.

Usage:
    python run_time_burn.py <session_screens_dir> [--output results.json]
    python run_time_burn.py <examples_dir> --validate

Example:
    python run_time_burn.py ../input_capture/re_resistance_captures/won_in_area2/screens -o ../input_capture/re_resistance_captures/won_in_area2/
    python run_time_burn.py examples --validate
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict

from PIL import Image

from schemas import TIME_BURN_POPUP_REGION
from time_burn import crop_time_region, extract_time_burn, ocr_time_value, parse_delta


def run_validation(frames_dir: str):
    """Validate OCR against a map.csv file in the given directory."""
    map_path = os.path.join(frames_dir, "map.csv")
    if not os.path.exists(map_path):
        print(f"Error: map.csv not found in {frames_dir}")
        return

    correct = 0
    wrong = 0
    errors = []

    with open(map_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            expected = int(row["expected_value"])

            img_path = os.path.join(frames_dir, name)
            if not os.path.exists(img_path):
                print(f"  Warning: {name} not found, skipping")
                continue

            image = Image.open(img_path)
            cropped = crop_time_region(image, TIME_BURN_POPUP_REGION)
            raw_text, sign = ocr_time_value(cropped)
            delta = parse_delta(raw_text, sign)

            if delta == expected:
                correct += 1
                status = "✓"
            else:
                wrong += 1
                status = "✗"
                errors.append((name, expected, delta, raw_text, sign))

            print(f"  {status} {name}: expected={expected:+d}, got={delta} (raw='{raw_text}', sign={sign})")

    print(f"\n{'='*50}")
    print(f"Results: {correct} correct, {wrong} wrong out of {correct + wrong} total")
    print(f"Accuracy: {100 * correct / (correct + wrong):.1f}%")

    if errors:
        print(f"\nErrors:")
        for name, expected, got, raw, sign in errors:
            print(f"  {name}: expected {expected:+d}, got {got} (raw='{raw}', sign={sign})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract time-burn events from captured game frames."
    )
    parser.add_argument(
        "frames_dir",
        help="Path to the session screens directory containing frame_NNNNNN.jpg files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save results as JSON. Defaults to <frames_dir>/time_burn_events.json.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation mode: compare OCR results against map.csv in the frames_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.validate:
        run_validation(args.frames_dir)
        return

    events = extract_time_burn(args.frames_dir)

    # Print summary
    print(f"\nFound {len(events)} time-burn events:\n")
    for ev in events:
        sign = "+" if ev.delta > 0 else ""
        print(f"  frame {ev.frame_number:06d}:  {sign}{ev.delta}s  (raw: '{ev.raw_text}')")

    # Save to JSON
    output_path = args.output or f"{args.frames_dir}/time_burn_events.json"
    with open(output_path, "w") as f:
        json.dump([asdict(ev) for ev in events], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
