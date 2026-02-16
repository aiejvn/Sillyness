"""CLI entry point for camera uptime extraction.

Usage:
    python run_camera_uptime.py <session_screens_dir> [--output results.json]
    python run_camera_uptime.py <examples_dir> --validate

Example:
    python run_camera_uptime.py ../input_capture/re_resistance_captures/won_in_area2/screens
    python run_camera_uptime.py examples/camera_uptime --validate
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict

from PIL import Image

from schemas import CAMERA_ICON_REGION
from camera_uptime import crop_region, classify_camera_status, extract_camera_uptime

# Best config: python run_camera_uptime.py examples/cameras/ --validate --debug-dir examples/cameras/debug

def run_validation(
    frames_dir: str,
    white_threshold: int = 180,
    white_sat_ceil: int = 50,
    debug_dir: str | None = None,
):
    """Validate camera status classification against a map.csv file in the given directory.

    map.csv columns: name, expected_status
    """
    map_path = os.path.join(frames_dir, "map.csv")
    if not os.path.exists(map_path):
        print(f"Error: map.csv not found in {frames_dir}")
        return

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"White threshold: {white_threshold}, White sat ceil: {white_sat_ceil}")
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
            expected_status = row["expected_status"]

            img_path = os.path.join(frames_dir, name)
            if not os.path.exists(img_path):
                print(f"  Warning: {name} not found, skipping")
                continue

            image = Image.open(img_path)
            camera_icon = crop_region(image, CAMERA_ICON_REGION)

            if debug_dir:
                stem = name.replace(".jpg", "")
                camera_icon.save(os.path.join(debug_dir, f"{stem}_camera_icon.png"))

            result = classify_camera_status(
                camera_icon,
                white_threshold=white_threshold,
                white_sat_ceil=white_sat_ceil,
            )

            match = result["camera_status"] == expected_status

            if match:
                correct += 1
            else:
                wrong += 1
                errors.append((name, expected_status, result))

            status_symbol = "✓" if match else "✗"

            print(f"  {status_symbol} {name}: camera={result['camera_status']} (exp {expected_status}) | "
                  f"r={result['red']:.2f} w={result['white']:.2f}")

    total = correct + wrong

    if total == 0:
        print("No frames to validate.")
        return

    print(f"\n{'='*60}")
    print(f"Correct: {correct}/{total} ({100 * correct / total:.1f}%)")

    if errors:
        print(f"\nErrors:")
        for name, exp, result in errors:
            print(f"  {name}: expected {exp}, got {result['camera_status']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera uptime status from captured game frames."
    )
    parser.add_argument(
        "frames_dir",
        help="Path to the session screens directory containing frame_NNNNNN.jpg files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save results as JSON. Defaults to <frames_dir>/camera_uptime.json.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation mode: compare results against map.csv in the frames_dir.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=180,
        help="Minimum brightness (0-255) for white pixels (active camera). Default: 180.",
    )
    parser.add_argument(
        "--white-sat-ceil",
        type=int,
        default=50,
        help="Maximum saturation (0-255) for white pixels. Default: 50.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Save cropped camera icons to this directory for debugging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.validate:
        run_validation(
            args.frames_dir,
            white_threshold=args.white_threshold,
            white_sat_ceil=args.white_sat_ceil,
            debug_dir=args.debug_dir,
        )
        return

    readings = extract_camera_uptime(args.frames_dir)

    # Print summary
    print(f"\nExtracted {len(readings)} camera status readings:\n")
    for r in readings:
        print(f"  frame {r.frame_number:06d}: camera={r.camera_status} | "
              f"r={r.red:.4f} w={r.white:.4f}")

    # Save to JSON
    output_path = args.output or os.path.join(args.frames_dir, "camera_uptime.json")
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in readings], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
