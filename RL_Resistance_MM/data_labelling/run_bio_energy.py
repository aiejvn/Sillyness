"""CLI entry point for bio energy extraction.

Usage:
    python run_bio_energy.py <session_screens_dir> [--output results.json]
    python run_bio_energy.py <examples_dir> --validate

Example:
    python run_bio_energy.py ../input_capture/re_resistance_captures/won_in_area2/screens
    python run_bio_energy.py examples --validate
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict

from PIL import Image

from schemas import BIO_ENERGY_REGION
from bio_energy import crop_region, extract_bio_energy, ocr_bio_value, parse_bio_value


def run_validation(
    frames_dir: str,
    equalization: str | None = None,
    thresholding: str | None = None,
    threshold_value: int = 180,
    scale_factor: int = 4,
    invert: bool = True,
    morph_clean: bool = True,
    debug_dir: str | None = None,
):
    """Validate OCR against a map.csv file in the given directory."""
    map_path = os.path.join(frames_dir, "map.csv")
    if not os.path.exists(map_path):
        print(f"Error: map.csv not found in {frames_dir}")
        return

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    thresh_desc = thresholding or f"fixed-{threshold_value}"
    print(f"Equalization: {equalization or 'none'}, Thresholding: {thresh_desc}")
    print(f"Scale: {scale_factor}x, Invert: {invert}, Morph: {morph_clean}")
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
            expected = int(row["expected_value"])

            img_path = os.path.join(frames_dir, name)
            if not os.path.exists(img_path):
                print(f"  Warning: {name} not found, skipping")
                continue

            image = Image.open(img_path)
            cropped = crop_region(image, BIO_ENERGY_REGION)
            debug_path = None
            if debug_dir:
                debug_path = os.path.join(debug_dir, name.replace(".jpg", "_thresh.png"))
            raw_text = ocr_bio_value(
                cropped,
                equalization=equalization,
                thresholding=thresholding,
                threshold_value=threshold_value,
                scale_factor=scale_factor,
                invert=invert,
                morph_clean=morph_clean,
                debug_path=debug_path,
            )
            value = parse_bio_value(raw_text)

            if value == expected:
                correct += 1
                status = "OK"
            else:
                wrong += 1
                status = "FAIL"
                errors.append((name, expected, value, raw_text))

            print(f"  {status} {name}: expected={expected}, got={value} (raw='{raw_text}')")

    total = correct + wrong
    if total == 0:
        print("No frames to validate.")
        return

    print(f"\n{'='*50}")
    print(f"Results: {correct} correct, {wrong} wrong out of {total} total")
    print(f"Accuracy: {100 * correct / total:.1f}%")

    if errors:
        print(f"\nErrors:")
        for name, expected, got, raw in errors:
            print(f"  {name}: expected {expected}, got {got} (raw='{raw}')")


def main():
    parser = argparse.ArgumentParser(
        description="Extract bio energy readings from captured game frames."
    )
    parser.add_argument(
        "frames_dir",
        help="Path to the session screens directory containing frame_NNNNNN.jpg files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save results as JSON. Defaults to <frames_dir>/bio_energy_readings.json.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation mode: compare OCR results against map.csv in the frames_dir.",
    )
    parser.add_argument(
        "--equalization", "-e",
        choices=["clahe", "hist"],
        default=None,
        help="Equalization method for preprocessing: 'clahe' (adaptive), 'hist' (standard histogram).",
    )
    parser.add_argument(
        "--thresholding", "-t",
        choices=["otsu", "adaptive"],
        default=None,
        help="Thresholding method: 'otsu' (automatic), 'adaptive' (Gaussian). Default: fixed threshold.",
    )
    parser.add_argument(
        "--threshold-value", "-tv",
        type=int,
        default=100,
        help="Fixed threshold value (0-255) when not using otsu/adaptive. Default: 100.",
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=4,
        help="Scale factor for image before OCR. Default: 4 (bio region is small: 64x76).",
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Disable threshold inversion. By default, we invert to get black text on white background.",
    )
    parser.add_argument(
        "--no-morph",
        action="store_true",
        help="Disable morphological cleanup operations.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Save preprocessed threshold images to this directory for debugging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.validate:
        run_validation(
            args.frames_dir,
            equalization=args.equalization,
            thresholding=args.thresholding,
            threshold_value=args.threshold_value,
            scale_factor=args.scale,
            invert=not args.no_invert,
            morph_clean=not args.no_morph,
            debug_dir=args.debug_dir,
        )
        return

    readings = extract_bio_energy(args.frames_dir)

    # Print summary
    print(f"\nFound {len(readings)} bio energy changes:\n")
    for r in readings:
        print(f"  frame {r.frame_number:06d}:  bio={r.value}  (raw: '{r.raw_text}')")

    # Save to JSON
    output_path = args.output or os.path.join(args.frames_dir, "bio_energy_readings.json")
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in readings], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
