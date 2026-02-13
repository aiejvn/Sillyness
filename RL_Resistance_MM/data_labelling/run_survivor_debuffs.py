"""CLI entry point for survivor debuff extraction.

Usage:
    python run_survivor_debuffs.py <session_screens_dir> [--output results.json]
    python run_survivor_debuffs.py <examples_dir> --validate

Example:
    python run_survivor_debuffs.py ../input_capture/re_resistance_captures/won_in_area2/screens
    python run_survivor_debuffs.py examples/survivor_debuffs --validate
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict

from PIL import Image

from schemas import SURVIVOR_HEALTH_BAR_REGIONS, SURVIVOR_FULL_ICON_REGIONS
from survivor_debuffs import crop_region, classify_health, classify_infection, extract_survivor_debuffs

# Best config so far: python run_survivor_debuffs.py examples/debuffs/ --validate --debug-dir examples/debuffs/debug

def run_validation(
    frames_dir: str,
    sat_floor: int = 50,
    val_floor: int = 50,
    debug_dir: str | None = None,
):
    """Validate color classification against a map.csv file in the given directory.

    map.csv columns: name, survivor_id, expected_health, expected_infection
    """
    map_path = os.path.join(frames_dir, "map.csv")
    if not os.path.exists(map_path):
        print(f"Error: map.csv not found in {frames_dir}")
        return

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"Saturation floor: {sat_floor}, Value floor: {val_floor}")
    if debug_dir:
        print(f"Debug images: {debug_dir}")
    print()

    health_correct = 0
    health_wrong = 0
    infection_correct = 0
    infection_wrong = 0
    errors = []

    with open(map_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            sid = int(row["survivor_id"])
            expected_health = row["expected_health"]
            expected_infection = row["expected_infection"]

            img_path = os.path.join(frames_dir, name)
            if not os.path.exists(img_path):
                print(f"  Warning: {name} not found, skipping")
                continue

            image = Image.open(img_path)
            health_bar_region = SURVIVOR_HEALTH_BAR_REGIONS[sid]
            full_icon_region = SURVIVOR_FULL_ICON_REGIONS[sid]

            health_bar = crop_region(image, health_bar_region)
            full_icon = crop_region(image, full_icon_region)

            if debug_dir:
                stem = name.replace(".jpg", "")
                # Save raw crops
                health_bar.save(os.path.join(debug_dir, f"{stem}_s{sid}_health_bar.png"))
                full_icon.save(os.path.join(debug_dir, f"{stem}_s{sid}_full_icon.png"))

            health_result = classify_health(health_bar, sat_floor=sat_floor, val_floor=val_floor)
            infection_result = classify_infection(full_icon, sat_floor=sat_floor, val_floor=val_floor)

            result = {
                **health_result,
                **infection_result,
            }

            health_match = health_result["health_status"] == expected_health
            infection_match = infection_result["infection_level"] == expected_infection

            if health_match:
                health_correct += 1
            else:
                health_wrong += 1

            if infection_match:
                infection_correct += 1
            else:
                infection_wrong += 1

            h_status = "✓" if health_match else "✗"
            i_status = "✓" if infection_match else "✗"

            if not (health_match and infection_match):
                errors.append((name, sid, expected_health, expected_infection, result))

            print(f"  {h_status}{i_status} {name} S{sid}: health={result['health_status']} (exp {expected_health}), "
                  f"infection={result['infection_level']} (exp {expected_infection}) | "
                  f"r={result['red']:.2f} y={result['yellow']:.2f} g={result['green']:.2f} p={result['purple']:.2f}")

    h_total = health_correct + health_wrong
    i_total = infection_correct + infection_wrong

    if h_total == 0:
        print("No frames to validate.")
        return

    print(f"\n{'='*60}")
    print(f"Health: {health_correct}/{h_total} correct ({100 * health_correct / h_total:.1f}%)")
    print(f"Infection: {infection_correct}/{i_total} correct ({100 * infection_correct / i_total:.1f}%)")
    print(f"Overall: {health_correct + infection_correct}/{h_total + i_total} checks passed "
          f"({100 * (health_correct + infection_correct) / (h_total + i_total):.1f}%)")

    if errors:
        print(f"\nErrors:")
        for name, sid, exp_h, exp_i, result in errors:
            print(f"  {name} S{sid}: expected health={exp_h} infection={exp_i}, "
                  f"got health={result['health_status']} infection={result['infection_level']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract survivor debuff status from captured game frames."
    )
    parser.add_argument(
        "frames_dir",
        help="Path to the session screens directory containing frame_NNNNNN.jpg files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save results as JSON. Defaults to <frames_dir>/survivor_debuffs.json.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation mode: compare results against map.csv in the frames_dir.",
    )
    parser.add_argument(
        "--sat-floor",
        type=int,
        default=50,
        help="Minimum saturation (0-255) for a pixel to be counted. Default: 50.",
    )
    parser.add_argument(
        "--val-floor",
        type=int,
        default=50,
        help="Minimum brightness (0-255) for a pixel to be counted. Default: 50.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Save color classification masks to this directory for debugging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.validate:
        run_validation(
            args.frames_dir,
            sat_floor=args.sat_floor,
            val_floor=args.val_floor,
            debug_dir=args.debug_dir,
        )
        return

    readings = extract_survivor_debuffs(args.frames_dir)

    # Print summary
    num_frames = len(readings) // 4 if len(readings) > 0 else 0
    print(f"\nExtracted {len(readings)} readings ({num_frames} frames × 4 survivors):\n")
    for r in readings:
        print(f"  frame {r.frame_number:06d} S{r.survivor_id}: "
              f"health={r.health_status} infection={r.infection_level} | "
              f"r={r.red:.2f} y={r.yellow:.2f} g={r.green:.2f} p={r.purple:.2f}")

    # Save to JSON
    output_path = args.output or os.path.join(args.frames_dir, "survivor_debuffs.json")
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in readings], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
