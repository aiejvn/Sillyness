"""Unified data labelling pipeline.

Runs all extraction tasks (time burn, bio energy, survivor debuffs, camera uptime)
over a captured game session directory.

Usage:
    python run_labelling.py <capture_dir>
    python run_labelling.py ../input_capture/re_resistance_captures/won_in_area2

Args:
    capture_dir: Path to a capture session directory containing a 'screens/' subdirectory
                 with frame_NNNNNN.jpg files.

Output:
    All results are saved to the capture_dir (or --output dir):
    - time_burn_events.json
    - bio_energy.json
    - survivor_debuffs.json
    - camera_uptime.json
    - labels.csv (combined per-frame output)
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict

from time_burn import extract_time_burn
from bio_energy import extract_bio_energy
from survivor_debuffs import extract_survivor_debuffs
from camera_uptime import extract_camera_uptime


CSV_COLUMNS = [
    "frame",
    "time_burn_delta",
    "bio_energy",
    "s1_health", "s1_infection",
    "s2_health", "s2_infection",
    "s3_health", "s3_infection",
    "s4_health", "s4_infection",
    "camera_status",
]


def write_combined_csv(path, time_burn_events, bio_readings, survivor_readings, camera_readings):
    """Merge all extraction results into a single per-frame CSV."""
    time_burn_by_frame = {ev.frame_number: ev.delta for ev in time_burn_events}
    bio_by_frame = {r.frame_number: r.value for r in bio_readings}

    survivor_by_frame = {}
    for r in survivor_readings:
        survivor_by_frame.setdefault(r.frame_number, {})[r.survivor_id] = r

    camera_by_frame = {r.frame_number: r.camera_status for r in camera_readings}

    # All frame numbers from per-frame sources (survivor + camera cover every frame)
    all_frames = sorted(
        set(survivor_by_frame.keys()) | set(camera_by_frame.keys())
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for frame in all_frames:
            row = {"frame": frame}

            if frame in time_burn_by_frame:
                row["time_burn_delta"] = time_burn_by_frame[frame]
            if frame in bio_by_frame:
                row["bio_energy"] = bio_by_frame[frame]

            survivors = survivor_by_frame.get(frame, {})
            for sid in range(1, 5):
                sr = survivors.get(sid)
                if sr:
                    row[f"s{sid}_health"] = sr.health_status
                    row[f"s{sid}_infection"] = sr.infection_level

            if frame in camera_by_frame:
                row["camera_status"] = camera_by_frame[frame]

            writer.writerow(row)

    return len(all_frames)


def run_labelling(capture_dir, output_dir=None):
    """Run all data labelling tasks on a capture session."""
    screens_dir = os.path.join(capture_dir, "screens")
    if not os.path.exists(screens_dir):
        raise FileNotFoundError(f"Screens directory not found: {screens_dir}")

    if output_dir is None:
        output_dir = capture_dir

    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Starting data labelling pipeline for: %s", capture_dir)
    logger.info("Output directory: %s", output_dir)

    # 1. Time Burn  (best config: otsu thresholding, no morph — baked into extract defaults)
    logger.info("=" * 60)
    logger.info("Extracting time burn events...")
    time_burn_events = extract_time_burn(screens_dir)
    time_burn_path = os.path.join(output_dir, "time_burn_events.json")
    with open(time_burn_path, "w") as f:
        json.dump([asdict(ev) for ev in time_burn_events], f, indent=2)
    logger.info("Time burn: %d events -> %s", len(time_burn_events), time_burn_path)

    # 2. Bio Energy  (best config: red channel isolation, threshold=100, PSM 8)
    logger.info("=" * 60)
    logger.info("Extracting bio energy readings...")
    bio_readings = extract_bio_energy(screens_dir)
    bio_path = os.path.join(output_dir, "bio_energy.json")
    with open(bio_path, "w") as f:
        json.dump([asdict(r) for r in bio_readings], f, indent=2)
    logger.info("Bio energy: %d readings -> %s", len(bio_readings), bio_path)

    # 3. Survivor Debuffs  (best config: sat_floor=50, val_floor=50, dual-region)
    logger.info("=" * 60)
    logger.info("Extracting survivor debuff status...")
    survivor_readings = extract_survivor_debuffs(screens_dir)
    survivor_path = os.path.join(output_dir, "survivor_debuffs.json")
    with open(survivor_path, "w") as f:
        json.dump([asdict(r) for r in survivor_readings], f, indent=2)
    num_frames = len(survivor_readings) // 4 if survivor_readings else 0
    logger.info("Survivor debuffs: %d readings (%d frames x 4) -> %s",
                len(survivor_readings), num_frames, survivor_path)

    # 4. Camera Uptime  (best config: white_threshold=180, sat_floor=50)
    logger.info("=" * 60)
    logger.info("Extracting camera uptime status...")
    camera_readings = extract_camera_uptime(screens_dir)
    camera_path = os.path.join(output_dir, "camera_uptime.json")
    with open(camera_path, "w") as f:
        json.dump([asdict(r) for r in camera_readings], f, indent=2)
    logger.info("Camera uptime: %d readings -> %s", len(camera_readings), camera_path)

    # 5. Combined CSV
    logger.info("=" * 60)
    logger.info("Writing combined labels CSV...")
    csv_path = os.path.join(output_dir, "labels.csv")
    csv_rows = write_combined_csv(
        csv_path, time_burn_events, bio_readings, survivor_readings, camera_readings,
    )
    logger.info("Combined CSV: %d rows -> %s", csv_rows, csv_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Data labelling complete!")
    logger.info("  Time burn events:    %d", len(time_burn_events))
    logger.info("  Bio energy readings: %d", len(bio_readings))
    logger.info("  Survivor readings:   %d (%d frames)", len(survivor_readings), num_frames)
    logger.info("  Camera readings:     %d", len(camera_readings))
    logger.info("  Combined CSV rows:   %d", csv_rows)
    logger.info("All results saved to: %s", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run all data labelling tasks on a captured game session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_labelling.py ../input_capture/re_resistance_captures/won_in_area2
  python run_labelling.py ../input_capture/re_resistance_captures/won_in_area2 -o ../labelled_data/won_in_area2

Output files:
  time_burn_events.json   Time burn/gain popup events
  bio_energy.json         Bio energy counter readings
  survivor_debuffs.json   Survivor health and infection status
  camera_uptime.json      Camera online/disabled/neutral status
  labels.csv              Combined per-frame output of all signals
        """
    )
    parser.add_argument(
        "capture_dir",
        help="Path to the capture session directory (contains screens/ subdirectory)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for results. Defaults to capture_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        run_labelling(args.capture_dir, args.output)
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1
    except Exception as e:
        logging.error("Labelling pipeline failed: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
