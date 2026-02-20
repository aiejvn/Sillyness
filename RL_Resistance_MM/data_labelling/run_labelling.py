"""Unified data labelling pipeline.

Runs all extraction tasks (time burn, bio energy, survivor debuffs, camera uptime)
over a captured game session directory.

Usage:
    python run_labelling.py <capture_dir>
    python run_labelling.py ../input_capture/re_resistance_captures/won_in_area2
    python run_labelling.py ../input_capture/re_resistance_captures/won_in_area2 --display

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
import glob
import json
import logging
import os
from dataclasses import asdict

import cv2
import numpy as np
from PIL import Image

from schemas import (
    BIO_ENERGY_REGION, TIME_BURN_POPUP_REGION, CAMERA_ICON_REGION,
    MAIN_CLOCK_REGION,
    SURVIVOR_HEALTH_BAR_REGIONS, SURVIVOR_FULL_ICON_REGIONS,
    TimeBurnEvent, BioEnergyReading, SurvivorStatusReading, CameraStatusReading,
    ClockReading, ClockTimeBurnEvent,
)
from time_burn import crop_time_region, ocr_time_value, parse_delta
from bio_energy import crop_region, ocr_bio_value, parse_bio_value
from survivor_debuffs import classify_health, classify_infection
from camera_uptime import classify_camera_status
from clock_time_burn import (
    crop_region as crop_clock_region, ocr_clock_value, parse_clock_text,
    extract_clock_readings, detect_time_burn_events as detect_clock_events,
)


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

WINDOW_NAME = "Data Labelling"


def write_combined_csv(path, time_burn_events, bio_readings, survivor_readings, camera_readings, time_burn_mode="clock"):
    """Merge all extraction results into a single per-frame CSV."""
    if time_burn_mode == "clock":
        # For clock mode, use anomaly as the delta (negative=burn, positive=gain)
        time_burn_by_frame = {ev.frame_number: ev.anomaly for ev in time_burn_events}
    else:
        time_burn_by_frame = {ev.frame_number: ev.delta for ev in time_burn_events}
    bio_by_frame = {r.frame_number: r.value for r in bio_readings}

    survivor_by_frame = {}
    for r in survivor_readings:
        survivor_by_frame.setdefault(r.frame_number, {})[r.survivor_id] = r

    camera_by_frame = {r.frame_number: r.camera_status for r in camera_readings}

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


def run_labelling(capture_dir, output_dir=None, display=False, time_burn_mode="clock"):
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
    logger.info("Time burn mode: %s", time_burn_mode)

    if display:
        time_burn_events, bio_readings, survivor_readings, camera_readings = \
            _run_with_display(screens_dir, logger, time_burn_mode=time_burn_mode)
    else:
        time_burn_events, bio_readings, survivor_readings, camera_readings = \
            _run_batch(screens_dir, logger, time_burn_mode=time_burn_mode)

    # Save individual JSONs
    tb_json_name = "clock_time_burn_events.json" if time_burn_mode == "clock" else "time_burn_events.json"
    for name, data in [
        (tb_json_name, time_burn_events),
        ("bio_energy.json", bio_readings),
        ("survivor_debuffs.json", survivor_readings),
        ("camera_uptime.json", camera_readings),
    ]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in data], f, indent=2)
        logger.info("%s: %d entries -> %s", name, len(data), path)

    # Also save clock_readings.json in clock mode
    if time_burn_mode == "clock" and hasattr(run_labelling, "_clock_readings"):
        readings_path = os.path.join(output_dir, "clock_readings.json")
        with open(readings_path, "w") as f:
            json.dump([asdict(r) for r in run_labelling._clock_readings], f, indent=2)
        logger.info("clock_readings.json: %d entries -> %s", len(run_labelling._clock_readings), readings_path)

    # Combined CSV
    csv_path = os.path.join(output_dir, "labels.csv")
    csv_rows = write_combined_csv(
        csv_path, time_burn_events, bio_readings, survivor_readings, camera_readings,
        time_burn_mode=time_burn_mode,
    )
    logger.info("labels.csv: %d rows -> %s", csv_rows, csv_path)

    # Summary
    num_frames = len(survivor_readings) // 4 if survivor_readings else 0
    logger.info("=" * 60)
    logger.info("Data labelling complete!")
    logger.info("  Time burn events:    %d", len(time_burn_events))
    logger.info("  Bio energy readings: %d", len(bio_readings))
    logger.info("  Survivor readings:   %d (%d frames)", len(survivor_readings), num_frames)
    logger.info("  Camera readings:     %d", len(camera_readings))
    logger.info("  Combined CSV rows:   %d", csv_rows)
    logger.info("All results saved to: %s", output_dir)


def _run_batch(screens_dir, logger, time_burn_mode="clock"):
    """Run all extractors in batch mode (no display)."""
    from bio_energy import extract_bio_energy
    from survivor_debuffs import extract_survivor_debuffs
    from camera_uptime import extract_camera_uptime

    logger.info("Running in batch mode (no display)")

    if time_burn_mode == "clock":
        clock_readings = extract_clock_readings(screens_dir)
        time_burn_events = detect_clock_events(clock_readings)
        run_labelling._clock_readings = clock_readings
    else:
        from time_burn import extract_time_burn
        time_burn_events = extract_time_burn(screens_dir)

    bio_readings = extract_bio_energy(screens_dir)
    survivor_readings = extract_survivor_debuffs(screens_dir)
    camera_readings = extract_camera_uptime(screens_dir)

    return time_burn_events, bio_readings, survivor_readings, camera_readings


def _run_with_display(screens_dir, logger, time_burn_mode="clock"):
    """Single-pass extraction with live frame display."""
    pattern = os.path.join(screens_dir, "frame_*.jpg")
    frame_paths = sorted(glob.glob(pattern))

    if not frame_paths:
        logger.warning("No frame JPEGs found in %s", screens_dir)
        return [], [], [], []

    logger.info("Processing %d frames with display", len(frame_paths))

    time_burn_events = []
    bio_readings = []
    survivor_readings = []
    camera_readings = []
    clock_readings_list = []  # only used in clock mode

    # Deduplication state
    prev_time_delta = None
    prev_bio_value = None
    prev_clock_seconds = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for i, path in enumerate(frame_paths):
        basename = os.path.basename(path)
        frame_number = int(basename.replace("frame_", "").replace(".jpg", ""))

        image = Image.open(path)

        # Display the frame
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Display closed by user at frame %d", frame_number)
            break

        # --- Time Burn ---
        if time_burn_mode == "clock":
            cropped_clock = crop_clock_region(image, MAIN_CLOCK_REGION)
            raw_text = ocr_clock_value(cropped_clock)
            if raw_text:
                seconds = parse_clock_text(raw_text)
                if seconds is not None and seconds != prev_clock_seconds:
                    clock_readings_list.append(ClockReading(
                        frame_number=frame_number,
                        clock_seconds=seconds,
                        raw_text=raw_text,
                    ))
                    prev_clock_seconds = seconds
        else:
            cropped_tb = crop_time_region(image, TIME_BURN_POPUP_REGION)
            raw_text, sign = ocr_time_value(cropped_tb)
            if raw_text and sign != 0:
                delta = parse_delta(raw_text, sign)
                if delta is not None and delta != prev_time_delta:
                    time_burn_events.append(TimeBurnEvent(
                        frame_number=frame_number, delta=delta, raw_text=raw_text,
                    ))
                    prev_time_delta = delta
            else:
                prev_time_delta = None

        # --- Bio Energy ---
        cropped_bio = crop_region(image, BIO_ENERGY_REGION)
        raw_bio = ocr_bio_value(cropped_bio)
        if raw_bio:
            bio_val = parse_bio_value(raw_bio)
            if bio_val is not None and bio_val != prev_bio_value:
                bio_readings.append(BioEnergyReading(
                    frame_number=frame_number, value=bio_val, raw_text=raw_bio,
                ))
                prev_bio_value = bio_val

        # --- Survivor Debuffs ---
        for sid in sorted(SURVIVOR_HEALTH_BAR_REGIONS.keys()):
            health_bar = crop_region(image, SURVIVOR_HEALTH_BAR_REGIONS[sid])
            full_icon = crop_region(image, SURVIVOR_FULL_ICON_REGIONS[sid])
            health_result = classify_health(health_bar)
            infection_result = classify_infection(full_icon)
            survivor_readings.append(SurvivorStatusReading(
                frame_number=frame_number,
                survivor_id=sid,
                red=health_result["red"],
                yellow=health_result["yellow"],
                green=health_result["green"],
                purple=infection_result["purple"],
                health_status=health_result["health_status"],
                infection_level=infection_result["infection_level"],
            ))

        # --- Camera Uptime ---
        cropped_cam = crop_region(image, CAMERA_ICON_REGION)
        cam_result = classify_camera_status(cropped_cam)
        camera_readings.append(CameraStatusReading(
            frame_number=frame_number,
            red=cam_result["red"],
            white=cam_result["white"],
            camera_status=cam_result["camera_status"],
        ))

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(frame_paths))

    cv2.destroyAllWindows()
    logger.info("Display pass complete: %d frames processed", len(frame_paths))

    # In clock mode, derive burn events from collected readings
    if time_burn_mode == "clock":
        time_burn_events = detect_clock_events(clock_readings_list)
        run_labelling._clock_readings = clock_readings_list

    return time_burn_events, bio_readings, survivor_readings, camera_readings


def main():
    parser = argparse.ArgumentParser(
        description="Run all data labelling tasks on a captured game session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_labelling.py ../input_capture/re_resistance_captures/won_in_area2
  python run_labelling.py ../input_capture/re_resistance_captures/won_in_area2 --display
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
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Show each frame in a window as it is being processed. Press 'q' to stop early.",
    )
    parser.add_argument(
        "--time-burn-mode", "-tb",
        choices=["clock", "popup"],
        default="clock",
        help="Time burn extraction mode. 'clock' (default) reads the main game clock and "
             "detects anomalies via frame gap analysis. 'popup' uses the old time burn popup OCR.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        run_labelling(args.capture_dir, args.output, display=args.display, time_burn_mode=args.time_burn_mode)
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1
    except Exception as e:
        logging.error("Labelling pipeline failed: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
