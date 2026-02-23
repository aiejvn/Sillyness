"""Link labelled data (labels.csv) with input capture (frame JSONs) into a training dataset.

Reads:
    - <capture_dir>/labels.csv          (produced by data_labelling/run_labelling.py)
    - <capture_dir>/frames/raw/frame_NNNNNN.json  (produced by input_capture)

Writes:
    - <output_dir>/training_data.csv    (one row per frame, inputs + labels)

Usage:
    python link_labels_to_inputs.py <capture_dir>
    python link_labels_to_inputs.py ../input_capture/re_resistance_captures/won_in_area2
    python link_labels_to_inputs.py ../input_capture/re_resistance_captures/won_in_area2 -o ../datasets/won_in_area2
"""

import argparse
import csv
import json
import logging
import os

logger = logging.getLogger(__name__)

# All keys the game uses - one-hot encoded in the output
TRACKED_KEYS = [
    "w", "a", "s", "d",
    "q", "e", "r", "f", "v", "m",
    "1", "2", "3", "4",
    "space", "up", "down", "left", "right",
]

OUTPUT_COLUMNS = [
    "frame",
    "timestamp",
    # Mouse
    "mouse_x", "mouse_y",
    "mouse_left", "mouse_middle", "mouse_right",
    "mouse_dx", "mouse_dy",
    # Keyboard (one column per tracked key)
    *[f"key_{k}" for k in TRACKED_KEYS],
    # Labels
    "time_burn_delta",
    "bio_energy",
    "s1_health", "s1_infection",
    "s2_health", "s2_infection",
    "s3_health", "s3_infection",
    "s4_health", "s4_infection",
    "camera_status",
]


def load_labels(labels_csv_path: str) -> dict[int, dict]:
    """Load labels.csv into a dict keyed by frame number."""
    labels = {}
    with open(labels_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row["frame"])
            labels[frame_num] = row
    return labels


def extract_input_features(frame_json: dict) -> dict:
    """Extract input features from a per-frame JSON."""
    inp = frame_json["input_raw"]
    kb = inp["keyboard"]
    mouse = inp["mouse"]

    # Keyboard: one-hot for each tracked key
    keys_pressed = set(k.lower() for k in kb["keys_pressed"])
    key_features = {f"key_{k}": int(k in keys_pressed) for k in TRACKED_KEYS}

    # Mouse position
    pos = mouse["position"]
    mouse_x, mouse_y = pos[0], pos[1]

    # Mouse buttons
    btns = mouse["buttons_current"]
    mouse_left = int(btns[0]) if len(btns) > 0 else 0
    mouse_middle = int(btns[1]) if len(btns) > 1 else 0
    mouse_right = int(btns[2]) if len(btns) > 2 else 0

    # Aggregate mouse movement deltas between frames
    dx_total, dy_total = 0, 0
    for evt in mouse.get("movement_events", []):
        dx_total += evt.get("dx", 0)
        dy_total += evt.get("dy", 0)

    return {
        "timestamp": frame_json["monotonic_timestamp"],
        "mouse_x": mouse_x,
        "mouse_y": mouse_y,
        "mouse_left": mouse_left,
        "mouse_middle": mouse_middle,
        "mouse_right": mouse_right,
        "mouse_dx": dx_total,
        "mouse_dy": dy_total,
        **key_features,
    }


def link_labels_to_inputs(capture_dir: str, output_dir: str | None = None):
    """Build training dataset by joining frame inputs with labels."""
    labels_path = os.path.join(capture_dir, "output/labels.csv")
    frames_dir = os.path.join(capture_dir, "frames", "raw")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"labels.csv not found at {labels_path}. Run data_labelling/run_labelling.py first."
        )
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    if output_dir is None:
        output_dir = capture_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load labels
    labels = load_labels(labels_path)
    logger.info("Loaded %d labelled frames from %s", len(labels), labels_path)

    # Discover frame JSONs
    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.startswith("frame_") and f.endswith(".json")
    )
    logger.info("Found %d frame JSONs in %s", len(frame_files), frames_dir)

    # Build training rows
    output_path = os.path.join(output_dir, "training_data.csv")
    matched = 0
    skipped = 0

    with open(output_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for frame_file in frame_files:
            frame_number = int(frame_file.replace("frame_", "").replace(".json", ""))

            if frame_number not in labels:
                skipped += 1
                continue

            frame_path = os.path.join(frames_dir, frame_file)
            with open(frame_path, "r") as f:
                frame_json = json.load(f)

            input_features = extract_input_features(frame_json)
            label_row = labels[frame_number]

            row = {
                "frame": frame_number,
                **input_features,
                # Labels (carry over from labels.csv, empty string if missing)
                "time_burn_delta": label_row.get("time_burn_delta", ""),
                "bio_energy": label_row.get("bio_energy", ""),
                "s1_health": label_row.get("s1_health", ""),
                "s1_infection": label_row.get("s1_infection", ""),
                "s2_health": label_row.get("s2_health", ""),
                "s2_infection": label_row.get("s2_infection", ""),
                "s3_health": label_row.get("s3_health", ""),
                "s3_infection": label_row.get("s3_infection", ""),
                "s4_health": label_row.get("s4_health", ""),
                "s4_infection": label_row.get("s4_infection", ""),
                "camera_status": label_row.get("camera_status", ""),
            }

            writer.writerow(row)
            matched += 1

            if matched % 500 == 0:
                logger.info("Processed %d frames", matched)

    logger.info("Training dataset: %d matched rows, %d skipped (no label)", matched, skipped)
    logger.info("Saved to %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Link input capture data with labelled data to create training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python link_labels_to_inputs.py ../input_capture/re_resistance_captures/won_in_area2
  python link_labels_to_inputs.py ../input_capture/re_resistance_captures/won_in_area2 -o ../datasets

Output:
  training_data.csv — one row per frame with input features + reward labels

Columns:
  frame, timestamp,
  mouse_x, mouse_y, mouse_left, mouse_middle, mouse_right, mouse_dx, mouse_dy,
  key_w, key_a, key_s, key_d, ..., (one-hot per tracked key)
  time_burn_delta, bio_energy,
  s1_health, s1_infection, ..., s4_health, s4_infection,
  camera_status
        """
    )
    parser.add_argument(
        "capture_dir",
        help="Path to the capture session directory (contains labels.csv and frames/raw/)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for training_data.csv. Defaults to capture_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        link_labels_to_inputs(args.capture_dir, args.output)
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1
    except Exception as e:
        logging.error("Failed: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
