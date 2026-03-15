"""Preprocess multiple capture sessions into a single training CSV.

Runs run_labelling + link_labels_to_inputs for each session, then
concatenates the results into one combined training.csv with a `session` column.

Usage:
    python preprocess_sessions.py <session_dir1> [<session_dir2> ...] -o <output_csv>

Example:
    python preprocess_sessions.py \\
        ../input_capture/re_resistance_captures/mind_over_matter_area1_win \\
        ../input_capture/re_resistance_captures/mind_over_matter_area2_win \\
        ../input_capture/re_resistance_captures/mind_over_matter_2_escaped \\
        -o ../modeling/data/mind_over_matter/training.csv
"""

import argparse
import csv
import logging
import os
import sys

from run_labelling import run_labelling
from link_labels_to_inputs import link_labels_to_inputs


def preprocess_session(capture_dir: str, logger: logging.Logger) -> str | None:
    """Run labelling + input linking for one session. Returns path to training_data.csv."""
    session_name = os.path.basename(os.path.normpath(capture_dir))
    output_dir = os.path.join(capture_dir, "output")

    logger.info("=" * 60)
    logger.info("Processing session: %s", session_name)

    # Stage 1: label extraction from screen files
    logger.info("Stage 1: run_labelling -> %s", output_dir)
    run_labelling(capture_dir, output_dir=output_dir)

    # Stage 2: link labels to input frame JSONs
    logger.info("Stage 2: link_labels_to_inputs -> %s", capture_dir)
    training_csv = link_labels_to_inputs(capture_dir, output_dir=capture_dir)

    logger.info("Session complete: %s -> %s", session_name, training_csv)
    return training_csv


def combine_csvs(csv_paths: list[str], session_names: list[str], output_path: str, logger: logging.Logger):
    """Concatenate per-session training CSVs, prepending a `session` column."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_rows = 0
    header_written = False

    with open(output_path, "w", newline="") as out_f:
        writer = None

        for csv_path, session_name in zip(csv_paths, session_names):
            logger.info("Combining: %s (%s)", session_name, csv_path)
            with open(csv_path, "r", newline="") as in_f:
                reader = csv.DictReader(in_f)
                if not header_written:
                    fieldnames = ["session"] + list(reader.fieldnames)
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True

                session_rows = 0
                for row in reader:
                    writer.writerow({"session": session_name, **row})
                    session_rows += 1
                    total_rows += 1

            logger.info("  -> %d rows from %s", session_rows, session_name)

    logger.info("Combined CSV: %d total rows -> %s", total_rows, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multiple capture sessions into one training CSV.",
    )
    parser.add_argument(
        "session_dirs",
        nargs="+",
        help="One or more session directories to process.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for the combined training CSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    csv_paths = []
    session_names = []
    failed = []

    for session_dir in args.session_dirs:
        session_name = os.path.basename(os.path.normpath(session_dir))
        try:
            csv_path = preprocess_session(session_dir, logger)
            csv_paths.append(csv_path)
            session_names.append(session_name)
        except Exception as e:
            logger.error("Failed to process %s: %s", session_dir, e, exc_info=True)
            failed.append(session_dir)

    if not csv_paths:
        logger.error("No sessions processed successfully. Aborting.")
        return 1

    if failed:
        logger.warning("Skipping failed sessions: %s", failed)

    combine_csvs(csv_paths, session_names, args.output, logger)

    logger.info("=" * 60)
    logger.info("All done. Combined training data saved to: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
