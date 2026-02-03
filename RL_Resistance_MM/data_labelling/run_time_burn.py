"""CLI entry point for time-burn extraction.

Usage:
    python run_time_burn.py <session_screens_dir> [--output results.json]

Example:
    python run_time_burn.py ../input_capture/re_resistance_captures/won_in_area2/screens
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict

from time_burn import extract_time_burn


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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
