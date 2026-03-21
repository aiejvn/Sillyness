"""build_lmdb.py — Convert training CSV + JPEG frames into an LMDB for fast training I/O.

Stores raw JPEG bytes keyed by:
  - Multi-session CSV (has 'session' col): b"{session}/{frame:06d}"
  - Single-session CSV:                   b"{frame:06d}"

Resumable: skips keys already present in an existing LMDB.

Usage:
    # Multi-session (mind_over_matter)
    python build_lmdb.py \\
        --training-csv data/mind_over_matter/training.csv \\
        --sessions-base-dir ../../input_capture/re_resistance_captures \\
        --output data/mind_over_matter/frames.lmdb

    # Single-session
    python build_lmdb.py \\
        --training-csv data/test_won_in_area2/training_data.csv \\
        --screens-dir ../../input_capture/re_resistance_captures/won_in_area2/screens \\
        --output data/test_won_in_area2/frames.lmdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lmdb
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def build(
    training_csv: Path,
    output: Path,
    screens_dir: Path | None = None,
    sessions_base_dir: Path | None = None,
    map_size_gb: int = 30,
) -> None:
    df = pd.read_csv(training_csv)
    multi_session = "session" in df.columns
    print(f"Loaded {len(df)} rows from {training_csv}")
    print(f"Mode: {'multi-session' if multi_session else 'single-session'}")
    print(f"Output: {output}\n")

    map_size = map_size_gb * 1024 ** 3
    env = lmdb.open(str(output), map_size=map_size, subdir=True)

    written = 0
    skipped = 0

    with env.begin(write=True) as txn:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing LMDB"):
            frame_num = int(row["frame"])

            if multi_session:
                key = f"{row['session']}/{frame_num:06d}".encode()
                img_path = sessions_base_dir / row["session"] / "screens" / f"frame_{frame_num:06d}.jpg"
            else:
                key = f"{frame_num:06d}".encode()
                img_path = screens_dir / f"frame_{frame_num:06d}.jpg"

            if txn.get(key) is not None:
                skipped += 1
                continue

            if not img_path.exists():
                print(f"\nWarning: missing {img_path} — skipping")
                continue

            txn.put(key, img_path.read_bytes())
            written += 1

    env.close()
    print(f"\nDone. Written: {written}  Skipped (already present): {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LMDB from training CSV + JPEG frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--training-csv", required=True,
                        help="Path to training CSV.")
    parser.add_argument("--output", required=True,
                        help="Output LMDB directory path.")
    parser.add_argument("--screens-dir", default=None,
                        help="Single-session: directory containing frame_NNNNNN.jpg files.")
    parser.add_argument("--sessions-base-dir", default=None,
                        help="Multi-session: captures root (contains <session>/screens/).")
    parser.add_argument("--map-size-gb", type=int, default=30,
                        help="LMDB map size ceiling in GB (sparse — not pre-allocated). Default: 30.")
    args = parser.parse_args()

    training_csv = Path(args.training_csv)
    output = Path(args.output)
    screens_dir = Path(args.screens_dir) if args.screens_dir else None
    sessions_base_dir = Path(args.sessions_base_dir) if args.sessions_base_dir else None

    df_peek = pd.read_csv(training_csv, nrows=1)
    multi_session = "session" in df_peek.columns

    if multi_session and sessions_base_dir is None:
        parser.error("Multi-session CSV requires --sessions-base-dir")
    if not multi_session and screens_dir is None:
        parser.error("Single-session CSV requires --screens-dir")

    build(training_csv, output,
          screens_dir=screens_dir,
          sessions_base_dir=sessions_base_dir,
          map_size_gb=args.map_size_gb)


if __name__ == "__main__":
    main()
