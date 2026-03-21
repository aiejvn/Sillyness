"""preprocess_frames.py — Pre-process capture frames to fully-transformed tensors.

Reads a training CSV, applies the full training transform pipeline
(Grayscale → Resize → ToTensor → Normalize) to each frame, and saves the
result as a float32 .npy array in a screens_{size}/ directory alongside the
originals. At training time the dataset simply np.load()s the file — zero
per-epoch decode, resize, or normalize overhead.

Output shape per file: (1, H, W) float32, values in [-1, 1].
Resumable: skips frames whose .npy already exists.

Usage:
    # Multi-session (mind_over_matter)
    python preprocess_frames.py \\
        --training-csv data/mind_over_matter/training.csv \\
        --sessions-base-dir ../../input_capture/re_resistance_captures \\
        --img-size 224

    # Single-session
    python preprocess_frames.py \\
        --training-csv data/test_won_in_area2/training_data.csv \\
        --screens-dir ../../input_capture/re_resistance_captures/won_in_area2/screens \\
        --img-size 224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),                        # → (1, H, W) float32 in [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # → [-1, 1]
])


def preprocess(
    training_csv: Path,
    img_size: int,
    screens_dir: Path | None = None,
    sessions_base_dir: Path | None = None,
) -> None:
    df = pd.read_csv(training_csv)
    multi_session = "session" in df.columns
    out_subdir = f"screens_{img_size}"

    print(f"Loaded {len(df)} rows from {training_csv}")
    print(f"Mode: {'multi-session' if multi_session else 'single-session'}")
    print(f"Output subdir: {out_subdir}  (shape: 1×{img_size}×{img_size} float32 .npy)\n")

    resize = transforms.Resize((img_size, img_size))

    written = 0
    skipped = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        frame_num = int(row["frame"])

        if multi_session:
            session  = row["session"]
            src_path = sessions_base_dir / session / "screens" / f"frame_{frame_num:06d}.jpg"
            dst_dir  = sessions_base_dir / session / out_subdir
        else:
            src_path = screens_dir / f"frame_{frame_num:06d}.jpg"
            dst_dir  = screens_dir.parent / out_subdir

        dst_path = dst_dir / f"frame_{frame_num:06d}.npy"

        if dst_path.exists():
            skipped += 1
            continue

        if not src_path.exists():
            print(f"\nWarning: missing {src_path} — skipping")
            missing += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        image = resize(Image.open(src_path).convert("RGB"))
        arr = TRANSFORM(image).numpy()  # (1, img_size, img_size) float32
        np.save(dst_path, arr)
        written += 1

    print(f"\nDone.  Written: {written}  Skipped (exists): {skipped}  Missing src: {missing}")


def debug_one(
    training_csv: Path,
    img_size: int,
    screens_dir: Path | None = None,
    sessions_base_dir: Path | None = None,
) -> None:
    """Apply the full transform pipeline to one random frame and save as a JPEG.

    The normalized tensor (values in [-1, 1]) is rescaled back to [0, 255]
    so the output is a viewable greyscale image showing exactly what the
    network receives as input.  Saved to debug_frame_{frame_num}.jpg in the
    current working directory.
    """
    df = pd.read_csv(training_csv)
    multi_session = "session" in df.columns

    row = df.sample(1).iloc[0]
    frame_num = int(row["frame"])

    if multi_session:
        src_path = sessions_base_dir / row["session"] / "screens" / f"frame_{frame_num:06d}.jpg"
    else:
        src_path = screens_dir / f"frame_{frame_num:06d}.jpg"

    print(f"Debug frame: {src_path}")

    resize = transforms.Resize((img_size, img_size))
    image = resize(Image.open(src_path).convert("RGB"))
    tensor = TRANSFORM(image)  # (1, H, W) float32, [-1, 1]

    # Invert normalisation: [-1, 1] → [0, 255] uint8
    display = ((tensor * 0.5 + 0.5).clamp(0, 1) * 255).byte()
    out_img = Image.fromarray(display.squeeze(0).numpy(), mode="L")

    out_path = Path(f"debug_frame_{frame_num}.jpg")
    out_img.save(out_path, format="JPEG", quality=95)
    print(f"Saved: {out_path}  (shape: {list(tensor.shape)}, "
          f"min={tensor.min():.3f}, max={tensor.max():.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process frames: Grayscale+Resize+ToTensor+Normalize → .npy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--training-csv", required=True,
                        help="Path to training CSV.")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Output image size (square). Default: 224.")
    parser.add_argument("--screens-dir", default=None,
                        help="Single-session: original screens/ directory.")
    parser.add_argument("--sessions-base-dir", default=None,
                        help="Multi-session: captures root (contains <session>/screens/).")
    parser.add_argument("--debug", action="store_true",
                        help="Apply transforms to one random frame and save as a viewable JPEG.")
    args = parser.parse_args()

    training_csv = Path(args.training_csv)
    screens_dir = Path(args.screens_dir) if args.screens_dir else None
    sessions_base_dir = Path(args.sessions_base_dir) if args.sessions_base_dir else None

    df_peek = pd.read_csv(training_csv, nrows=1)
    multi_session = "session" in df_peek.columns

    if multi_session and sessions_base_dir is None:
        parser.error("Multi-session CSV requires --sessions-base-dir")
    if not multi_session and screens_dir is None:
        parser.error("Single-session CSV requires --screens-dir")

    if args.debug:
        debug_one(training_csv, args.img_size,
                  screens_dir=screens_dir,
                  sessions_base_dir=sessions_base_dir)
    else:
        preprocess(training_csv, args.img_size,
                   screens_dir=screens_dir,
                   sessions_base_dir=sessions_base_dir)


if __name__ == "__main__":
    main()
