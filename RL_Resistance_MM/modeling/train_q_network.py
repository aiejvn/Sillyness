"""train_q_network.py — CLI training script for RL Resistance MM.

Equivalent to train_q_network.ipynb but runnable headless from the command line.
All core logic lives in trainer.py; this file only handles CLI args and orchestration.

Defaults to multi-session mind_over_matter training data. Pass --screens-dir
to fall back to single-session mode.

Usage:
    python train_q_network.py
    python train_q_network.py --experiment deep_q_v1.1 --epochs 20
    python train_q_network.py \\
        --training-csv data/test_won_in_area2/training_data.csv \\
        --screens-dir ../input_capture/re_resistance_captures/won_in_area2/screens
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiment import REGISTRY, build_model
import torch.distributed as dist

from trainer import (
    prepare_dataframe,
    build_dataloaders,
    train_epoch,
    eval_epoch,
    save_checkpoint,
    setup_device,
    wrap_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Deep Q-Network for RL Resistance MM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", "-e",
        default="deep_q_v1.1",
        choices=list(REGISTRY.keys()),
        help="Experiment name from experiment.py REGISTRY (default: deep_q_v1.1)",
    )
    parser.add_argument(
        "--training-csv",
        default=None,
        help="Path to training CSV. Defaults to data/mind_over_matter/training.csv.",
    )
    parser.add_argument(
        "--lmdb",
        default=None,
        help="Path to LMDB built by build_lmdb.py. Defaults to data/mind_over_matter/frames.lmdb.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for checkpoint output. Defaults to modeling/checkpoints/",
    )
    parser.add_argument(
        "--epochs", "-n",
        type=int,
        default=None,
        help="Number of training epochs. Defaults to cfg.num_epochs.",
    )
    args = parser.parse_args()

    cfg = REGISTRY[args.experiment]

    training_csv = Path(args.training_csv) if args.training_csv else \
        PROJECT_ROOT / "data" / "mind_over_matter" / "training.csv"
    lmdb_path = Path(args.lmdb) if args.lmdb else \
        PROJECT_ROOT / "data" / "mind_over_matter" / "frames.lmdb"
    output_dir = Path(args.output_dir) if args.output_dir else \
        PROJECT_ROOT / "checkpoints"
    num_epochs = args.epochs if args.epochs is not None else cfg.num_epochs

    device, local_rank, world_size = setup_device()
    output_columns = list(cfg.output_columns)
    space_idx    = output_columns.index("key_space")

    if local_rank == 0:
        print(f"Experiment:   {cfg.name}")
        print(f"Network:      {cfg.network_class}")
        print(f"Device:       {device}")
        print(f"Epochs:       {num_epochs}")
        print(f"Training CSV: {training_csv}")
        if sessions_base_dir:
            print(f"Sessions base: {sessions_base_dir}")
        else:
            print(f"Screens dir:   {screens_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = prepare_dataframe(training_csv, cfg)

    multi_session = "session" in df.columns
    if multi_session:
        valid_mask = df.apply(
            lambda r: (sessions_base_dir / r["session"] / "screens" / f"frame_{int(r['frame']):06d}.jpg").exists(),
            axis=1,
        )
    else:
        valid_mask = df["frame"].apply(
            lambda f: (screens_dir / f"frame_{int(f):06d}.jpg").exists()
        )
    df_valid = df[valid_mask].reset_index(drop=True)
    if local_rank == 0:
        print(f"Frames with images: {len(df_valid)} / {len(df)}")

    train_loader, val_loader, train_generator = build_dataloaders(
        df_valid, cfg, rank=local_rank, world_size=world_size,
        screens_dir=screens_dir, sessions_base_dir=sessions_base_dir,
    )
    train_base_seed = train_generator.initial_seed()

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = wrap_model(build_model(cfg).to(device), device, local_rank, world_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses = []
    val_losses   = []

    for epoch in range(1, num_epochs + 1):
        if world_size > 1: # Re-seed for multi-GPU training
            train_generator.manual_seed(train_base_seed + epoch - 1)
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch,
                                 cfg.l1_inactive_weight, space_idx)
        val_loss, play_rate = eval_epoch(model, val_loader, device, epoch,
                                         cfg.l1_inactive_weight, space_idx)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if local_rank == 0:
            print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  play_rate={play_rate:.4f}")

    if local_rank == 0:
        print(f"\nFinished {len(train_losses)} epochs.")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, cfg, train_losses, val_losses, output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
