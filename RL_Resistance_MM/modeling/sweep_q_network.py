"""sweep_q_network.py — Hyperparameter sweep over reward weights + PER weight.

Runs a grid of 3 models x 6 reward presets x 4 PER weights = 72 total training
runs, each for 3 epochs. Ranks results by a combined score prioritizing play
rate (60%) and eval loss (40%).

Usage:
    python sweep_q_network.py
    python sweep_q_network.py --dry-run            # 1 epoch per run to validate
    python sweep_q_network.py --training-csv data/mind_over_matter/training_data.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiment import REGISTRY, build_model
from reward import RewardWeights
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


# ── Reward presets ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RewardPreset:
    name: str
    weights: RewardWeights


PRESETS: list[RewardPreset] = [
    # RewardPreset("baseline",               RewardWeights(time_burn=1.0, bio_efficiency=1.0, survivor_debuff=1.0, camera_uptime=1.0)),
    RewardPreset("time_burn_focused",      RewardWeights(time_burn=3.0, bio_efficiency=1.0, survivor_debuff=0.5, camera_uptime=0.5)),
    RewardPreset("debuff_focused",         RewardWeights(time_burn=1.0, bio_efficiency=0.5, survivor_debuff=3.0, camera_uptime=1.0)),
    # RewardPreset("bio_efficiency_focused", RewardWeights(time_burn=1.0, bio_efficiency=3.0, survivor_debuff=1.0, camera_uptime=0.5)),
    # RewardPreset("camera_deemphasized",    RewardWeights(time_burn=2.0, bio_efficiency=1.0, survivor_debuff=2.0, camera_uptime=0.1)),
    # RewardPreset("game_optimal",           RewardWeights(time_burn=2.0, bio_efficiency=1.5, survivor_debuff=2.5, camera_uptime=0.25)),
]

PER_WEIGHTS: list[float] = [5, 10, 20] 

# Combined-score weighting: lower score is better.
# play_rate weighted 60% — direct proxy for game-relevant aggression.
# val_loss weighted 40% — necessary condition for learning anything useful.
LOSS_WEIGHT = 0.4
RATE_WEIGHT = 0.6

# ── Model configs ─────────────────────────────────────────────────────────────
# All derived from deep_q_v1.1 — same hyperparams, different architecture.
BASE_CONFIGS = [
    REGISTRY["deep_q_v1.1"],
    dataclasses.replace(REGISTRY["deep_q_v1.1"], name="deep_q_multibranch_mini", network_class="DQN_MultiBranch_Mini"),
    dataclasses.replace(REGISTRY["deep_q_v1.1"], name="deep_q_anynet_mini",      network_class="DQN_AnyNet_Mini"),
]


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(
    preset: RewardPreset,
    per_weight: float,
    training_csv: Path,
    screens_dir: Path,
    base_cfg,
    device: torch.device,
    sweep_ckpt_dir: Path,
    num_epochs: int = 3,
    local_rank: int = 0,
    world_size: int = 1,
) -> dict:
    """Train one (model, preset, per_weight) combination and return its metrics."""
    run_id = f"sweep_{base_cfg.name}_{preset.name}_per{int(per_weight)}"
    cfg = dataclasses.replace(
        base_cfg,
        name=run_id,
        per_space_weight=per_weight,
        num_epochs=num_epochs,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    df = prepare_dataframe(training_csv, cfg, reward_weights=preset.weights)
    valid_mask = df["frame"].apply(
        lambda f: (screens_dir / f"frame_{int(f):06d}.jpg").exists()
    )
    df_valid = df[valid_mask].reset_index(drop=True)
    if local_rank == 0:
        print(f"Frames with images: {len(df_valid)} / {len(df)}")

    train_loader, val_loader, train_generator = build_dataloaders(df_valid, screens_dir, cfg,
                                                                  rank=local_rank, world_size=world_size)
    train_base_seed = train_generator.initial_seed()

    # ── Model ─────────────────────────────────────────────────────────────────
    output_columns = list(cfg.output_columns)
    space_idx = output_columns.index("key_space")
    model = wrap_model(build_model(cfg).to(device), device, local_rank, world_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    final_val_loss, final_play_rate = float("inf"), 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        if world_size > 1: # Re-seed for multi-GPU training
            train_generator.manual_seed(train_base_seed + epoch - 1)
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            cfg.l1_inactive_weight, space_idx,
        )
        val_loss, play_rate = eval_epoch(
            model, val_loader, device, epoch,
            cfg.l1_inactive_weight, space_idx,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if local_rank == 0:
            print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  play_rate={play_rate:.4f}")
        final_val_loss, final_play_rate = val_loss, play_rate

    # ── Save ──────────────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, cfg, train_losses, val_losses, sweep_ckpt_dir)

    # GPU memory cleanup between runs
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run_id": run_id,
        "network": base_cfg.name,
        "per_weight": per_weight,
        "preset_name": preset.name,
        "time_burn": preset.weights.time_burn,
        "bio_efficiency": preset.weights.bio_efficiency,
        "survivor_debuff": preset.weights.survivor_debuff,
        "camera_uptime": preset.weights.camera_uptime,
        "final_val_loss": final_val_loss,
        "final_play_rate": final_play_rate,
    }


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_results(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)

    loss_min, loss_max = df["final_val_loss"].min(), df["final_val_loss"].max()
    rate_min, rate_max = df["final_play_rate"].min(), df["final_play_rate"].max()

    df["norm_loss"] = (df["final_val_loss"] - loss_min) / (loss_max - loss_min + 1e-9)
    df["norm_rate"] = (df["final_play_rate"] - rate_min) / (rate_max - rate_min + 1e-9)
    df["combined_score"] = LOSS_WEIGHT * df["norm_loss"] + RATE_WEIGHT * (1.0 - df["norm_rate"])

    return df.sort_values("combined_score").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep reward weights and PER weight for deep_q_v1.1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--training-csv", default=None,
                        help="Path to training_data.csv.")
    parser.add_argument("--screens-dir", default=None,
                        help="Path to screens/ directory with frame_NNNNNN.jpg files.")
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for sweep checkpoints and results.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 epoch per combination to validate paths and grid.")
    args = parser.parse_args()

    training_csv = Path(args.training_csv) if args.training_csv else \
        PROJECT_ROOT / "data" / "test_won_in_area2" / "training_data.csv"
    screens_dir = Path(args.screens_dir) if args.screens_dir else \
        PROJECT_ROOT.parent / "input_capture" / "re_resistance_captures" / "won_in_area2" / "screens"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT

    sweep_ckpt_dir = output_dir / "checkpoints" / "sweep"
    results_dir = output_dir / "sweep_results"
    sweep_ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 1 if args.dry_run else 5
    device, local_rank, world_size = setup_device()

    total_runs = len(BASE_CONFIGS) * len(PRESETS) * len(PER_WEIGHTS)
    print(f"Sweep: {len(BASE_CONFIGS)} models x {len(PRESETS)} presets x {len(PER_WEIGHTS)} PER weights = {total_runs} runs")
    print(f"Epochs per run: {num_epochs}{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"Device: {device}")
    print(f"Training CSV: {training_csv}")
    print(f"Screens dir:  {screens_dir}\n")

    today = datetime.date.today().strftime("%Y-%m-%d")
    suffix = "-dry" if args.dry_run else ""
    csv_path = results_dir / f"{today}-sweep{suffix}.csv"

    grid = [(b, p, w) for b in BASE_CONFIGS for p in PRESETS for w in PER_WEIGHTS]
    results = []
    for run_num, (base_cfg, preset, per_weight) in enumerate(grid, start=1):
        print(f"\n{'='*70}")
        print(f"Run {run_num}/{total_runs}: model={base_cfg.name}  preset={preset.name}  PER={per_weight}")
        print(f"  weights: time_burn={preset.weights.time_burn}  bio={preset.weights.bio_efficiency}"
              f"  debuff={preset.weights.survivor_debuff}  camera={preset.weights.camera_uptime}")
        print(f"{'='*70}")

        result = run_single(
            preset, per_weight, training_csv, screens_dir,
            base_cfg, device, sweep_ckpt_dir, num_epochs=num_epochs,
            local_rank=local_rank, world_size=world_size,
        )

        if local_rank == 0:
            results.append(result)
            # Save after every run so results are available if the sweep is interrupted
            rank_results(results).to_csv(csv_path, index=False)
            print(f"Results saved ({run_num}/{total_runs}): {csv_path}")

    if local_rank == 0:
        # ── Print summary ─────────────────────────────────────────────────────
        df_ranked = rank_results(results)
        w = 85
        print(f"\n{'='*w}")
        print(f"SWEEP RESULTS  (score = {LOSS_WEIGHT}*norm_loss + {RATE_WEIGHT}*(1-norm_rate), lower=better)")
        print(f"{'='*w}")
        header = f"{'Rank':<5} {'Network':<26} {'Preset':<24} {'PER':>4} {'ValLoss':>8} {'PlayRate':>9} {'Score':>7}"
        print(header)
        print(f"{'-'*w}")
        for rank, row in df_ranked.iterrows():
            print(
                f"{rank+1:<5} {row['network']:<26} {row['preset_name']:<24} {row['per_weight']:>4.0f} "
                f"{row['final_val_loss']:>8.4f} {row['final_play_rate']:>9.4f} {row['combined_score']:>7.4f}"
            )
        print(f"\nResults saved to: {csv_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
