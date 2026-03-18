"""trainer.py — Shared training logic for RL Resistance MM.

Importable by both train_q_network.ipynb and train_q_network.py.
Neither file should duplicate any of the logic here.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from experiment import ExperimentConfig, build_model
from reward import compute_rewards_for_episode, RewardWeights


# ── Dataset ──────────────────────────────────────────────────────────────────

class ResistanceDataset(Dataset):
    """Dataset over sliding-window stacks of STACK_SIZE consecutive frames.

    Each item corresponds to one stack:
    - `sample_starts` is a list of row indices into `dataframe`, one per stack.
    - frames [start, start+1, ..., start+S-1] are stacked → (STACK_SIZE, H, W)
    - actions and Q-target come from the LAST frame in each stack.

    Decoupling sample selection from the dataframe lets train/val splits be
    spread across the full episode rather than split temporally.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        screens_dir: Path,
        sample_starts: list[int],
        output_columns: list[str],
        transform=None,
        stack_size: int = 4,
    ):
        self.df = dataframe
        self.screens_dir = Path(screens_dir)
        self.sample_starts = sample_starts
        self.output_columns = output_columns
        self.transform = transform
        self.stack_size = stack_size

    def __len__(self):
        return len(self.sample_starts)

    def _load_frame(self, frame_num: int) -> torch.Tensor:
        img_path = self.screens_dir / f"frame_{frame_num:06d}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        start = self.sample_starts[idx]
        frames = [
            self._load_frame(int(self.df.iloc[start + k]["frame"]))
            for k in range(self.stack_size)
        ]
        stacked = torch.cat(frames, dim=0)  # (STACK_SIZE, H, W)

        row = self.df.iloc[start + self.stack_size - 1]
        actions = torch.tensor([row[c] for c in self.output_columns], dtype=torch.float32)
        target  = torch.tensor(row["discounted_return"], dtype=torch.float32)

        return stacked, actions, target

    def get_sample_weights(self, space_weight: float = 5.0) -> list[float]:
        """Per-sample weights for WeightedRandomSampler.

        Stacks whose last frame has key_space=1 get weight=space_weight;
        all others get 1.0. Oversamples space-press moments so the model
        sees more of those high-value frames per epoch.
        """
        weights = []
        for start in self.sample_starts:
            last_row = self.df.iloc[start + self.stack_size - 1]
            is_space = float(last_row["key_space"]) == 1.0
            weights.append(space_weight if is_space else 1.0)
        return weights


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_dataframe(training_csv: str | Path, cfg: ExperimentConfig, reward_weights: RewardWeights | None = None) -> pd.DataFrame:
    """Load training CSV, compute rewards and discounted returns.

    Returns a DataFrame with added columns:
        reward            — normalised per-frame reward
        discounted_return — discounted future return (Q-learning target)

    Rows before cfg.starting_frame and rows with |reward| >= 100 are dropped.
    """
    df = pd.read_csv(training_csv)
    print(f"Loaded {len(df)} frames from {training_csv}")

    rows = df.to_dict(orient="records")
    df["reward"] = compute_rewards_for_episode(rows, weights=reward_weights, apply_relu=True)

    # Drop outlier rewards (likely extraction artefacts)
    df.loc[df["reward"].abs() >= 100, "reward"] = 0

    # Skip pre-game frames (loading screen / white flash)
    df = df[df["frame"] >= cfg.starting_frame]

    # Normalise rewards to zero mean, unit variance
    reward_mean = df["reward"].mean()
    reward_std  = df["reward"].std()
    df["reward"] = (df["reward"] - reward_mean) / (reward_std + 1e-8)
    print(f"Reward normalised: mean={reward_mean:.4f}, std={reward_std:.4f}")

    # Discounted return (backward pass)
    returns = np.zeros(len(df))
    running = 0.0
    for t in reversed(range(len(df))):
        running = df["reward"].iloc[t] + cfg.gamma * running
        returns[t] = running
    df["discounted_return"] = returns

    return df


def build_dataloaders(
    df_valid: pd.DataFrame,
    screens_dir: str | Path,
    cfg: ExperimentConfig,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with PER for space-press frames.

    Args:
        df_valid:    DataFrame already filtered to frames with existing screen files.
        screens_dir: Directory containing frame_NNNNNN.jpg files.
        cfg:         Experiment config (stack_size, img_size, batch_size, etc.)
        seed:        RNG seed for reproducible train/val split.
        rank:        DDP process rank (0 in single-process mode).
        world_size:  Total number of DDP processes (1 in single-process mode).

    Returns:
        (train_loader, val_loader)
    """
    output_columns = list(cfg.output_columns)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    n_samples  = len(df_valid) - cfg.stack_size + 1
    all_starts = list(range(n_samples))

    # All ranks use the same seed so the train/val split is identical everywhere.
    # Per-rank training diversity comes from WeightedRandomSampler (replacement draws).
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(all_starts)

    split        = int(n_samples * cfg.train_split)
    train_starts = all_starts[:split]
    val_starts   = all_starts[split:]

    train_ds = ResistanceDataset(df_valid, screens_dir, train_starts, output_columns, transform, cfg.stack_size)
    val_ds   = ResistanceDataset(df_valid, screens_dir, val_starts,   output_columns, transform, cfg.stack_size)

    space_base_rate = df_valid["key_space"].mean()
    print(f"Space-press base rate: {space_base_rate:.2%}  |  PER weight: {cfg.per_space_weight}x")

    # Train: WeightedRandomSampler preserves PER space-press oversampling on every process.
    # Explicit per-rank generator ensures different batches across GPUs in DDP mode.
    train_weights = train_ds.get_sample_weights(space_weight=cfg.per_space_weight)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed + rank)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights) // world_size,
        replacement=True,
        generator=train_generator,
    )

    # Val: DistributedSampler shards across processes so all_reduce gives the global metric.
    if world_size > 1:
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader  = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler, num_workers=0)
    else:
        val_loader  = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=0)

    print(f"Samples: {n_samples}  (train: {len(train_ds)}, val: {len(val_ds)})")
    return train_loader, val_loader, train_generator


# ── Loss & training ───────────────────────────────────────────────────────────

def masked_q_loss(
    q_pred: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    l1_weight: float,
    space_idx: int,
) -> torch.Tensor:
    """Combined loss for active and inactive action heads.

    Active heads   (action != 0): MSE vs discounted return.
    Inactive heads (action == 0): L1 toward zero.

    key_space is exempt from the inactive L1 penalty so the model is never
    penalised for wanting to press space on frames where the human didn't.
    PER (WeightedRandomSampler) handles oversampling space-press moments.

    Args:
        q_pred:    (B, NUM_OUTPUTS) predicted Q-values
        actions:   (B, NUM_OUTPUTS) recorded action values
        targets:   (B,)             discounted return per frame
        l1_weight: scalar weight for inactive-head L1 penalty
        space_idx: column index of key_space in the action vector
    """
    active   = (actions != 0).float()
    inactive = 1.0 - active

    targets_exp = targets.unsqueeze(1).expand_as(q_pred)

    # MSE on active heads
        # To investigate: should we limit MSE to just active keys with q_pred < 1? 
    n_active = active.sum().clamp(min=1)
    mse_loss = ((q_pred - targets_exp) ** 2 * active).sum() / n_active

    # Bonus: reward active heads whose predicted Q >= 1 (model is confident
    # the action was good). Subtract a small bonus from the loss for each such
    # head so that high-confidence correct predictions are encouraged.
    active_high = ((q_pred >= 1) & (actions != 0)).float()
    n_active_high = active_high.sum().clamp(min=1)
    confidence_bonus = active_high.sum() / n_active_high  # normalised count → subtract from loss
    mse_loss = mse_loss - confidence_bonus

    # L1 toward zero on inactive heads — exempt key_space so the model is
    # free to output high Q-values for space regardless of whether it was pressed.
    inactive_masked = inactive.clone()
    inactive_masked[:, space_idx] = 0.0

    n_inactive = inactive_masked.sum().clamp(min=1)
    l1_loss = (q_pred.abs() * inactive_masked).sum() / n_inactive

    return mse_loss + l1_weight * l1_loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    l1_weight: float,
    space_idx: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    _rank = dist.get_rank() if dist.is_initialized() else 0
    for images, actions, targets in tqdm(loader, desc=f"Epoch {epoch}", disable=(_rank != 0)):
        images  = images.to(device)
        actions = actions.to(device)
        targets = targets.to(device)

        q_pred = model(images)
        loss   = masked_q_loss(q_pred, actions, targets, l1_weight, space_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    l1_weight: float,
    space_idx: int
) -> tuple[float, float]:
    model.eval()
    total_loss           = 0.0
    n_batches            = 0
    space_presses_hat    = 0
    space_presses_truth  = 0

    _rank = dist.get_rank() if dist.is_initialized() else 0
    for images, actions, targets in tqdm(loader, desc=f"Eval {epoch}", disable=(_rank != 0)):
        images  = images.to(device)
        actions = actions.to(device)
        targets = targets.to(device)

        q_pred = model(images)

        # Checking aggression
        space_presses_truth += (actions[:, space_idx] == 1).sum().item()
        space_presses_hat   += (q_pred[:, space_idx] > 1).sum().item()

        loss = masked_q_loss(q_pred, actions, targets, l1_weight, space_idx)
        total_loss += loss.item()
        n_batches  += 1

    # In DDP mode, sum metrics across all processes before computing final values.
    if dist.is_initialized():
        t = torch.tensor(
            [total_loss, float(n_batches), float(space_presses_hat), float(space_presses_truth)],
            device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss          = t[0].item()
        n_batches           = int(t[1].item())
        space_presses_hat   = int(t[2].item())
        space_presses_truth = int(t[3].item())

    val_loss = total_loss / max(n_batches, 1)
    play_rate = space_presses_hat / max(space_presses_truth, 1)
    print(f"Play Rate: {play_rate:.4f}")

    return val_loss, play_rate


# ── Device utilities ──────────────────────────────────────────────────────────

def setup_device() -> tuple[torch.device, int, int]:
    """Return (device, local_rank, world_size).

    Single-process mode (LOCAL_RANK not set): local_rank=0, world_size=1.
    DDP mode (launched via torchrun): initialises the NCCL process group and
    assigns each process its own CUDA device.
    """
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"Single-process mode. GPU: {torch.cuda.get_device_name(0)}")
            if n > 1:
                print(f"  ({n} GPUs available — launch with torchrun for multi-GPU)")
            return torch.device("cuda"), 0, 1
        print("Single-process mode. Using CPU.")
        return torch.device("cpu"), 0, 1

    local_rank = int(local_rank_str)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if local_rank == 0:
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(world_size))
        print(f"DDP mode: {world_size} GPUs ({names})")
    return device, local_rank, world_size


def wrap_model(model: nn.Module, device: torch.device, local_rank: int, world_size: int) -> nn.Module:
    """Wrap with DistributedDataParallel when world_size > 1, otherwise no-op."""
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    train_losses: list[float],
    val_losses: list[float],
    output_dir: str | Path,
) -> Path:
    """Save model + training state to a dated checkpoint file.

    In DDP mode only rank 0 saves; other ranks return None immediately.
    The model is always unwrapped before saving so checkpoints are identical
    whether trained on 1 or N GPUs.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    path  = output_dir / f"{today}-{cfg.name}.pt"

    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "model_state_dict":     raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses":         train_losses,
        "val_losses":           val_losses,
        "experiment_config":    cfg.to_dict(),
    }, path)

    print(f"Saved checkpoint: {path}")
    return path
