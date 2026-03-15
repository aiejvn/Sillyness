"""trainer.py — Shared training logic for RL Resistance MM.

Importable by both train_q_network.ipynb and train_q_network.py.
Neither file should duplicate any of the logic here.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with PER for space-press frames.

    Args:
        df_valid:    DataFrame already filtered to frames with existing screen files.
        screens_dir: Directory containing frame_NNNNNN.jpg files.
        cfg:         Experiment config (stack_size, img_size, batch_size, etc.)
        seed:        RNG seed for reproducible train/val split.

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
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(all_starts)

    split        = int(n_samples * cfg.train_split)
    train_starts = all_starts[:split]
    val_starts   = all_starts[split:]

    train_ds = ResistanceDataset(df_valid, screens_dir, train_starts, output_columns, transform, cfg.stack_size)
    val_ds   = ResistanceDataset(df_valid, screens_dir, val_starts,   output_columns, transform, cfg.stack_size)

    space_base_rate = df_valid["key_space"].mean()
    print(f"Space-press base rate: {space_base_rate:.2%}  |  PER weight: {cfg.per_space_weight}x")

    train_weights = train_ds.get_sample_weights(space_weight=cfg.per_space_weight)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,          num_workers=0)

    print(f"Samples: {n_samples}  (train: {len(train_ds)}, val: {len(val_ds)})")
    return train_loader, val_loader


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
    for images, actions, targets in tqdm(loader, desc=f"Epoch {epoch}"):
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

    for images, actions, targets in tqdm(loader, desc=f"Eval {epoch}"):
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

    val_loss = total_loss / max(n_batches, 1)
    play_rate = space_presses_hat / max(space_presses_truth, 1)
    print(f"Play Rate: {play_rate:.4f}")

    return val_loss, play_rate


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    train_losses: list[float],
    val_losses: list[float],
    output_dir: str | Path,
) -> Path:
    """Save model + training state to a dated checkpoint file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    path  = output_dir / f"{today}-{cfg.name}.pt"

    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses":         train_losses,
        "val_losses":           val_losses,
        "experiment_config":    cfg.to_dict(),
    }, path)

    print(f"Saved checkpoint: {path}")
    return path
