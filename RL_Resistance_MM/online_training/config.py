"""config.py — Hyperparameters for online RL fine-tuning.

All online-specific settings live here.  The ExperimentConfig (architecture,
img_size, stack_size, etc.) is loaded from the pretrained checkpoint and is
NOT duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
from reward import RewardWeights


@dataclass
class OnlineConfig:
    # ── Replay buffer ────────────────────────────────────────────────────────
    # Memory budget: at img_size=224, stack_size=20, float32:
    #   each state ≈ 20 × 224 × 224 × 4 bytes ≈ 4 MB
    #   5_000 capacity → states + next_states ≈ 4 GB RAM
    # Reduce to 2_000 if memory constrained, or use deep_q_v1 (84×84 → ~0.4 MB/state)
    buffer_capacity: int = 5_000
    min_buffer_size: int = 500        # steps before first gradient update

    # ── Training cadence ─────────────────────────────────────────────────────
    update_every_n_steps: int = 4         # gradient update frequency (steps)
    target_update_every_n_steps: int = 500  # hard target-network copy (steps)
    checkpoint_every_n_steps: int = 1_000

    # ── Epsilon-greedy exploration ───────────────────────────────────────────
    # Start low since the model is pretrained — mostly exploit, a little explore
    epsilon_start: float = 0.10
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 20_000

    # ── Reward ───────────────────────────────────────────────────────────────
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    max_reward: float = 10.0       # clip reward to [-max, max] before storing

    # ── Frame capture ────────────────────────────────────────────────────────
    # 10 Hz is intentionally conservative — Tesseract OCR takes 50-200 ms/call.
    # The OCR pipeline runs in a background thread so the main loop stays on pace,
    # but rewards lag the actions by ~1 frame at this rate.
    capture_fps: int = 10
    resolution: tuple = (1920, 1080)

    # ── Action execution ─────────────────────────────────────────────────────
    # Q-value threshold above which a discrete key is considered "active"
    action_threshold: float = 0.5
    # Suppress mouse movement heads (mouse_x/y/dx/dy) during fine-tuning to
    # avoid chaotic exploration while discrete-key policy is still converging.
    suppress_mouse_movement: bool = True
    # Keys to never send to the game (by output column name)
    blacklisted_keys: tuple = ("escape",)

    # ── Paths ────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "online_training/checkpoints"
    pretrained_checkpoint: str = ""   # set by CLI --checkpoint arg
