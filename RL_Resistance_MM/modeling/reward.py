"""Reward function for RL Resistance MM.

Computes a scalar reward from per-frame labels produced by the data labelling
pipeline.  All reward components are weighted equally (naive baseline).

Input columns consumed from training_data.csv:
    time_burn_delta   — seconds burned from survivor clock (int, may be empty)
    bio_energy        — current bio energy counter (int, may be empty)
    s{1..4}_health    — survivor health status (red/yellow/green or empty)
    s{1..4}_infection — survivor infection level (none/low/medium/high or empty)
    camera_status     — camera status (online/disabled/neutral or empty)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Container obj for all reward weights.
    By default, uses equal weights for every component (naive baseline)."""
    time_burn: float = 1.0
    bio_efficiency: float = 1.0
    survivor_debuff: float = 1.0
    camera_uptime: float = 1.0


# Mapping survivor health to a reward value.
HEALTH_REWARD = {
    "red": 1.0,       # damaged/downed — best outcome
    "yellow": 0.5,    # debuffed — moderate
    "green": 0.0,     # healthy — no reward
}

# Mapping infection level to a reward value (independent of health).
INFECTION_REWARD = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.25,
    "none": 0.0,
}

# Camera status reward/punishment.
CAMERA_REWARD = {
    "online": 1.0,     # camera active — good
    "neutral": 0.0,    # no info
    "disabled": -1.0,  # camera destroyed — bad
}


def compute_time_burn_reward(time_burn_delta: float | None) -> float:
    """Reward from time burn events.

    Positive delta = survivors gained time (punishment).
    Negative delta = time burned from survivors (reward).
    The raw delta is negated so that burning time yields positive reward.
    """
    if time_burn_delta is None or math.isnan(time_burn_delta):
        return 0.0
    return -time_burn_delta


def compute_bio_efficiency_reward(
    bio_energy: float | None,
    prev_bio_energy: float | None,
    time_burn_delta: float | None,
) -> float:
    """Reward for efficient bio energy usage.

    Spending bio without producing time burn or debuffs is punished.
    Spending bio WITH results is rewarded.
    If bio didn't change, no reward/punishment.
    """
    if bio_energy is None or prev_bio_energy is None:
        return 0.0
    if math.isnan(bio_energy) or math.isnan(prev_bio_energy):
        return 0.0

    bio_spent = prev_bio_energy - bio_energy
    if bio_spent <= 0:
        # Bio regenerated or unchanged — no efficiency signal
        return 0.0

    # Did spending bio produce a result?
    tb = 0.0 if (time_burn_delta is None or math.isnan(time_burn_delta)) else abs(time_burn_delta)
    if tb > 0:
        return 1.0   # spent bio and got time burn — efficient
    return -1.0       # spent bio with no time burn — wasteful


def compute_survivor_debuff_reward(row: dict) -> float:
    """Reward for negative status effects across all 4 survivors.

    Sums health reward + infection reward for each survivor.
    """
    total = 0.0
    for i in range(1, 5):
        health = row.get(f"s{i}_health", "")
        infection = row.get(f"s{i}_infection", "")
        total += HEALTH_REWARD.get(health, 0.0)
        total += INFECTION_REWARD.get(infection, 0.0)
    return total


def compute_camera_reward(camera_status: str | None) -> float:
    """Reward for camera uptime."""
    if not camera_status:
        return 0.0
    return CAMERA_REWARD.get(camera_status, 0.0)


def compute_reward(
    row: dict,
    prev_row: dict | None = None,
    weights: RewardWeights | None = None,
    apply_relu: bool = False
) -> float:
    """Compute the total scalar reward for a single frame.

    Parameters
    ----------
    row : dict
        Current frame's columns from training_data.csv.
    prev_row : dict | None
        Previous frame's columns (needed for bio efficiency delta).
    weights : RewardWeights | None
        Component weights.  Defaults to equal weights.

    Returns
    -------
    float
        Total reward (sum of weighted components).
    """
    if weights is None:
        weights = RewardWeights()

    def _safe_float(val):
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    time_burn_delta = _safe_float(row.get("time_burn_delta"))
    bio_energy = _safe_float(row.get("bio_energy"))
    prev_bio = _safe_float(prev_row.get("bio_energy")) if prev_row else None
    camera_status = row.get("camera_status", "") or None

    r_time = compute_time_burn_reward(time_burn_delta)
    r_bio = compute_bio_efficiency_reward(bio_energy, prev_bio, time_burn_delta)

    if apply_relu:
        r_time = max(0, r_time)
        r_bio = max(0, r_bio)

    r_debuff = compute_survivor_debuff_reward(row)
    r_camera = compute_camera_reward(camera_status)

    return (
        weights.time_burn * r_time
        + weights.bio_efficiency * r_bio
        + weights.survivor_debuff * r_debuff
        + weights.camera_uptime * r_camera
    )


def compute_rewards_for_episode(rows: list[dict], weights: RewardWeights | None = None, apply_relu: bool = False) -> list[float]:
    """Compute rewards for an entire episode (list of frame rows).

    Parameters
    ----------
    rows : list[dict]
        Ordered list of frame rows from training_data.csv.
    weights : RewardWeights | None
        Component weights.

    Returns
    -------
    list[float]
        Per-frame reward values.
    """
    rewards = []
    prev_row = None
    for row in rows:
        rewards.append(compute_reward(row, prev_row=prev_row, weights=weights, apply_relu=apply_relu))
        prev_row = row
    return rewards
