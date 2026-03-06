"""experiment.py — Experiment configuration registry for RL Resistance MM.

Usage (training notebook):
    from experiment import REGISTRY, build_model
    cfg = REGISTRY["deep_q_v1"]

Usage (demo/run_agent.py):
    from experiment import ExperimentConfig, build_model
    cfg = ExperimentConfig.from_checkpoint(ckpt)
    model = build_model(cfg)

To add a new experiment:
    1. If new architecture: add subclass to networks.py, add entry to _NETWORK_REGISTRY below.
    2. Add a named ExperimentConfig entry to REGISTRY.
    3. In the notebook: change cfg = REGISTRY["your_new_name"].  That's it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Type

import torch.nn as nn

from networks import DecomposedQNetwork, DQN_V1


# ── Output column definitions ────────────────────────────────────────────────
_KEY_COLUMNS: list[str] = [
    "mouse_left", "mouse_middle", "mouse_right",
    "key_w", "key_a", "key_s", "key_d",
    "key_q", "key_e", "key_r", "key_f", "key_v", "key_m",
    "key_1", "key_2", "key_3", "key_4",
    "key_space", "key_up", "key_down", "key_left", "key_right",
]
_MOUSE_COLUMNS: list[str] = ["mouse_x", "mouse_y", "mouse_dx", "mouse_dy"]
_ALL_OUTPUT_COLUMNS: list[str] = _KEY_COLUMNS + _MOUSE_COLUMNS


# ── Network registry ─────────────────────────────────────────────────────────
# Maps the string stored in ExperimentConfig.network_class to the actual class.
# Add one entry here whenever a new subclass is added to networks.py.
_NETWORK_REGISTRY: dict[str, Type[DecomposedQNetwork]] = {
    "DQN_V1": DQN_V1,
}


# ── Config dataclass ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, self-describing specification for one training experiment.

    frozen=True prevents accidental mutation after construction.
    All fields are typed — no silent KeyError on missing dict keys.
    """
    # Identity
    name: str                           # must match the REGISTRY key exactly

    # Architecture
    network_class: str                  # key into _NETWORK_REGISTRY, e.g. "DQN_V1"
    img_size: tuple[int, int]           # (H, W) fed to the CNN
    stack_size: int                     # frames stacked along the channel dim

    # Action space
    output_columns: tuple[str, ...]     # ordered list of action head names

    # Optimisation
    batch_size: int
    learning_rate: float
    gamma: float                        # discount factor

    # Training schedule
    num_epochs: int
    max_early_stop_epochs: int

    # Loss
    l1_inactive_weight: float           # weight for inactive-head L1 penalty

    # Data
    train_split: float
    starting_frame: int                 # frames before this index are skipped

    @property
    def num_outputs(self) -> int:
        return len(self.output_columns)

    def to_dict(self) -> dict:
        """Serialise to a plain dict safe for torch.save."""
        d = dataclasses.asdict(self)
        d["img_size"] = list(self.img_size)
        d["output_columns"] = list(self.output_columns)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        """Reconstruct from a dict loaded out of a checkpoint.

        All required fields must be present — no try/except, no silent defaults.
        A KeyError here means the checkpoint is genuinely incomplete (a bug that
        should surface immediately, not be papered over).
        """
        d = dict(d)
        d["img_size"] = tuple(d["img_size"])
        d["output_columns"] = tuple(d["output_columns"])
        return cls(**d)

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> ExperimentConfig:
        """Extract the ExperimentConfig embedded in a checkpoint dict."""
        return cls.from_dict(ckpt["experiment_config"])


# ── Factory ──────────────────────────────────────────────────────────────────
def build_model(cfg: ExperimentConfig) -> DecomposedQNetwork:
    """Instantiate the correct network subclass for an experiment config.

    This is the single authoritative place where config -> model mapping happens.
    Both the notebook and run_agent.py call this function.

    Raises KeyError if cfg.network_class is not in _NETWORK_REGISTRY — which
    means someone added a config without registering the class (a clear bug).
    """
    network_cls = _NETWORK_REGISTRY[cfg.network_class]
    return network_cls(num_outputs=cfg.num_outputs, stack_size=cfg.stack_size)


# ── Experiment registry ──────────────────────────────────────────────────────
REGISTRY: dict[str, ExperimentConfig] = {
    "deep_q_v1": ExperimentConfig(
        name="deep_q_v1",
        network_class="DQN_V1",
        img_size=(84, 84),
        stack_size=20,
        output_columns=tuple(_ALL_OUTPUT_COLUMNS),
        batch_size=32,
        learning_rate=1e-4,
        gamma=0.99,
        num_epochs=20,
        max_early_stop_epochs=5,
        l1_inactive_weight=0.1,
        train_split=0.8,
        starting_frame=391,
    ),
    # ── Add new experiments below ────────────────────────────────────────────
    # "deep_q_v2": ExperimentConfig(
    #     name="deep_q_v2",
    #     network_class="DQN_V1",      # or "DQN_V2" once defined in networks.py
    #     img_size=(84, 84),
    #     stack_size=20,
    #     output_columns=tuple(_ALL_OUTPUT_COLUMNS),
    #     batch_size=64,
    #     learning_rate=3e-4,
    #     gamma=0.99,
    #     num_epochs=30,
    #     max_early_stop_epochs=5,
    #     l1_inactive_weight=0.05,
    #     train_split=0.8,
    #     starting_frame=391,
    # ),
}
