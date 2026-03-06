"""networks.py — Network architecture definitions for RL Resistance MM.

Single source of truth for all network classes.
Both the training notebook and demo/run_agent.py import from here.

Hierarchy:
  DecomposedQNetwork (abstract base)
    └── DQN_V1  — 3-layer CNN + 2-layer FC, LazyBatchNorm throughout

To add a new architecture: subclass DecomposedQNetwork, implement forward(),
then register it in experiment._NETWORK_REGISTRY.
"""

import torch
import torch.nn as nn


class DecomposedQNetwork(nn.Module):
    """Abstract base for Q-functions decomposed into independent per-action heads.

    All subclasses share the same interface:
      - __init__(num_outputs, stack_size)
      - forward(image: (B, stack_size, H, W)) -> (B, num_outputs)

    Training only propagates loss through heads whose action was active in that frame.
    """

    def __init__(self, num_outputs: int, stack_size: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.stack_size = stack_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DQN_V1(DecomposedQNetwork):
    """3-layer CNN + 2-layer FC with LazyBatchNorm.

    Input:  (B, stack_size, 84, 84) stacked grayscale frames
    Output: (B, num_outputs) Q-value score per action dimension
    """

    def __init__(self, num_outputs: int, stack_size: int = 4):
        super().__init__(num_outputs, stack_size)

        # --- 3-layer CNN for visual + temporal features ---
        # Input: stack_size x 84 x 84
        self.cnn = nn.Sequential(
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4, padding=1), nn.LazyBatchNorm2d(),  # -> 32 x 20 x 20
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 9 x 9
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 7 x 7
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- State embedding ---
        self.state_fc = nn.Sequential(
            nn.LazyLinear(4096), nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(256), nn.LazyBatchNorm1d(),
            nn.ReLU(),
        )

        # --- Per-action Q heads (binary keys + mouse axes) ---
        self.action_heads = nn.Linear(256, num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn(image)             # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)      # (B, 3136)
        state = self.state_fc(x)        # (B, 256)
        return self.action_heads(state) # (B, num_outputs)


class DQN_V1_Mini(DecomposedQNetwork):
    """Mini debug CNN for iteration.

    Input:  (B, stack_size, 84, 84) stacked grayscale frames
    Output: (B, num_outputs) Q-value score per action dimension
    """

    def __init__(self, num_outputs: int, stack_size: int = 4):
        super().__init__(num_outputs, stack_size)

        # --- 3-layer CNN for visual + temporal features ---
        # Input: stack_size x 84 x 84
        self.cnn = nn.Sequential(
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4, padding=1), nn.LazyBatchNorm2d(),  # -> 32 x 20 x 20
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 9 x 9
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 7 x 7
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- State embedding ---
        self.state_fc = nn.Sequential(
            nn.LazyLinear(256), nn.LazyBatchNorm1d(),
            nn.ReLU(),
        )

        # --- Per-action Q heads (binary keys + mouse axes) ---
        self.action_heads = nn.Linear(256, num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn(image)             # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)      # (B, 3136)
        state = self.state_fc(x)        # (B, 256)
        return self.action_heads(state) # (B, num_outputs)

