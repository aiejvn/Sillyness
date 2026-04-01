"""replay_buffer.py — Fixed-capacity circular replay buffer.

Stores (state, action_vec, reward, next_state, done) transitions as preallocated
numpy arrays to avoid per-step memory allocation overhead.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular buffer for experience replay.

    Args:
        capacity:     Maximum number of transitions to store.
        state_shape:  Shape of one state array, e.g. (stack_size, H, W).
        action_dim:   Number of action heads (len(output_columns)).
    """

    def __init__(self, capacity: int, state_shape: tuple, action_dim: int):
        self._capacity = capacity
        self._ptr = 0       # write head
        self._size = 0      # current fill level

        # Preallocate contiguous numpy arrays — avoids GC pressure and fragmentation
        self._states      = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._actions     = np.zeros((capacity, action_dim),   dtype=np.float32)
        self._rewards     = np.zeros(capacity,                  dtype=np.float32)
        self._dones       = np.zeros(capacity,                  dtype=np.float32)

    def push(
        self,
        state:      np.ndarray,   # (stack_size, H, W) float32
        action_vec: np.ndarray,   # (action_dim,) float32
        reward:     float,
        next_state: np.ndarray,   # (stack_size, H, W) float32
        done:       bool,
    ) -> None:
        self._states[self._ptr]      = state
        self._next_states[self._ptr] = next_state
        self._actions[self._ptr]     = action_vec
        self._rewards[self._ptr]     = reward
        self._dones[self._ptr]       = float(done)

        self._ptr  = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch of transitions.

        Returns:
            (states, actions, rewards, next_states, dones) — all on `device`.
        """
        idxs = np.random.choice(self._size, size=batch_size, replace=False)

        to_tensor = lambda arr: torch.from_numpy(arr).to(device)

        states      = to_tensor(self._states[idxs])
        next_states = to_tensor(self._next_states[idxs])
        actions     = to_tensor(self._actions[idxs])
        rewards     = to_tensor(self._rewards[idxs])
        dones       = to_tensor(self._dones[idxs])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self._size
