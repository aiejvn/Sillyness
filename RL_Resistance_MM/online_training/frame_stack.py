"""frame_stack.py — Rolling deque of preprocessed frames.

Maintains a fixed-length window of the most recent frames, preprocessed
with the same TRANSFORM used during offline training (Grayscale → ToTensor
→ Normalize to [-1, 1]).  Each push replaces the oldest frame.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Import the canonical preprocessing transform from the offline pipeline.
# IMPORTANT: TRANSFORM does NOT include Resize — we apply it here separately.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
from preprocess_frames import TRANSFORM


class FrameStack:
    """Rolling deque of (1, H, W) float32 preprocessed frames.

    Args:
        stack_size: Number of frames to keep (e.g. 20).
        img_size:   Target spatial resolution (applied before TRANSFORM).
    """

    def __init__(self, stack_size: int, img_size: int):
        self.stack_size = stack_size
        self._resize = transforms.Resize((img_size, img_size), antialias=True)
        # Store each frame as (1, H, W) float32 numpy — avoids repeated conversion
        self._deque: deque[np.ndarray] = deque(maxlen=stack_size)

    def push(self, pil_image: Image.Image) -> None:
        """Preprocess one RGB PIL image and push it onto the stack.

        Applies Resize → Grayscale → ToTensor → Normalize (matching offline training).
        The oldest frame is automatically dropped when the deque is full.
        """
        resized = self._resize(pil_image)           # still PIL, (H, W, 3)
        tensor = TRANSFORM(resized)                 # (1, H, W) float32 in [-1, 1]
        self._deque.append(tensor.numpy())

    def is_ready(self) -> bool:
        """True once the deque holds a full stack of frames."""
        return len(self._deque) == self.stack_size

    def get_stack(self) -> np.ndarray:
        """Return the current stack as a (stack_size, H, W) float32 array."""
        # Concatenate along axis 0: each frame is (1, H, W) → result (S, H, W)
        return np.concatenate(list(self._deque), axis=0)

    def get_stack_tensor(self, device: torch.device) -> torch.Tensor:
        """Return (1, stack_size, H, W) float32 Tensor ready for model inference."""
        arr = self.get_stack()   # (S, H, W)
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    def reset(self) -> None:
        """Clear the deque.  Call at episode boundaries."""
        self._deque.clear()
