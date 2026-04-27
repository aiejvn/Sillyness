"""action_executor.py — Convert Q-values to keyboard/mouse actions via SendInput.

Uses key_interface.py (ctypes/SendInput) for hardware-level input that reaches
game processes ignoring higher-level APIs (pywinauto, pynput, etc.).

⚠ Windows only.  Run as Administrator if the game process is UAC-protected.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
from key_interface import PressKey, ReleaseKey, SendMouse  

logger = logging.getLogger(__name__)

# ── Scan code table ───────────────────────────────────────────────────────────
# PS/2 Set 1 scan codes — mirrors key_codes in run_agent.py.
_SCAN_MAP: dict[str, int] = {
    "key_w":     0x11,
    "key_a":     0x1E,
    "key_s":     0x1F,
    "key_d":     0x20,
    "key_q":     0x10,
    "key_e":     0x12,
    "key_r":     0x13,
    "key_f":     0x21,
    "key_v":     0x2F,
    "key_m":     0x32,
    "key_1":     0x02,
    "key_2":     0x03,
    "key_3":     0x04,
    "key_4":     0x05,
    "key_space": 0x39,
    "key_up":    0x48,
    "key_down":  0x50,
    "key_left":  0x4B,
    "key_right": 0x4D,
    "escape":    0x01,
}

# Mouse button SendInput down/up flag pairs
_MOUSE_BUTTON_FLAGS: dict[str, tuple[int, int]] = {
    "mouse_left":   (0x0002, 0x0004),
    "mouse_right":  (0x0008, 0x0010),
    "mouse_middle": (0x0020, 0x0040),
}

# Relative movement columns — gated by suppress_mouse_movement
_MOUSE_MOVE_COLS = {"mouse_dx", "mouse_dy"}
# Absolute position columns — always suppressed (require screen-space normalisation)
_MOUSE_ABS_COLS  = {"mouse_x", "mouse_y"}



class ActionExecutor:
    """Converts a Q-value vector into keyboard/mouse actions via SendInput.

    Keyboard keys and mouse buttons use a threshold to decide active/inactive.
    Mouse relative movement (mouse_dx/dy) uses the Q-value directly as a pixel
    delta — suppress_mouse_movement=True (default) disables it during fine-tuning
    while the discrete-key policy converges.
    Absolute mouse position (mouse_x/y) is always suppressed.

    Blacklisted keys have scan codes in _SCAN_MAP but are runtime-gated so the
    model can never trigger them regardless of Q-value (e.g. escape).

    Args:
        output_columns:          Ordered list of action head names
                                 (must match the network's output_columns).
        action_threshold:        Q-value threshold above which a key/button is active.
        suppress_mouse_movement: If True (default), mouse_dx/dy heads are ignored.
        blacklisted_keys:        Column names to never press (e.g. "escape").
    """

    def __init__(
        self,
        output_columns: list[str],
        action_threshold: float = 0.5,
        suppress_mouse_movement: bool = False,
        blacklisted_keys: tuple[str, ...] = ("escape",),
    ):
        self.output_columns      = output_columns
        self.action_threshold    = action_threshold
        self.suppress_mouse_move = suppress_mouse_movement
        self.blacklisted         = set(blacklisted_keys)

        self._pressed_keys:    set[str] = set()
        self._pressed_buttons: set[str] = set()

        col_idx = {c: i for i, c in enumerate(self.output_columns)}
        self._dx_idx = col_idx.get("mouse_dx")
        self._dy_idx = col_idx.get("mouse_dy")

    def execute(self, q_values: np.ndarray) -> np.ndarray:
        """Apply Q-values as input actions and return action_vec for replay buffer.

        Returns:
            (action_dim,) float32 — 1.0 for active discrete inputs, raw Q for mouse axes.
        """
        action_vec = np.zeros(len(self.output_columns), dtype=np.float32)

        for i, col in enumerate(self.output_columns):
            if col in _MOUSE_ABS_COLS or col in self.blacklisted:
                continue

            if col in _MOUSE_MOVE_COLS:
                if not self.suppress_mouse_move:
                    action_vec[i] = float(q_values[i])
                continue

            if col in _MOUSE_BUTTON_FLAGS:
                is_active = float(q_values[i]) > self.action_threshold
                if is_active:
                    action_vec[i] = 1.0
                    self._press_button(col)
                else:
                    self._release_button(col)
                continue

            if col in _SCAN_MAP:
                is_active = float(q_values[i]) > self.action_threshold
                if is_active:
                    action_vec[i] = 1.0
                    self._press_key(col)
                else:
                    self._release_key(col)

        if (not self.suppress_mouse_move
                and self._dx_idx is not None
                and self._dy_idx is not None):
            dx = int(q_values[self._dx_idx])
            dy = int(q_values[self._dy_idx])
            if dx != 0 or dy != 0:
                try:
                    SendMouse(dx, dy, flags=0x0001)  # MOUSEEVENTF_MOVE
                except Exception as e:
                    logger.warning(f"Failed to move mouse: {e}")

        return action_vec

    def release_all(self) -> None:
        """Release all held keys and buttons.  Call on cleanup / episode end."""
        for col in list(self._pressed_keys):
            self._release_key(col)
        for col in list(self._pressed_buttons):
            self._release_button(col)

    # ── Keyboard ─────────────────────────────────────────────────────────────

    def _press_key(self, col: str) -> None:
        if col in self._pressed_keys:
            return
        scan = _SCAN_MAP[col]
        try:
            PressKey(scan)
            self._pressed_keys.add(col)
            logger.debug(f"pressed {col} (0x{scan:02X})")
        except Exception as e:
            logger.warning(f"Failed to press {col}: {e}")

    def _release_key(self, col: str) -> None:
        if col not in self._pressed_keys:
            return
        scan = _SCAN_MAP[col]
        try:
            ReleaseKey(scan)
            self._pressed_keys.discard(col)
            logger.debug(f"released {col} (0x{scan:02X})")
        except Exception as e:
            logger.warning(f"Failed to release {col}: {e}")

    # ── Mouse buttons ─────────────────────────────────────────────────────────

    def _press_button(self, col: str) -> None:
        if col in self._pressed_buttons:
            return
        down_flag, _ = _MOUSE_BUTTON_FLAGS[col]
        try:
            SendMouse(0, 0, flags=down_flag)
            self._pressed_buttons.add(col)
            logger.debug(f"mouse down: {col}")
        except Exception as e:
            logger.warning(f"Failed to press {col}: {e}")

    def _release_button(self, col: str) -> None:
        if col not in self._pressed_buttons:
            return
        _, up_flag = _MOUSE_BUTTON_FLAGS[col]
        try:
            SendMouse(0, 0, flags=up_flag)
            self._pressed_buttons.discard(col)
            logger.debug(f"mouse up: {col}")
        except Exception as e:
            logger.warning(f"Failed to release {col}: {e}")
