"""action_executor.py — Convert Q-values to keyboard/mouse actions via pynput.

Maintains stateful key hold/release (the game sees clean long-presses, not
rapid taps).  Mouse movement heads are always suppressed during fine-tuning;
mouse button heads are suppressed by default (controlled by OnlineConfig).

⚠ Windows note: pynput requires Administrator privileges to inject input into
  game processes protected by UAC.  Run the training script as Administrator.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

try:
    from pynput import keyboard as pynput_keyboard
    from pynput import mouse as pynput_mouse
    from pynput.keyboard import Key, KeyCode
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False
    logging.warning("pynput not installed — ActionExecutor will log actions only (no actual keypresses).")

logger = logging.getLogger(__name__)

# ── Key mapping ───────────────────────────────────────────────────────────────
# Maps output_column name → pynput Key or KeyCode.
# Columns whose value is None are handled separately (mouse) or suppressed.
_KEY_MAP: dict[str, object] = {}

if _PYNPUT_AVAILABLE:
    _KEY_MAP = {
        "key_w":     KeyCode.from_char("w"),
        "key_a":     KeyCode.from_char("a"),
        "key_s":     KeyCode.from_char("s"),
        "key_d":     KeyCode.from_char("d"),
        "key_q":     KeyCode.from_char("q"),
        "key_e":     KeyCode.from_char("e"),
        "key_r":     KeyCode.from_char("r"),
        "key_f":     KeyCode.from_char("f"),
        "key_v":     KeyCode.from_char("v"),
        "key_m":     KeyCode.from_char("m"),
        "key_1":     KeyCode.from_char("1"),
        "key_2":     KeyCode.from_char("2"),
        "key_3":     KeyCode.from_char("3"),
        "key_4":     KeyCode.from_char("4"),
        "key_space": Key.space,
        "key_up":    Key.up,
        "key_down":  Key.down,
        "key_left":  Key.left,
        "key_right": Key.right,
    }

# Mouse button columns (handled via pynput.mouse separately)
_MOUSE_BUTTON_COLS = {"mouse_left", "mouse_middle", "mouse_right"}
# Mouse axis columns (always suppressed — no physical mouse movement)
_MOUSE_AXIS_COLS   = {"mouse_x", "mouse_y", "mouse_dx", "mouse_dy"}


class ActionExecutor:
    """Converts a Q-value vector into keyboard/mouse actions.

    Args:
        output_columns:          Ordered list of action head names
                                 (must match the network's output_columns).
        action_threshold:        Q-value threshold above which a key is pressed.
        suppress_mouse_movement: If True (default), mouse axes are never sent.
        blacklisted_keys:        Column names to never press (e.g. "escape").
    """

    def __init__(
        self,
        output_columns: Sequence[str],
        action_threshold: float = 0.5,
        suppress_mouse_movement: bool = True,
        blacklisted_keys: tuple[str, ...] = ("escape",),
    ):
        self.output_columns     = list(output_columns)
        self.action_threshold   = action_threshold
        self.suppress_mouse     = suppress_mouse_movement
        self.blacklisted        = set(blacklisted_keys)

        self._pressed_keys: set[str] = set()   # currently held keyboard keys

        if _PYNPUT_AVAILABLE:
            self._kb    = pynput_keyboard.Controller()
            self._mouse = pynput_mouse.Controller()
        else:
            self._kb    = None
            self._mouse = None

    def execute(self, q_values: np.ndarray) -> np.ndarray:
        """Apply Q-values as keypresses and return the action_vec for the replay buffer.

        Args:
            q_values: (action_dim,) float32 array from model forward pass.

        Returns:
            action_vec: (action_dim,) float32 — 1.0 for active discrete keys,
                        0.0 for inactive keys and all mouse axes.
        """
        action_vec = np.zeros(len(self.output_columns), dtype=np.float32)

        for i, col in enumerate(self.output_columns):
            # Mouse axes: always skip (no physical input, no reward signal stored)
            if col in _MOUSE_AXIS_COLS:
                continue

            # Mouse buttons: suppressed by default
            if col in _MOUSE_BUTTON_COLS:
                continue

            # Blacklisted keys: never press
            if col in self.blacklisted:
                continue

            # Discrete key: press/release based on Q threshold
            is_active = float(q_values[i]) > self.action_threshold
            if is_active:
                action_vec[i] = 1.0
                self._press_key(col)
            else:
                self._release_key(col)

        return action_vec

    def release_all(self) -> None:
        """Release every currently held key.  Call on cleanup / episode end."""
        for col in list(self._pressed_keys):
            self._release_key(col)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _press_key(self, col: str) -> None:
        if col in self._pressed_keys:
            return   # already held — pynput does not need a repeat press

        pynput_key = _KEY_MAP.get(col)
        if pynput_key is None:
            return

        if self._kb is not None:
            try:
                self._kb.press(pynput_key)
                self._pressed_keys.add(col)
                logger.debug(f"pressed {col}")
            except Exception as e:
                logger.warning(f"Failed to press {col}: {e}")
        else:
            # Dry-run mode (pynput unavailable)
            self._pressed_keys.add(col)
            logger.debug(f"[DRY RUN] press {col}")

    def _release_key(self, col: str) -> None:
        if col not in self._pressed_keys:
            return

        pynput_key = _KEY_MAP.get(col)
        if pynput_key is None:
            self._pressed_keys.discard(col)
            return

        if self._kb is not None:
            try:
                self._kb.release(pynput_key)
                self._pressed_keys.discard(col)
                logger.debug(f"released {col}")
            except Exception as e:
                logger.warning(f"Failed to release {col}: {e}")
        else:
            self._pressed_keys.discard(col)
            logger.debug(f"[DRY RUN] release {col}")
