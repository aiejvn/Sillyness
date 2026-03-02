"""
run_agent.py — Run a trained DecomposedQNetwork against the live game.

Captures the screen each step, stacks 20 grayscale frames, runs inference,
and translates Q-value outputs into keyboard presses via key_interface.py.
"""

# ============================================================
# USER CONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\Py Torch\Documents\GitHub\Sillyness\RL_Resistance_MM\modeling\checkpoints\2026-03-01-deep_q_v1.pt"

# Window title/class used to locate the game process via pywinauto.
RESISTANCE_HANDLE = "BIOHAZARD_RESISTANCE"

# Screen region to capture. Default: full 1080p display.
CAPTURE_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# Q-value threshold: heads above this value trigger a key press.
DECISION_THRESHOLD = 0.0

# Seconds to wait between inference steps (~20 steps/s at 0.05).
STEP_DELAY = 0.05
# ============================================================

import time
from collections import deque
from time import sleep
from typing import List

import mss
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from pywinauto import Application
from pywinauto.findwindows import find_windows

from key_interface import PressKey, ReleaseKey, KeyPress

# ── Model constants ──────────────────────────────────────────
IMG_SIZE   = 84
STACK_SIZE = 20

# ── Key scan codes (Quartz Events / DirectInput) ─────────────
key_codes = {
    # Numbers
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05,
    "5": 0x06, "6": 0x07, "7": 0x08, "8": 0x09,
    "9": 0x0A, "0": 0x0B,

    # Top letter row
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14,
    "y": 0x15, "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19,

    # Home row
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22,
    "h": 0x23, "j": 0x24, "k": 0x25, "l": 0x26,

    # Bottom row
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F,
    "b": 0x30, "n": 0x31, "m": 0x32,

    # Special
    "space":     0x39,
    "caps lock": 0x3A,

    # Function keys
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E,
    "f5": 0x3F, "f6": 0x40, "f7": 0x41, "f8": 0x42,
    "f9": 0x43, "f10": 0x44, "f11": 0x57, "f12": 0x58,

    # Arrow keys
    "up":    0x48,
    "down":  0x50,
    "left":  0x4B,
    "right": 0x4D,
}

# Maps model output column names -> key_codes keys.
# mouse_left / mouse_middle / mouse_right are omitted (keyboard-only interface).
MODEL_ACTION_MAP = {
    "key_1": "1", "key_2": "2", "key_3": "3", "key_4": "4",
    "key_q": "q", "key_w": "w", "key_e": "e", "key_r": "r",
    "key_a": "a", "key_s": "s", "key_d": "d",
    "key_f": "f", "key_v": "v", "key_m": "m",
    "key_space": "space",
    "key_up":    "up",
    "key_down":  "down",
    "key_left":  "left",
    "key_right": "right",
}


# ── Model definition (matches training notebook cell-11) ─────
class DecomposedQNetwork(nn.Module):
    """Q-function decomposed into independent per-action values.

    Input:  (B, stack_size, 84, 84) stacked grayscale frames
    Output: (B, num_outputs) Q-value score per action dimension
    """

    def __init__(self, num_outputs: int, stack_size: int = 4):
        super().__init__()
        self.num_outputs = num_outputs

        self.cnn = nn.Sequential(
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4, padding=1), nn.LazyBatchNorm2d(),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.LazyBatchNorm2d(),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LazyBatchNorm2d(),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.state_fc = nn.Sequential(
            nn.LazyLinear(4096), nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(256), nn.LazyBatchNorm1d(),
            nn.ReLU(),
        )

        self.action_heads = nn.Linear(256, num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn(image)
        x = x.view(x.size(0), -1)
        state = self.state_fc(x)
        return self.action_heads(state)


# ── Window connection ────────────────────────────────────────
def connect_to_game(handle: str = RESISTANCE_HANDLE) -> Application:
    """Locate the game window by title and return a connected Application.

    Raises RuntimeError if no matching window is found.
    """
    hwnds: List[int] = find_windows(title_re=handle)
    if not hwnds:
        raise RuntimeError(f"No window found matching title: {handle!r}")
    return Application(backend="win32").connect(handle=hwnds[0])


# ── Agent ────────────────────────────────────────────────────
class GameAgent:
    def __init__(self, model_path: str, device: torch.device, threshold: float = 0.0,
                 handle: str = RESISTANCE_HANDLE):
        self.device    = device
        self.threshold = threshold

        # Locate and connect to the game window.
        self.app         = connect_to_game(handle)
        self.game_window = self.app.window(title_re=handle)

        ckpt = torch.load(model_path, map_location=device)
        num_outputs      = ckpt["NUM_OUTPUTS"]
        self.key_columns = ckpt["key_columns"]  # 22 binary key names, in output order

        self.model = DecomposedQNetwork(num_outputs=num_outputs, stack_size=STACK_SIZE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.model.to(device)

        # Deque of preprocessed (1, 84, 84) tensors; fills up over the first STACK_SIZE steps
        self.frame_stack: deque = deque(maxlen=STACK_SIZE)

        # Scan codes of keys currently held down
        self.pressed: set = set()

        # Must match training preprocessing (notebook cell-9)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.sct = mss.mss()

    def capture_frame(self) -> Image.Image:
        raw = self.sct.grab(CAPTURE_REGION)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)  # (1, 84, 84)

    def update_keys(self, q_values: torch.Tensor):
        """Press or release each mapped key based on its Q-value vs threshold."""
        for i, col in enumerate(self.key_columns):
            key_name = MODEL_ACTION_MAP.get(col)
            if key_name is None:
                continue  # mouse buttons — not handled here

            scan_code   = key_codes[key_name]
            should_press = q_values[i].item() > self.threshold

            if should_press and scan_code not in self.pressed:
                PressKey(scan_code)
                self.pressed.add(scan_code)
            elif not should_press and scan_code in self.pressed:
                ReleaseKey(scan_code)
                self.pressed.discard(scan_code)

    def release_all(self):
        for scan_code in list(self.pressed):
            ReleaseKey(scan_code)
        self.pressed.clear()

    def step(self):
        frame = self.preprocess(self.capture_frame())  # (1, 84, 84)
        self.frame_stack.append(frame)

        # Pad stack to STACK_SIZE by repeating the earliest available frame
        stack = list(self.frame_stack)
        while len(stack) < STACK_SIZE:
            stack.insert(0, stack[0])

        state = torch.cat(stack, dim=0).unsqueeze(0).to(self.device)  # (1, 20, 84, 84)

        with torch.no_grad():
            q_values = self.model(state)[0]  # (26,)

        self.update_keys(q_values)

    def run(self):
        # Bring the game window to the foreground before starting the loop.
        self.game_window.set_focus()
        sleep(0.5)  # brief pause so the OS registers the focus change

        try:
            while True:
                self.step()
                time.sleep(STEP_DELAY)
        except KeyboardInterrupt:
            pass
        finally:
            self.release_all()
            print("Agent stopped. All keys released.")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from: {MODEL_PATH}")

    print(f"Connecting to game window: {RESISTANCE_HANDLE!r}")
    agent = GameAgent(MODEL_PATH, device, threshold=DECISION_THRESHOLD, handle=RESISTANCE_HANDLE)
    print(f"Model loaded. Running inference every {STEP_DELAY}s. Press Ctrl+C to stop.")

    agent.run()
