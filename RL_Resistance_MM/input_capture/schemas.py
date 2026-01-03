import pyautogui
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import json
import base64
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import logging
from PIL import Image
import io
import keyboard
import mouse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
from datetime import datetime

# ==================== CONFIGURATION ====================

@dataclass # automatically adds methods like __init__()
class CaptureConfig:
    """Configuration for capture parameters"""
    screen_resolution: tuple = (1920, 1080)  # Target resolution
    capture_fps: int = 60  # Target FPS for screen capture
    audio_sample_rate: int = 48000  # Hz
    audio_channels: int = 2  # Stereo
    audio_buffer_seconds: float = 1.0  # Rolling audio buffer
    input_poll_rate: int = 1000  # Hz
    output_dir: str = "./capture_data"
    session_id: str = None
    compression_quality: int = 85  # JPEG quality (0-100)
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def to_dict(self):
        return asdict(self)

# ==================== DATA STRUCTURES ====================

class InputState(Enum):
    PRESS = "press"
    RELEASE = "release"
    HOLD = "hold"

@dataclass
class KeyEvent:
    key: str
    state: InputState
    timestamp: float
    duration: float = 0.0  # For hold events

@dataclass
class MouseEvent:
    x: int
    y: int
    dx: int = 0
    dy: int = 0
    buttons: tuple = (0, 0, 0)  # Left, Middle, Right
    wheel: tuple = (0, 0)  # dx, dy
    timestamp: float = 0.0

@dataclass
class AudioBuffer:
    data: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float
    duration: float

@dataclass
class ScreenFrame:
    image: np.ndarray  # RGB format
    timestamp: float
    frame_number: int
    regions: Dict[str, np.ndarray] = None  # Pre-cropped regions
    
    def to_jpeg_base64(self, quality=85):
        """Convert image to base64 encoded JPEG"""
        img_pil = Image.fromarray(self.image)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def crop_regions(self, regions_dict):
        """Crop predefined regions from the screen"""
        self.regions = {}
        for name, coords in regions_dict.items():
            x1, y1, x2, y2 = coords
            self.regions[name] = self.image[y1:y2, x1:x2]
        return self.regions