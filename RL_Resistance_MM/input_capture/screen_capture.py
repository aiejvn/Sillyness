import cv2
import logging
import numpy as np
import pyautogui
import queue
import threading
import time


from schemas import ScreenFrame

# ==================== SCREEN CAPTURE ====================

class ScreenCapture:
    """Captures screen frames at specified FPS"""
    
    def __init__(self, target_fps=60, resolution=(1920, 1080)):
        self.target_fps = target_fps
        self.target_resolution = resolution
        self.frame_interval = 1.0 / target_fps
        self.is_capturing = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer 30 frames
        self.frame_count = 0
        self.region_definitions = { # TODO: Verify and fix these dimensions...
            'ui_top_bar': (0, 0, 1920, 60),
            'camera_view': (300, 100, 1620, 900),
            'ability_bar': (0, 980, 1920, 1080)
        }
        
    def start(self):
        """Start screen capture in a separate thread"""
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
        logging.info(f"Screen capture started at {self.target_fps}FPS")
        
    def stop(self):
        """Stop screen capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
    def _capture_loop(self):
        """Main capture loop maintaining target FPS"""
        last_capture_time = time.perf_counter()
        
        while self.is_capturing:
            current_time = time.perf_counter()
            elapsed = current_time - last_capture_time
            
            if elapsed >= self.frame_interval:
                try:
                    frame = self._capture_single_frame(current_time)
                    if frame:
                        self.frame_queue.put(frame)
                    last_capture_time = current_time
                except Exception as e:
                    logging.error(f"Error capturing frame: {e}")
                    time.sleep(0.001)  # Prevent tight loop on error
            else:
                # Sleep to maintain FPS without excessive CPU usage
                sleep_time = self.frame_interval - elapsed - 0.001
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
    def _capture_single_frame(self, timestamp):
        """Capture a single screen frame"""
        try:
            # Capture screen
            screenshot = pyautogui.screenshot()
            
            # Convert to numpy array
            frame_np = np.array(screenshot)
            
            # Convert RGB to BGR for OpenCV if needed
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Resize to target resolution
            if frame_np.shape[:2] != self.target_resolution[::-1]:
                frame_np = cv2.resize(frame_np, self.target_resolution)
                
            # Convert back to RGB
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            
            # Create frame object
            frame = ScreenFrame(
                image=frame_np,
                timestamp=timestamp,
                frame_number=self.frame_count
            )
            
            # Crop regions
            frame.crop_regions(self.region_definitions)
            
            self.frame_count += 1
            return frame
            
        except Exception as e:
            logging.error(f"Error in single frame capture: {e}")
            return None
            
    def get_latest_frame(self, block=True, timeout=None):
        """Get the latest captured frame"""
        try:
            return self.frame_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def set_regions(self, regions_dict):
        """Update region definitions for cropping"""
        self.region_definitions = regions_dict