import base64
import json
import logging
import os
import queue
import threading
import time

from schemas import CaptureConfig, InputState
from input_capture import InputCapture
# from audio_capture import AudioCapture
from screen_capture import ScreenCapture

# ==================== MAIN CAPTURE CONTROLLER ====================

class GameCaptureController:
    """Main controller coordinating all capture modules"""
    
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.session_id = config.session_id
        self.output_dir = os.path.join(config.output_dir, self.session_id)
        
        # Initialize capture modules
        self.input_capture = InputCapture(poll_rate=config.input_poll_rate)
        # self.audio_capture = AudioCapture(
        #     sample_rate=config.audio_sample_rate,
        #     channels=config.audio_channels,
        #     buffer_seconds=config.audio_buffer_seconds
        # )
        self.screen_capture = ScreenCapture(
            target_fps=config.capture_fps,
            resolution=config.screen_resolution
        )
        
        # State management
        self.is_capturing = False
        self.capture_start_time = 0
        self.frame_data_buffer = []
        self.save_thread = None
        self.save_queue = queue.Queue(maxsize=100)
        self.capture_thread = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames", "raw"), exist_ok=True)
        # os.makedirs(os.path.join(self.output_dir, "audio"), exist_ok=True) # don't double-save audio, for now
        os.makedirs(os.path.join(self.output_dir, "screens"), exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        log_file = os.path.join(self.output_dir, "capture.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start all capture modules"""
        if self.is_capturing:
            self.logger.warning("Capture already running")
            return
            
        self.logger.info(f"Starting capture session: {self.session_id}")
        
        # Save configuration
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
            
        # Start capture modules
        if not self.input_capture.is_running:
            self.input_capture.start()

        # self.audio_capture.start()
        self.screen_capture.start()

        # Start save thread
        self.save_thread = threading.Thread(target=self._save_worker)
        self.save_thread.start()

        # Mark capturing started (capture loop runs in caller thread)
        self.is_capturing = True
        self.capture_start_time = time.perf_counter()
        
        self.logger.info("All capture modules started")
        
    def stop(self):
        """Stop all capture modules

        If called with keep_input=True (used when toggling off), keep `input_capture` running
        so the hotkey listener can restart capture later.
        """
        # Default: stop everything
        self.is_capturing = False
        
        self.logger.info("Stopping capture modules...")
        
        # Stop capture modules
        self.screen_capture.stop()
        # self.audio_capture.stop()
        # Stop input capture
        try:
            self.input_capture.stop()
        except Exception:
            pass
        
        # Signal save thread to stop
        self.save_queue.put(None)
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
        # No background capture thread to join
            
        # Save any remaining data
        self._save_remaining_data()
        
        self.logger.info(f"Capture session {self.session_id} stopped")
        
    def capture_loop(self):
        """Main capture loop - run this in main thread or separate thread"""
        last_frame_time = self.capture_start_time
        
        while self.is_capturing:
            try:
                # Get the latest screen frame (non-blocking)
                frame = self.screen_capture.get_latest_frame(block=False)
                
                if frame:
                    # Get synchronized audio
                    # audio_buffer = self.audio_capture.get_audio_buffer(frame.timestamp)
                    
                    # Get input events since last frame
                    input_events = self.input_capture.get_events_since(last_frame_time)
                    
                    # Get current input state
                    current_input_state = self.input_capture.get_current_state()
                    
                    # Create combined data frame
                    frame_data = self._create_frame_data(
                        frame, input_events, current_input_state
                    )
                    
                    # Save to queue for background writing
                    self.save_queue.put(frame_data)
                    
                    last_frame_time = frame.timestamp
                    
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Recover from error
                
    def _create_frame_data(self, frame, input_events, input_state):
        """Create a complete frame data object"""
        # Convert screen frame to base64
        frame_jpeg_base64 = frame.to_jpeg_base64(self.config.compression_quality)
        
        # Process input events
        keyboard_events = []
        mouse_events = []
        
        for event_type, event in input_events:
            if 'mouse' in event_type:
                mouse_events.append({
                    'type': event_type,
                    'x': event.x,
                    'y': event.y,
                    'dx': event.dx,
                    'dy': event.dy,
                    'buttons': event.buttons,
                    'timestamp_offset': event.timestamp - frame.timestamp
                })
        
        # Create frame data structure
        frame_data = {
            "frame_id": f"{self.session_id}_frame_{frame.frame_number:06d}",
            "session_id": self.session_id,
            "monotonic_timestamp": frame.timestamp - self.capture_start_time,
            "frame_number": frame.frame_number,
            
            "visual_raw": {
                "full_screen": {
                    "format": f"jpeg_{self.config.compression_quality}",
                    "data_base64": frame_jpeg_base64,
                    "width": frame.image.shape[1],
                    "height": frame.image.shape[0]
                },
                "regions_of_interest": {}
            },
            
            # "audio_raw": {
            #     "format": "wav",  # Would be "opus" in production
            #     "duration_ms": audio_buffer.duration * 1000,
            #     "data_base64": audio_buffer.data,
            #     "timestamp_offset": audio_buffer.timestamp - frame.timestamp
            # },
            
            "input_raw": {
                "keyboard": {
                    "keys_pressed": [k for k, v in input_state['keyboard'].items() 
                                    if v == InputState.PRESS or v == InputState.HOLD],
                    "key_state_changes": keyboard_events
                },
                "mouse": {
                    "position": input_state['mouse']['position'],
                    "movement_events": mouse_events,
                    "buttons_current": input_state['mouse']['buttons']
                }
            },
            
            "synchronization_marks": {
                "frame_number": frame.frame_number,
                "system_time": time.time(),
                "audio_peak_frequency": 0  # Would be calculated in production
            }
        }
        
        return frame_data
    
    def _save_worker(self):
        """Background thread for saving data to disk"""
        while True:
            try:
                frame_data = self.save_queue.get(timeout=1.0)
                
                # None is our stop signal
                if frame_data is None:
                    self.logger.info("Save worker received stop signal")
                    break
                    
                # Save frame data to JSON
                frame_filename = f"frame_{frame_data['frame_number']:06d}.json"
                frame_path = os.path.join(self.output_dir, "frames", "raw", frame_filename)
                
                with open(frame_path, 'w') as f:
                    json.dump(frame_data, f, indent=2)
                    
                # Save screen image separately (for easy viewing)
                screen_filename = f"frame_{frame_data['frame_number']:06d}.jpg"
                screen_path = os.path.join(self.output_dir, "screens", screen_filename)
                
                # Decode and save image
                img_data = base64.b64decode(frame_data['visual_raw']['full_screen']['data_base64'])
                with open(screen_path, 'wb') as f:
                    f.write(img_data)
                    
                # Periodic logging
                if frame_data['frame_number'] % 100 == 0:
                    self.logger.info(f"Saved frame {frame_data['frame_number']}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in save worker: {e}")
                
    def _save_remaining_data(self):
        """Save any data remaining in buffers"""
        try:
            # Save remaining frames
            while True:
                frame = self.screen_capture.get_latest_frame(block=False)
                if not frame:
                    break
                    
                # Process and save the frame
                audio_buffer = self.audio_capture.get_audio_buffer(frame.timestamp)
                input_events = self.input_capture.get_events_since(frame.timestamp - 0.1)
                current_input_state = self.input_capture.get_current_state()
                
                frame_data = self._create_frame_data(
                    frame, audio_buffer, input_events, current_input_state
                )
                
                frame_path = os.path.join(self.output_dir, "frames", "raw", 
                                         f"frame_{frame.frame_number:06d}.json")
                with open(frame_path, 'w') as f:
                    json.dump(frame_data, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error saving remaining data: {e}")

    
            
    def get_session_info(self):
        """Get information about current capture session"""
        return {
            "session_id": self.session_id,
            "output_dir": self.output_dir,
            "is_capturing": self.is_capturing,
            "frames_captured": self.screen_capture.frame_count,
            "duration": time.perf_counter() - self.capture_start_time if self.capture_start_time > 0 else 0
        }

# ==================== MAIN ====================

def main():
    """Example usage of the capture system"""
    
    # Configuration
    config = CaptureConfig(
        screen_resolution=(1920, 1080),
        capture_fps=60,
        audio_sample_rate=48000,
        output_dir="./re_resistance_captures",
        session_id="session_004"
    )
    
    # Create controller
    controller = GameCaptureController(config)
    
    try:
        # Delay before starting recording (allow user to switch to game)
        print("Starting capture in 5 seconds...")
        time.sleep(5.0)

        # Start capture
        controller.start()
        
        print(f"Capture started. Session ID: {controller.session_id}")
        print("Press Ctrl+C to stop capture...")
        
        # Run capture loop in main thread
        controller.capture_loop()
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        controller.stop()
        
        # Print session summary
        info = controller.get_session_info()
        print(f"\nCapture session completed:")
        print(f"  Session ID: {info['session_id']}")
        print(f"  Frames captured: {info['frames_captured']}")
        print(f"  Duration: {info['duration']:.2f} seconds")
        print(f"  Data saved to: {info['output_dir']}")

if __name__ == "__main__":
    main()