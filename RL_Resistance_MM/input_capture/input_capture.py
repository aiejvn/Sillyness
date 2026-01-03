from collections import defaultdict
from collections import defaultdict
import keyboard
import logging
import mouse
import pyautogui
import queue
import threading
import time

from schemas import InputState, MouseEvent

# ==================== INPUT CAPTURE ====================

class InputCapture:
    """Captures keyboard and mouse inputs with high precision"""
    
    def __init__(self, poll_rate=1000):
        self.poll_rate = poll_rate
        self.key_states = defaultdict(lambda: {'state': InputState.RELEASE, 'press_time': 0})
        self.mouse_states = {
            'position': (0, 0),
            'buttons': (0, 0, 0),
            'last_position': (0, 0)
        }
        self.event_queue = queue.Queue(maxsize=10000)
        self.is_running = False
        self.capture_thread = None
        
    def start(self):
        """Start input capture in a separate thread"""
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
        logging.info(f"Input capture started at {self.poll_rate}Hz")
        
    def stop(self):
        """Stop input capture"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
    def _capture_loop(self):
        """Main capture loop running at specified poll rate"""
        last_poll_time = time.perf_counter()
        poll_interval = 1.0 / self.poll_rate
        
        while self.is_running:
            current_time = time.perf_counter()
            elapsed = current_time - last_poll_time
            
            if elapsed >= poll_interval:
                self._poll_inputs(current_time)
                last_poll_time = current_time
            else:
                time.sleep(poll_interval * 0.1)  # Small sleep to prevent CPU overload
                
    def _poll_inputs(self, timestamp):
        """Poll current input states and detect changes"""
        try:
            # Capture mouse
            x, y = pyautogui.position()
            buttons = (
                mouse.is_pressed(button='left'),
                mouse.is_pressed(button='middle'),
                mouse.is_pressed(button='right')
            )
            
            # Detect mouse movement
            last_x, last_y = self.mouse_states['position']
            dx, dy = x - last_x, y - last_y
            
            if dx != 0 or dy != 0:
                mouse_event = MouseEvent(
                    x=x, y=y, dx=dx, dy=dy,
                    buttons=buttons,
                    timestamp=timestamp
                )
                self.event_queue.put(('mouse_move', mouse_event))
            
            # Detect mouse button changes
            for i, (button_name, button_idx) in enumerate([('left', 0), ('middle', 1), ('right', 2)]):
                current_state = buttons[button_idx]
                last_state = self.mouse_states['buttons'][button_idx]
                
                if current_state != last_state:
                    event_type = 'mouse_press' if current_state else 'mouse_release'
                    mouse_event = MouseEvent(
                        x=x, y=y, buttons=buttons,
                        timestamp=timestamp
                    )
                    self.event_queue.put((f'{event_type}_{button_name}', mouse_event))
            
            # Update mouse state
            self.mouse_states = {
                'position': (x, y),
                'buttons': buttons,
                'last_position': (x, y)
            }
            
        except Exception as e:
            logging.error(f"Error polling mouse: {e}")
            
    def get_events_since(self, since_timestamp):
        """Get all input events since a given timestamp"""
        events = []
        while not self.event_queue.empty():
            try:
                event_type, event = self.event_queue.get_nowait() # Async retrieval
                if event.timestamp > since_timestamp:
                    events.append((event_type, event))
            except queue.Empty:
                break
        return events
    
    def get_current_state(self):
        """Get current input state snapshot"""
        return {
            'keyboard': {k: v['state'] for k, v in self.key_states.items()},
            'mouse': self.mouse_states
        }

    def is_key_pressed(self, key: str) -> bool:
        """Return whether a watched key is currently pressed (best-effort)."""
        state = self.key_states.get(key)
        if not state:
            try:
                return keyboard.is_pressed(key)
            except Exception:
                return False
        return state['state'] == InputState.PRESS