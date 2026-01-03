import base64
import io
import logging
import numpy as np
import pyaudio
import soundfile as sf
import threading

from schemas import AudioBuffer

# ==================== AUDIO CAPTURE ====================

class AudioCapture:
    """Captures system audio with rolling buffer. Optionally collects all frames for debug export. Can target a specific device by name. Uses PyAudio with WASAPI shared mode."""

    def __init__(self, sample_rate=48000, channels=2, buffer_seconds=1.0, debug=True, device_name=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_seconds = buffer_seconds
        self.buffer_samples = int(sample_rate * buffer_seconds)
        self.audio_buffer = np.zeros((self.buffer_samples, channels), dtype=np.float32)
        self.buffer_index = 0
        self.is_recording = False
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.lock = threading.Lock()
        self.debug = debug
        self._debug_frames = [] if debug else None
        self.device_name = device_name
        
    def start(self):
        """Start audio capture stream using PyAudio (WASAPI shared mode if available)"""
        # Print all available audio devices for debug
        logging.info("[DEBUG] Listing all available audio devices:")
        selected_index = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            host_api_name = self.pyaudio_instance.get_host_api_info_by_index(dev['hostApi'])['name']
            logging.info(f"[DEBUG] Device {i}: {dev['name']} (maxInputChannels: {dev['maxInputChannels']}, hostApi: {host_api_name})")
            if self.device_name and self.device_name.lower() in dev['name'].lower() and int(dev['maxInputChannels']) > 0:
                selected_index = i
        if self.device_name and selected_index is None:
            logging.error(f"[DEBUG] Requested device '{self.device_name}' does not support input (maxInputChannels=0) or was not found. Aborting audio capture.")
            raise RuntimeError(f"Requested device '{self.device_name}' does not support input (maxInputChannels=0) or was not found.")
        if selected_index is not None:
            dev_info = self.pyaudio_instance.get_device_info_by_index(selected_index)
            logging.info(f"[DEBUG] Selected device by name: {dev_info['name']} (index {selected_index})")
        else:
            # Fallback: use first available input device with maxInputChannels > 0
            fallback_index = None
            for i in range(self.pyaudio_instance.get_device_count()):
                dev = self.pyaudio_instance.get_device_info_by_index(i)
                if int(dev['maxInputChannels']) > 0:
                    fallback_index = i
                    break
            if fallback_index is not None:
                dev_info = self.pyaudio_instance.get_device_info_by_index(fallback_index)
                logging.info(f"[DEBUG] Using fallback input device: {dev_info['name']} (index {fallback_index})")
            else:
                logging.error("[DEBUG] No valid input device with input channels found. Aborting audio capture.")
                raise RuntimeError("No valid input device with input channels found.")

        max_input_channels = int(dev_info.get('maxInputChannels', 1))
        # Use the minimum of requested channels and device's supported channels
        actual_channels = min(self.channels, max_input_channels)
        if actual_channels < 1:
            actual_channels = 1
        if self.channels != actual_channels:
            logging.warning(f"[DEBUG] Requested {self.channels} channels, but device supports only {max_input_channels}. Using {actual_channels} channel(s).")
        self.channels = actual_channels
        host_api = self.pyaudio_instance.get_host_api_info_by_index(dev_info['hostApi'])['name']
        wasapi = 'wasapi' in host_api.lower()
        logging.info(f"[DEBUG] Device host API: {host_api}")

        def callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.channels)
            with self.lock:
                end_idx = self.buffer_index + frame_count
                if end_idx <= self.buffer_samples:
                    self.audio_buffer[self.buffer_index:end_idx] = audio
                else:
                    first_part = self.buffer_samples - self.buffer_index
                    self.audio_buffer[self.buffer_index:] = audio[:first_part]
                    self.audio_buffer[:frame_count - first_part] = audio[first_part:]
                self.buffer_index = (self.buffer_index + frame_count) % self.buffer_samples
                if self.debug:
                    self._debug_frames.append(np.copy(audio))
            return (None, pyaudio.paContinue)

        # WASAPI shared mode: set 'wasapi' specific arguments if available
        stream_kwargs = dict(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=dev_info['index'],
            frames_per_buffer=1024,
            stream_callback=callback
        )
        # Do NOT set wasapi_loopback for input (shared mode) devices; only needed for output loopback
        self.stream = self.pyaudio_instance.open(**stream_kwargs)
        self.stream.start_stream()
        self.is_recording = True
        logging.info(f"Audio capture started: {self.sample_rate}Hz, {self.channels} channels (PyAudio, WASAPI shared mode: {wasapi})")
        
    def stop(self, debug_output_path=None):
        """Stop audio capture. If debug enabled, merge and save all collected frames to a WAV file."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.is_recording = False
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        if self.debug and self._debug_frames:
            try:
                audio_data = np.concatenate(self._debug_frames, axis=0)
                # Amplify by 10x, clip to [-1.0, 1.0] to avoid overflow
                amplified = np.clip(audio_data * 150, -1.0, 1.0)
                audio_int16 = (amplified * 32767).astype(np.int16)
                out_path = debug_output_path or "debug_audio_capture.wav"
                sf.write(out_path, audio_int16, self.sample_rate, format='WAV')
                logging.info(f"[DEBUG] Audio frames merged, amplified, and saved to {out_path}")
            except Exception as e:
                logging.exception(f"[DEBUG] Failed to merge/save debug audio: {e}")
            self._debug_frames = []
            
    def get_audio_buffer(self, timestamp):
        """Get the current audio buffer as base64 encoded Opus"""
        with self.lock:
            # Get the most recent audio data (circular buffer)
            if self.buffer_index == 0:
                audio_data = self.audio_buffer
            else:
                audio_data = np.vstack([
                    self.audio_buffer[self.buffer_index:],
                    self.audio_buffer[:self.buffer_index]
                ])
            
            # Convert to int16 for Opus encoding
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # In production, you'd use opus encoding here
            # For this example, we'll use WAV format
            buffer = io.BytesIO()
            sf.write(buffer, audio_int16, self.sample_rate, format='WAV')
            audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return AudioBuffer(
                data=audio_base64,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=timestamp,
                duration=self.buffer_seconds
            )