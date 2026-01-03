import base64
import io
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading

from schemas import AudioBuffer

# ==================== AUDIO CAPTURE ====================

class AudioCapture:
    """Captures system audio with rolling buffer"""
    
    def __init__(self, sample_rate=48000, channels=2, buffer_seconds=1.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_seconds = buffer_seconds
        self.buffer_samples = int(sample_rate * buffer_seconds)
        self.audio_buffer = np.zeros((self.buffer_samples, channels), dtype=np.float32)
        self.buffer_index = 0
        self.is_recording = False
        self.stream = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start audio capture stream"""
        def callback(indata, frames, time_info, status):
            with self.lock:
                # Add new audio data to rolling buffer
                end_idx = self.buffer_index + frames
                if end_idx <= self.buffer_samples:
                    self.audio_buffer[self.buffer_index:end_idx] = indata
                else:
                    # Wrap around circular buffer
                    first_part = self.buffer_samples - self.buffer_index
                    self.audio_buffer[self.buffer_index:] = indata[:first_part]
                    self.audio_buffer[:frames - first_part] = indata[first_part:]
                self.buffer_index = (self.buffer_index + frames) % self.buffer_samples
                
        self.stream = sd.InputStream(
            callback=callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype='float32'
        )
        self.stream.start()
        self.is_recording = True
        logging.info(f"Audio capture started: {self.sample_rate}Hz, {self.channels} channels")
        
    def stop(self):
        """Stop audio capture"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            
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