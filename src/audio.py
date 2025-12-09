import pyaudio
import wave
import threading
import time
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

class AudioRecorder:
    def __init__(self, output_filename="lecture_audio.wav", channels=1, rate=None, chunk=None, input_device_index=None):
        self.output_filename = output_filename
        self.channels = channels
        self.rate = rate if rate is not None else config.audio.sample_rate
        self.chunk = chunk if chunk is not None else config.audio.chunk_size
        self.input_device_index = input_device_index
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.is_paused = False
        self.thread = None
        
    def start_recording(self):
        if self.is_recording:
            return
            
        self.is_recording = True
        self.frames = []
        
        # Callback function for non-blocking audio capture
        def callback(in_data, frame_count, time_info, status):
            if status:
                logger.warning(f"Audio Status: {status}")
            if not self.is_paused:
                self.frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        try:
            self.stream = self.audio.open(format=self.format,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     input_device_index=self.input_device_index,
                                     frames_per_buffer=self.chunk,
                                     stream_callback=callback)
            
            self.stream.start_stream()
            logger.info(f"Audio recording started: {self.output_filename} (Device ID: {self.input_device_index})")
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_recording = False

    def _record(self):
        # Deprecated: No longer needed with callback mode
        pass

    def pause_recording(self):
        self.is_paused = True
        logger.info("Audio PAUSED.")

    def resume_recording(self):
        self.is_paused = False
        logger.info("Audio RESUMED.")

    def stop_recording(self):
        if not self.is_recording:
            return
        
        logger.info("Stopping audio recording...")
        self.is_recording = False
        
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        self._save_file()
        logger.info("Audio recording stopped and saved.")

    def _save_file(self):
        if not self.frames:
            logger.warning("No audio frames captured!")
            return
            
        try:
            wf = wave.open(self.output_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            logger.info(f"Audio file saved: {self.output_filename}")
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
        
    def get_current_duration(self):
        # Calculate duration in seconds based on frames captured
        # Duration = Total Frames * Chunk Size / Sample Rate
        return len(self.frames) * self.chunk / self.rate
