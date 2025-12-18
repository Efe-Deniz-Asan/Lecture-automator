# ------------------------------------------------------------------------------
#  Copyright (c) 2025 Efe Deniz Asan
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

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
        self.silent_frames = 0  # Track consecutive silent frames
        self.last_warning_time = 0
        
        # Callback function for non-blocking audio capture
        def callback(in_data, frame_count, time_info, status):
            import struct
            import math
            
            if status:
                logger.warning(f"Audio Status: {status}")
            if not self.is_paused:
                self.frames.append(in_data)
                
                # Calculate audio level (RMS) for silence detection
                try:
                    samples = struct.unpack(f'{len(in_data)//2}h', in_data)
                    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                    
                    # Silence threshold (adjust based on mic sensitivity)
                    if rms < 100:  # Very low audio level
                        self.silent_frames += 1
                        # Warn after ~30 seconds of silence (depends on chunk size and rate)
                        silent_duration = self.silent_frames * self.chunk / self.rate
                        if silent_duration > 30 and (time.time() - self.last_warning_time) > 60:
                            logger.warning(f"Audio appears silent for {silent_duration:.0f}s - check microphone!")
                            print(f"\n⚠️ WARNING: Microphone may be muted or disconnected!")
                            self.last_warning_time = time.time()
                    else:
                        self.silent_frames = 0  # Reset on any sound
                except:
                    pass  # Don't crash if calculation fails
                    
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
