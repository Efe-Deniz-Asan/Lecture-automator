import pyaudio
import wave
import threading
import time

class AudioRecorder:
    def __init__(self, output_filename="lecture_audio.wav", channels=1, rate=48000, chunk=4096, input_device_index=None):
        self.output_filename = output_filename
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
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
                print(f"Audio Status: {status}")
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
            print(f"Audio recording started: {self.output_filename} (Device ID: {self.input_device_index})")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_recording = False

    def _record(self):
        # Deprecated: No longer needed with callback mode
        pass

    def pause_recording(self):
        self.is_paused = True
        print("Audio PAUSED.")

    def resume_recording(self):
        self.is_paused = False
        print("Audio RESUMED.")

    def stop_recording(self):
        if not self.is_recording:
            return
        
        print("Stopping audio recording...")
        self.is_recording = False
        
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        self._save_file()
        print("Audio recording stopped and saved.")

    def _save_file(self):
        if not self.frames:
            print("Warning: No audio frames captured!")
            return
            
        try:
            wf = wave.open(self.output_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            print(f"Audio file saved: {self.output_filename}")
        except Exception as e:
            print(f"Error saving audio file: {e}")
        
    def get_current_duration(self):
        # Calculate duration in seconds based on frames captured
        # Duration = Total Frames * Chunk Size / Sample Rate
        return len(self.frames) * self.chunk / self.rate
