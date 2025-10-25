# import numpy as np
# import sounddevice as sd
# import threading
# import queue
# import time
# import wave
# import io
# import tempfile
# import os
# from typing import Optional, Callable, Dict, Any, List, Tuple
# import logging
# from datetime import datetime

# class AudioProcessor:
#     """
#     Audio processing utility for microphone operations and audio handling
#     Supports real-time audio capture and processing for emotion detection
#     """
    
#     def __init__(self, sample_rate: int = 44100, channels: int = 1, dtype=np.float32):
#         """
#         Initialize audio processor
        
#         Args:
#             sample_rate: Audio sample rate in Hz
#             channels: Number of audio channels (1 for mono, 2 for stereo)
#             dtype: Audio data type
#         """
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.dtype = dtype
        
#         # Audio capture parameters
#         self.chunk_duration = 1.0  # Duration of audio chunks in seconds
#         self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
#         # Recording state
#         self.is_recording = False
#         self.audio_queue = queue.Queue(maxsize=50)
#         self.recording_thread = None
#         self.audio_callback = None
        
#         # Audio data storage
#         self.recorded_audio = []
#         self.audio_buffer = np.array([], dtype=self.dtype)
        
#         # Device information
#         self.input_device = None
#         self.output_device = None
        
#         # Initialize logger
#         logging.basicConfig(level=logging.INFO)
#         self.logger = logging.getLogger(__name__)
        
#         # Initialize audio system
#         self._initialize_audio_system()
    
#     def _initialize_audio_system(self):
#         """Initialize audio system and detect devices"""
#         try:
#             # Get available audio devices
#             devices = sd.query_devices()
#             self.logger.info(f"Found {len(devices)} audio devices")
            
#             # Find default input device
#             try:
#                 self.input_device = sd.default.device[0]  # Input device
#                 self.output_device = sd.default.device[1]  # Output device
#                 self.logger.info(f"Using input device {self.input_device}, output device {self.output_device}")
#             except Exception as e:
#                 self.logger.warning(f"Could not determine default devices: {str(e)}")
                
#         except Exception as e:
#             self.logger.error(f"Error initializing audio system: {str(e)}")
    
#     def list_audio_devices(self) -> List[Dict]:
#         """
#         List available audio devices
        
#         Returns:
#             List of dictionaries containing device information
#         """
#         try:
#             devices = sd.query_devices()
#             device_list = []
            
#             for i, device in enumerate(devices):
#                 device_info = {
#                     'index': i,
#                     'name': device['name'],
#                     'max_input_channels': device['max_input_channels'],
#                     'max_output_channels': device['max_output_channels'],
#                     'default_sample_rate': device['default_samplerate'],
#                     'is_input': device['max_input_channels'] > 0,
#                     'is_output': device['max_output_channels'] > 0
#                 }
#                 device_list.append(device_info)
            
#             return device_list
            
#         except Exception as e:
#             self.logger.error(f"Error listing audio devices: {str(e)}")
#             return []
    
#     def set_input_device(self, device_index: Optional[int] = None):
#         """
#         Set input audio device
        
#         Args:
#             device_index: Device index (None for default)
#         """
#         try:
#             if device_index is not None:
#                 devices = sd.query_devices()
#                 if 0 <= device_index < len(devices):
#                     self.input_device = device_index
#                     self.logger.info(f"Set input device to {device_index}: {devices[device_index]['name']}")
#                 else:
#                     self.logger.error(f"Invalid device index: {device_index}")
#             else:
#                 self.input_device = None
#                 self.logger.info("Using default input device")
                
#         except Exception as e:
#             self.logger.error(f"Error setting input device: {str(e)}")
    
#     def start_recording(self, audio_callback: Optional[Callable] = None) -> bool:
#         """
#         Start audio recording in a separate thread
        
#         Args:
#             audio_callback: Optional callback function to process audio chunks
            
#         Returns:
#             True if recording started successfully
#         """
#         try:
#             if self.is_recording:
#                 self.logger.warning("Audio recording already active")
#                 return True
            
#             self.audio_callback = audio_callback
#             self.is_recording = True
#             self.recorded_audio = []
            
#             # Start recording thread
#             self.recording_thread = threading.Thread(target=self._recording_loop)
#             self.recording_thread.daemon = True
#             self.recording_thread.start()
            
#             self.logger.info("Audio recording started")
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error starting audio recording: {str(e)}")
#             return False
    
#     def stop_recording(self) -> np.ndarray:
#         """
#         Stop audio recording and return recorded data
        
#         Returns:
#             Recorded audio as numpy array
#         """
#         try:
#             self.is_recording = False
            
#             if self.recording_thread and self.recording_thread.is_alive():
#                 self.recording_thread.join(timeout=2.0)
            
#             # Clear audio queue
#             while not self.audio_queue.empty():
#                 try:
#                     self.audio_queue.get_nowait()
#                 except queue.Empty:
#                     break
            
#             # Concatenate recorded audio
#             if self.recorded_audio:
#                 full_audio = np.concatenate(self.recorded_audio)
#             else:
#                 full_audio = np.array([], dtype=self.dtype)
            
#             self.logger.info(f"Audio recording stopped. Recorded {len(full_audio)} samples")
#             return full_audio
            
#         except Exception as e:
#             self.logger.error(f"Error stopping audio recording: {str(e)}")
#             return np.array([], dtype=self.dtype)
    
#     def _recording_loop(self):
#         """Main recording loop running in separate thread"""
#         try:
#             with sd.InputStream(
#                 device=self.input_device,
#                 channels=self.channels,
#                 samplerate=self.sample_rate,
#                 dtype=self.dtype,
#                 blocksize=self.chunk_size
#             ) as stream:
                
#                 while self.is_running:
#                     audio_chunk, overflowed = stream.read(self.chunk_size)
                    
#                     if overflowed:
#                         self.logger.warning("Audio input overflow detected")
                    
#                     # Flatten audio if multi-channel
#                     if audio_chunk.ndim > 1:
#                         audio_chunk = np.mean(audio_chunk, axis=1)
                    
#                     # Store audio chunk
#                     self.recorded_audio.append(audio_chunk.copy())
                    
#                     # Process with callback if provided
#                     if self.audio_callback:
#                         try:
#                             self.audio_callback(audio_chunk.copy())
#                         except Exception as e:
#                             self.logger.error(f"Error in audio callback: {str(e)}")
                    
#                     # Add to queue for real-time processing
#                     try:
#                         self.audio_queue.put_nowait({
#                             'audio': audio_chunk.copy(),
#                             'timestamp': datetime.now(),
#                             'sample_rate': self.sample_rate
#                         })
#                     except queue.Full:
#                         # Remove oldest chunk if queue is full
#                         try:
#                             self.audio_queue.get_nowait()
#                             self.audio_queue.put_nowait({
#                                 'audio': audio_chunk.copy(),
#                                 'timestamp': datetime.now(),
#                                 'sample_rate': self.sample_rate
#                             })
#                         except queue.Empty:
#                             pass
                            
#         except Exception as e:
#             self.logger.error(f"Error in recording loop: {str(e)}")
#         finally:
#             self.is_recording = False
    
#     def get_latest_audio_chunk(self) -> Optional[Dict]:
#         """
#         Get the most recent audio chunk
        
#         Returns:
#             Latest audio chunk data or None if no audio available
#         """
#         try:
#             if self.audio_queue.empty():
#                 return None
            
#             # Get the most recent chunk (drain queue except last)
#             latest_chunk = None
#             while not self.audio_queue.empty():
#                 try:
#                     latest_chunk = self.audio_queue.get_nowait()
#                 except queue.Empty:
#                     break
            
#             return latest_chunk
            
#         except Exception as e:
#             self.logger.error(f"Error getting latest audio chunk: {str(e)}")
#             return None
    
#     def record_fixed_duration(self, duration_seconds: float) -> np.ndarray:
#         """
#         Record audio for a fixed duration
        
#         Args:
#             duration_seconds: Recording duration in seconds
            
#         Returns:
#             Recorded audio array
#         """
#         try:
#             samples_to_record = int(duration_seconds * self.sample_rate)
            
#             self.logger.info(f"Recording {duration_seconds} seconds of audio...")
            
#             audio_data = sd.rec(
#                 samples_to_record,
#                 samplerate=self.sample_rate,
#                 channels=self.channels,
#                 dtype=self.dtype,
#                 device=self.input_device
#             )
            
#             # Wait for recording to complete
#             sd.wait()
            
#             # Flatten if multi-channel
#             if audio_data.ndim > 1:
#                 audio_data = np.mean(audio_data, axis=1)
            
#             self.logger.info(f"Recorded {len(audio_data)} samples")
#             return audio_data
            
#         except Exception as e:
#             self.logger.error(f"Error recording fixed duration audio: {str(e)}")
#             return np.array([], dtype=self.dtype)
    
#     def calculate_rms_energy(self, audio_data: np.ndarray) -> float:
#         """
#         Calculate RMS energy of audio signal
        
#         Args:
#             audio_data: Audio array
            
#         Returns:
#             RMS energy value
#         """
#         try:
#             if len(audio_data) == 0:
#                 return 0.0
            
#             rms = np.sqrt(np.mean(audio_data ** 2))
#             return float(rms)
            
#         except Exception as e:
#             self.logger.error(f"Error calculating RMS energy: {str(e)}")
#             return 0.0
    
#     def calculate_volume_level(self, audio_data: np.ndarray) -> float:
#         """
#         Calculate volume level (0-100)
        
#         Args:
#             audio_data: Audio array
            
#         Returns:
#             Volume level as percentage
#         """
#         try:
#             rms = self.calculate_rms_energy(audio_data)
#             # Convert RMS to decibels and normalize to 0-100 scale
#             if rms > 0:
#                 db = 20 * np.log10(rms)
#                 # Normalize assuming -60 dB to 0 dB range
#                 volume = max(0, min(100, (db + 60) / 60 * 100))
#             else:
#                 volume = 0.0
            
#             return volume
            
#         except Exception as e:
#             self.logger.error(f"Error calculating volume level: {str(e)}")
#             return 0.0
    
#     def detect_silence(self, audio_data: np.ndarray, 
#                       threshold: float = 0.01, min_duration: float = 0.5) -> bool:
#         """
#         Detect silence in audio
        
#         Args:
#             audio_data: Audio array
#             threshold: Silence threshold
#             min_duration: Minimum silence duration in seconds
            
#         Returns:
#             True if silence detected
#         """
#         try:
#             if len(audio_data) == 0:
#                 return True
            
#             rms = self.calculate_rms_energy(audio_data)
            
#             # Check if RMS is below threshold
#             is_quiet = rms < threshold
            
#             # Check duration (simplified - assumes entire chunk is silent or not)
#             duration = len(audio_data) / self.sample_rate
#             meets_duration = duration >= min_duration
            
#             return is_quiet and meets_duration
            
#         except Exception as e:
#             self.logger.error(f"Error detecting silence: {str(e)}")
#             return False
    
#     def apply_noise_gate(self, audio_data: np.ndarray, 
#                         threshold: float = 0.01) -> np.ndarray:
#         """
#         Apply noise gate to audio (suppress quiet sounds)
        
#         Args:
#             audio_data: Input audio array
#             threshold: Noise gate threshold
            
#         Returns:
#             Processed audio array
#         """
#         try:
#             if len(audio_data) == 0:
#                 return audio_data
            
#             # Calculate RMS in sliding windows
#             window_size = int(0.1 * self.sample_rate)  # 100ms windows
#             processed_audio = audio_data.copy()
            
#             for i in range(0, len(audio_data), window_size):
#                 end_idx = min(i + window_size, len(audio_data))
#                 window = audio_data[i:end_idx]
                
#                 if len(window) > 0:
#                     rms = np.sqrt(np.mean(window ** 2))
#                     if rms < threshold:
#                         processed_audio[i:end_idx] = 0  # Gate the audio
            
#             return processed_audio
            
#         except Exception as e:
#             self.logger.error(f"Error applying noise gate: {str(e)}")
#             return audio_data
    
#     def normalize_audio(self, audio_data: np.ndarray, 
#                        target_level: float = 0.8) -> np.ndarray:
#         """
#         Normalize audio to target level
        
#         Args:
#             audio_data: Input audio array
#             target_level: Target normalization level (0-1)
            
#         Returns:
#             Normalized audio array
#         """
#         try:
#             if len(audio_data) == 0:
#                 return audio_data
            
#             max_amplitude = np.max(np.abs(audio_data))
            
#             if max_amplitude > 0:
#                 scaling_factor = target_level / max_amplitude
#                 normalized_audio = audio_data * scaling_factor
#             else:
#                 normalized_audio = audio_data
            
#             return normalized_audio
            
#         except Exception as e:
#             self.logger.error(f"Error normalizing audio: {str(e)}")
#             return audio_data
    
#     def save_audio_to_wav(self, audio_data: np.ndarray, filename: str) -> bool:
#         """
#         Save audio data to WAV file
        
#         Args:
#             audio_data: Audio array to save
#             filename: Output filename
            
#         Returns:
#             True if saved successfully
#         """
#         try:
#             # Normalize audio to 16-bit range
#             if audio_data.dtype != np.int16:
#                 audio_int16 = (audio_data * 32767).astype(np.int16)
#             else:
#                 audio_int16 = audio_data
            
#             with wave.open(filename, 'w') as wav_file:
#                 wav_file.setnchannels(self.channels)
#                 wav_file.setsampwidth(2)  # 2 bytes for int16
#                 wav_file.setframerate(self.sample_rate)
#                 wav_file.writeframes(audio_int16.tobytes())
            
#             self.logger.info(f"Audio saved to {filename}")
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error saving audio to WAV: {str(e)}")
#             return False
    
#     def audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
#         """
#         Convert audio array to bytes (WAV format)
        
#         Args:
#             audio_data: Audio array
            
#         Returns:
#             Audio data as bytes
#         """
#         try:
#             # Create temporary WAV file in memory
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
#                 temp_filename = temp_file.name
            
#             # Save to temporary file
#             if self.save_audio_to_wav(audio_data, temp_filename):
#                 with open(temp_filename, 'rb') as f:
#                     audio_bytes = f.read()
                
#                 # Clean up temporary file
#                 os.unlink(temp_filename)
#                 return audio_bytes
#             else:
#                 return b''
                
#         except Exception as e:
#             self.logger.error(f"Error converting audio to bytes: {str(e)}")
#             return b''
    
#     def get_audio_stats(self, audio_data: np.ndarray) -> Dict[str, float]:
#         """
#         Get statistical information about audio
        
#         Args:
#             audio_data: Audio array
            
#         Returns:
#             Dictionary with audio statistics
#         """
#         try:
#             if len(audio_data) == 0:
#                 return {}
            
#             stats = {
#                 'duration_seconds': len(audio_data) / self.sample_rate,
#                 'sample_rate': self.sample_rate,
#                 'num_samples': len(audio_data),
#                 'rms_energy': self.calculate_rms_energy(audio_data),
#                 'volume_level': self.calculate_volume_level(audio_data),
#                 'max_amplitude': float(np.max(np.abs(audio_data))),
#                 'mean_amplitude': float(np.mean(np.abs(audio_data))),
#                 'std_amplitude': float(np.std(audio_data)),
#                 'zero_crossing_rate': float(np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data))
#             }
            
#             return stats
            
#         except Exception as e:
#             self.logger.error(f"Error calculating audio stats: {str(e)}")
#             return {}
    
#     def is_microphone_available(self) -> bool:
#         """
#         Check if microphone is available
        
#         Returns:
#             True if microphone is available
#         """
#         try:
#             # Try to query input devices
#             devices = sd.query_devices()
            
#             # Check if there's at least one input device
#             for device in devices:
#                 if device['max_input_channels'] > 0:
#                     return True
            
#             return False
            
#         except Exception as e:
#             self.logger.error(f"Error checking microphone availability: {str(e)}")
#             return False
    
#     def test_audio_input(self, duration: float = 2.0) -> Dict[str, Any]:
#         """
#         Test audio input and return diagnostics
        
#         Args:
#             duration: Test duration in seconds
            
#         Returns:
#             Dictionary with test results
#         """
#         try:
#             self.logger.info(f"Testing audio input for {duration} seconds...")
            
#             # Record test audio
#             test_audio = self.record_fixed_duration(duration)
            
#             # Get statistics
#             stats = self.get_audio_stats(test_audio)
            
#             # Determine test results
#             results = {
#                 'success': len(test_audio) > 0,
#                 'audio_detected': stats.get('volume_level', 0) > 1.0,
#                 'statistics': stats,
#                 'device_info': {
#                     'input_device': self.input_device,
#                     'sample_rate': self.sample_rate,
#                     'channels': self.channels
#                 }
#             }
            
#             self.logger.info(f"Audio test completed: {results['success']}")
#             return results
            
#         except Exception as e:
#             self.logger.error(f"Error testing audio input: {str(e)}")
#             return {'success': False, 'error': str(e)}
    
#     def cleanup(self):
#         """Clean up audio resources"""
#         try:
#             if self.is_recording:
#                 self.stop_recording()
            
#             self.logger.info("Audio processor cleaned up")
            
#         except Exception as e:
#             self.logger.error(f"Error during audio cleanup: {str(e)}")
    
#     def __del__(self):
#         """Destructor to ensure proper cleanup"""
#         self.cleanup()

import numpy as np
import sounddevice as sd
import threading
import queue
import time
import wave
import io
import tempfile
import os
from typing import Optional, Callable, Dict, Any, List, Tuple
import logging
from datetime import datetime

class AudioProcessor:
    """
    Audio processing utility for microphone operations and audio handling
    Supports real-time audio capture and processing for emotion detection
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 1, dtype=np.float32):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            dtype: Audio data type
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        # Audio capture parameters
        self.chunk_duration = 1.0  # Duration of audio chunks in seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=10)
        self.recorded_audio = []
        self.recording_thread = None
        self.audio_callback = None
        
        # Device settings
        self.input_device = None
        self.output_device = None
        
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize audio devices
        self._check_audio_devices()
    
    def _check_audio_devices(self):
        """Check available audio devices"""
        try:
            devices = sd.query_devices()
            self.logger.info(f"Available audio devices: {len(devices)}")
            
            # Get default input device
            default_input = sd.query_devices(kind='input')
            self.logger.info(f"Default input device: {default_input['name']}")
            
        except Exception as e:
            self.logger.error(f"Error checking audio devices: {str(e)}")
    
    def list_audio_devices(self) -> List[Dict]:
        """
        List all available audio devices
        
        Returns:
            List of device information dictionaries
        """
        try:
            devices = sd.query_devices()
            device_list = []
            
            for idx, device in enumerate(devices):
                device_info = {
                    'index': idx,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                }
                device_list.append(device_info)
            
            return device_list
            
        except Exception as e:
            self.logger.error(f"Error listing audio devices: {str(e)}")
            return []
    
    def set_input_device(self, device_index: Optional[int] = None):
        """
        Set input audio device
        
        Args:
            device_index: Device index (None for default)
        """
        self.input_device = device_index
        self.logger.info(f"Input device set to: {device_index}")
    
    def start_recording(self, callback: Optional[Callable] = None):
        """
        Start audio recording in a separate thread
        
        Args:
            callback: Optional callback function for real-time audio processing
        """
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return
        
        self.is_recording = True
        self.recorded_audio = []
        self.audio_callback = callback
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.logger.info("Audio recording started")
    
    def _recording_loop(self):
        """Main recording loop running in separate thread"""
        try:
            with sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                blocksize=self.chunk_size
            ) as stream:
                while self.is_recording:  # Fixed: was self.is_running
                    audio_chunk, overflowed = stream.read(self.chunk_size)
                    
                    if overflowed:
                        self.logger.warning("Audio input overflow detected")
                    
                    # Flatten audio if multi-channel
                    if audio_chunk.ndim > 1:
                        audio_chunk = np.mean(audio_chunk, axis=1)
                    
                    # Store audio chunk
                    self.recorded_audio.append(audio_chunk.copy())
                    
                    # Process with callback if provided
                    if self.audio_callback:
                        try:
                            self.audio_callback(audio_chunk.copy())
                        except Exception as e:
                            self.logger.error(f"Error in audio callback: {str(e)}")
                    
                    # Add to queue for real-time processing
                    try:
                        self.audio_queue.put_nowait({
                            'audio': audio_chunk.copy(),
                            'timestamp': datetime.now(),
                            'sample_rate': self.sample_rate
                        })
                    except queue.Full:
                        # Remove oldest chunk if queue is full
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait({
                                'audio': audio_chunk.copy(),
                                'timestamp': datetime.now(),
                                'sample_rate': self.sample_rate
                            })
                        except queue.Empty:
                            pass
                            
        except Exception as e:
            self.logger.error(f"Error in recording loop: {str(e)}")
        finally:
            self.is_recording = False
    
    def stop_recording(self) -> np.ndarray:
        """
        Stop audio recording
        
        Returns:
            Recorded audio as numpy array
        """
        if not self.is_recording:
            self.logger.warning("No recording in progress")
            return np.array([])
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # Concatenate recorded audio chunks
        if self.recorded_audio:
            audio_data = np.concatenate(self.recorded_audio)
            self.logger.info(f"Recording stopped. Duration: {len(audio_data)/self.sample_rate:.2f}s")
            return audio_data
        else:
            self.logger.warning("No audio data recorded")
            return np.array([])
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next audio chunk from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio chunk dictionary or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def save_audio(self, audio_data: np.ndarray, filename: str):
        """
        Save audio data to WAV file
        
        Args:
            audio_data: Audio data as numpy array
            filename: Output filename
        """
        try:
            # Normalize audio to 16-bit PCM range
            audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            
            with wave.open(filename, 'w') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_normalized.tobytes())
            
            self.logger.info(f"Audio saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving audio: {str(e)}")
    
    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from WAV file
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            with wave.open(filename, 'r') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32767.0
                
                self.logger.info(f"Audio loaded from {filename}")
                return audio_data, sample_rate
                
        except Exception as e:
            self.logger.error(f"Error loading audio: {str(e)}")
            return np.array([]), self.sample_rate
    
    def audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """
        Convert audio numpy array to WAV bytes
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            WAV file bytes
        """
        try:
            # Normalize audio to 16-bit PCM range
            audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'w') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_normalized.tobytes())
            
            return wav_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error converting audio to bytes: {str(e)}")
            return b''
    
    def bytes_to_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Convert WAV bytes to audio numpy array
        
        Args:
            audio_bytes: WAV file bytes
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'r') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32767.0
                
                return audio_data, sample_rate
                
        except Exception as e:
            self.logger.error(f"Error converting bytes to audio: {str(e)}")
            return np.array([]), self.sample_rate
    
    def play_audio(self, audio_data: np.ndarray):
        """
        Play audio through speakers
        
        Args:
            audio_data: Audio data as numpy array
        """
        try:
            sd.play(audio_data, self.sample_rate)
            sd.wait()
            self.logger.info("Audio playback completed")
            
        except Exception as e:
            self.logger.error(f"Error playing audio: {str(e)}")
    
    def get_audio_level(self, audio_data: np.ndarray) -> float:
        """
        Calculate audio level (RMS)
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            RMS audio level
        """
        try:
            return float(np.sqrt(np.mean(audio_data**2)))
        except Exception as e:
            self.logger.error(f"Error calculating audio level: {str(e)}")
            return 0.0
    
    def detect_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if audio is silent
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Silence threshold
            
        Returns:
            True if audio is silent
        """
        level = self.get_audio_level(audio_data)
        return level < threshold
    
    def trim_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Trim silence from beginning and end of audio
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Silence threshold
            
        Returns:
            Trimmed audio data
        """
        try:
            # Calculate frame energy
            frame_length = int(self.sample_rate * 0.02)  # 20ms frames
            energy = np.array([
                np.sqrt(np.mean(audio_data[i:i+frame_length]**2))
                for i in range(0, len(audio_data), frame_length)
            ])
            
            # Find non-silent frames
            non_silent = energy > threshold
            
            if not np.any(non_silent):
                return audio_data
            
            # Get indices of first and last non-silent frames
            first_frame = np.argmax(non_silent)
            last_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1
            
            # Convert frame indices to sample indices
            start_sample = first_frame * frame_length
            end_sample = min((last_frame + 1) * frame_length, len(audio_data))
            
            return audio_data[start_sample:end_sample]
            
        except Exception as e:
            self.logger.error(f"Error trimming silence: {str(e)}")
            return audio_data
    
    def resample_audio(self, audio_data: np.ndarray, target_sample_rate: int) -> np.ndarray:
        """
        Resample audio to different sample rate
        
        Args:
            audio_data: Audio data as numpy array
            target_sample_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        try:
            if self.sample_rate == target_sample_rate:
                return audio_data
            
            # Calculate resampling ratio
            ratio = target_sample_rate / self.sample_rate
            new_length = int(len(audio_data) * ratio)
            
            # Simple linear interpolation resampling
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
            
            self.logger.info(f"Audio resampled from {self.sample_rate}Hz to {target_sample_rate}Hz")
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling audio: {str(e)}")
            return audio_data
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
        
        self.logger.info("AudioProcessor cleanup completed")


# Example usage
if __name__ == "__main__":
    # Create audio processor
    processor = AudioProcessor(sample_rate=44100, channels=1)
    
    # List available devices
    devices = processor.list_audio_devices()
    print("Available audio devices:")
    for device in devices:
        print(f"  {device['index']}: {device['name']}")
    
    # Record audio for 3 seconds
    print("\nRecording for 3 seconds...")
    processor.start_recording()
    time.sleep(3)
    audio_data = processor.stop_recording()
    
    # Get audio level
    level = processor.get_audio_level(audio_data)
    print(f"Audio level: {level:.4f}")
    
    # Save audio
    processor.save_audio(audio_data, "test_recording.wav")
    print("Audio saved to test_recording.wav")
    
    # Clean up
    processor.cleanup()
