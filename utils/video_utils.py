import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Callable, Dict, Any
import logging
from datetime import datetime

class VideoProcessor:
    """
    Video processing utility for camera operations and frame handling
    Supports real-time video capture and processing for emotion detection
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize video processor
        
        Args:
            camera_index: Camera device index (0 for default camera)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.frame_callback = None
        
        # Video parameters
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
        # Processing parameters
        self.process_every_n_frames = 1  # Process every frame by default
        self.frame_count = 0
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"Cannot open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error("Cannot read from camera")
                self.cap.release()
                self.cap = None
                return False
            
            self.logger.info(f"Camera {self.camera_index} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def start_capture(self, frame_callback: Optional[Callable] = None) -> bool:
        """
        Start video capture in a separate thread
        
        Args:
            frame_callback: Optional callback function to process frames
            
        Returns:
            True if capture started successfully
        """
        try:
            if self.is_running:
                self.logger.warning("Video capture already running")
                return True
            
            if not self.cap and not self.initialize_camera():
                return False
            
            self.frame_callback = frame_callback
            self.is_running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._capture_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info("Video capture started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting video capture: {str(e)}")
            return False
    
    def stop_capture(self):
        """Stop video capture"""
        try:
            self.is_running = False
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("Video capture stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping video capture: {str(e)}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        try:
            while self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Process frame if callback is provided and it's time to process
                if (self.frame_callback and 
                    self.frame_count % self.process_every_n_frames == 0):
                    
                    try:
                        processed_frame = self.frame_callback(frame.copy())
                        if processed_frame is not None:
                            frame = processed_frame
                    except Exception as e:
                        self.logger.error(f"Error in frame callback: {str(e)}")
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame,
                        'timestamp': datetime.now(),
                        'frame_number': self.frame_count
                    })
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait({
                            'frame': frame,
                            'timestamp': datetime.now(),
                            'frame_number': self.frame_count
                        })
                    except queue.Empty:
                        pass
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
        except Exception as e:
            self.logger.error(f"Error in capture loop: {str(e)}")
        finally:
            self.is_running = False
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame
        
        Returns:
            Latest frame or None if no frame available
        """
        try:
            if self.frame_queue.empty():
                return None
            
            # Get the most recent frame (drain queue except last)
            latest_frame_data = None
            while not self.frame_queue.empty():
                try:
                    latest_frame_data = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_frame_data:
                return latest_frame_data['frame']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest frame: {str(e)}")
            return None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame (useful for one-shot captures)
        
        Returns:
            Captured frame or None if failed
        """
        try:
            if not self.cap and not self.initialize_camera():
                return None
            
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                return frame
            else:
                self.logger.warning("Failed to capture single frame")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing single frame: {str(e)}")
            return None
    
    def resize_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Resize frame to specified dimensions
        
        Args:
            frame: Input frame
            width: Target width
            height: Target height
            
        Returns:
            Resized frame
        """
        try:
            return cv2.resize(frame, (width, height))
        except Exception as e:
            self.logger.error(f"Error resizing frame: {str(e)}")
            return frame
    
    def convert_color_space(self, frame: np.ndarray, conversion: int) -> np.ndarray:
        """
        Convert frame color space
        
        Args:
            frame: Input frame
            conversion: OpenCV color conversion code (e.g., cv2.COLOR_BGR2RGB)
            
        Returns:
            Converted frame
        """
        try:
            return cv2.cvtColor(frame, conversion)
        except Exception as e:
            self.logger.error(f"Error converting color space: {str(e)}")
            return frame
    
    def add_timestamp_overlay(self, frame: np.ndarray, 
                             position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Add timestamp overlay to frame
        
        Args:
            frame: Input frame
            position: Text position (x, y)
            
        Returns:
            Frame with timestamp overlay
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cv2.putText(frame, timestamp, position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error adding timestamp overlay: {str(e)}")
            return frame
    
    def detect_motion(self, frame: np.ndarray, 
                     background_frame: Optional[np.ndarray] = None,
                     threshold: int = 30) -> Tuple[bool, np.ndarray]:
        """
        Simple motion detection
        
        Args:
            frame: Current frame
            background_frame: Background reference frame
            threshold: Motion detection threshold
            
        Returns:
            Tuple of (motion_detected, difference_frame)
        """
        try:
            if background_frame is None:
                return False, frame
            
            # Convert to grayscale
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_background, gray_current)
            
            # Apply threshold
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate percentage of changed pixels
            changed_pixels = np.sum(thresh > 0)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            change_ratio = changed_pixels / total_pixels
            
            # Motion detected if more than 1% of pixels changed
            motion_detected = change_ratio > 0.01
            
            return motion_detected, diff
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {str(e)}")
            return False, frame
    
    def apply_gaussian_blur(self, frame: np.ndarray, 
                          kernel_size: Tuple[int, int] = (15, 15)) -> np.ndarray:
        """
        Apply Gaussian blur to frame
        
        Args:
            frame: Input frame
            kernel_size: Blur kernel size
            
        Returns:
            Blurred frame
        """
        try:
            return cv2.GaussianBlur(frame, kernel_size, 0)
        except Exception as e:
            self.logger.error(f"Error applying Gaussian blur: {str(e)}")
            return frame
    
    def enhance_contrast(self, frame: np.ndarray, 
                        alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Enhance frame contrast
        
        Args:
            frame: Input frame
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Enhanced frame
        """
        try:
            return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        except Exception as e:
            self.logger.error(f"Error enhancing contrast: {str(e)}")
            return frame
    
    def get_camera_properties(self) -> Dict[str, Any]:
        """
        Get current camera properties
        
        Returns:
            Dictionary of camera properties
        """
        try:
            if not self.cap:
                return {}
            
            properties = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'hue': self.cap.get(cv2.CAP_PROP_HUE)
            }
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error getting camera properties: {str(e)}")
            return {}
    
    def set_processing_frequency(self, every_n_frames: int):
        """
        Set how often to process frames (for performance optimization)
        
        Args:
            every_n_frames: Process every N frames (1 = every frame)
        """
        self.process_every_n_frames = max(1, every_n_frames)
        self.logger.info(f"Set processing frequency to every {self.process_every_n_frames} frames")
    
    def is_camera_available(self) -> bool:
        """
        Check if camera is available
        
        Returns:
            True if camera is available
        """
        try:
            temp_cap = cv2.VideoCapture(self.camera_index)
            available = temp_cap.isOpened()
            temp_cap.release()
            return available
        except Exception as e:
            self.logger.error(f"Error checking camera availability: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up video resources"""
        try:
            self.stop_capture()
            self.logger.info("Video processor cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.cleanup()

