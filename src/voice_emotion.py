import numpy as np
import librosa
import librosa.display
import speech_recognition as sr
import logging
from typing import Dict, Optional, List
import io
import wave
import tempfile
import os

# Try to import transformers, but continue without it if unavailable
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers library not available: {e}. Using acoustic-only emotion detection.")
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

class VoiceEmotionDetector:
    """
    Voice emotion detection using audio features and pre-trained models
    Analyzes both acoustic features and speech-to-text content
    """
    
    def __init__(self, model_name: str = 'j-hartmann/emotion-english-distilroberta-base'):
        """
        Initialize the voice emotion detector
        
        Args:
            model_name: HuggingFace model for emotion classification
        """
        self.model_name = model_name
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize emotion classification pipeline
        if not TRANSFORMERS_AVAILABLE:
            logging.warning("Transformers library not available. Using acoustic-only emotion detection.")
            self.emotion_classifier = None
        else:
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    top_k=None
                )
            except Exception as e:
                logging.warning(f"Could not load emotion model: {e}. Using fallback.")
                self.emotion_classifier = None
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.frame_length = 2048
        self.hop_length = 512
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Calibrate microphone for ambient noise
        self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            self.logger.warning(f"Microphone calibration failed: {str(e)}")
    
    def extract_audio_features(self, audio_data: np.ndarray, sr: int = None) -> Dict:
        """
        Extract acoustic features from audio data
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        try:
            if sr is None:
                sr = self.sample_rate
            
            features = {}
            
            # Basic features
            features['duration'] = len(audio_data) / sr
            features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            features['tempo'] = tempo
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            return {}
    
    def classify_acoustic_emotion(self, features: Dict) -> Dict:
        """
        Classify emotion based on acoustic features
        
        Args:
            features: Extracted audio features
            
        Returns:
            Emotion probabilities based on acoustic analysis
        """
        try:
            # Simple rule-based emotion classification based on acoustic features
            # In a production system, this would use a trained ML model
            
            emotions = {
                'anger': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'joy': 0.0,
                'neutral': 0.5,  # Default baseline
                'sadness': 0.0,
                'surprise': 0.0
            }
            
            if not features:
                return emotions
            
            # High energy + high pitch = anger/excitement
            if features.get('rms_energy', 0) > 0.1 and features.get('spectral_centroid_mean', 0) > 2000:
                emotions['anger'] += 0.3
                emotions['joy'] += 0.2
                emotions['neutral'] -= 0.2
            
            # Low energy + low pitch = sadness
            if features.get('rms_energy', 0) < 0.05 and features.get('spectral_centroid_mean', 0) < 1500:
                emotions['sadness'] += 0.4
                emotions['neutral'] -= 0.2
            
            # High variability in pitch = surprise/fear
            if features.get('spectral_centroid_std', 0) > 500:
                emotions['surprise'] += 0.2
                emotions['fear'] += 0.1
                emotions['neutral'] -= 0.1
            
            # Fast tempo = excitement/joy
            if features.get('tempo', 0) > 120:
                emotions['joy'] += 0.2
                emotions['neutral'] -= 0.1
            
            # Slow tempo = sadness
            if features.get('tempo', 0) < 80:
                emotions['sadness'] += 0.2
                emotions['neutral'] -= 0.1
            
            # Normalize probabilities
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: max(0, v/total) for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            self.logger.error(f"Acoustic emotion classification error: {str(e)}")
            return {'neutral': 1.0}
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """
        Convert speech to text using speech recognition
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
        """
        try:
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Load audio file for speech recognition
            with sr.AudioFile(temp_file_path) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return text
            
        except sr.UnknownValueError:
            self.logger.warning("Could not understand audio")
            return ""
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition error: {str(e)}")
            return ""
        except Exception as e:
            self.logger.error(f"Speech to text error: {str(e)}")
            return ""
    
    def classify_text_emotion(self, text: str) -> Dict:
        """
        Classify emotion from text using transformer model
        
        Args:
            text: Input text
            
        Returns:
            Emotion probabilities
        """
        try:
            if not text or not self.emotion_classifier:
                return {'neutral': 1.0}
            
            results = self.emotion_classifier(text)
            
            # Convert to our emotion format
            emotion_map = {
                'ANGER': 'anger',
                'DISGUST': 'disgust',
                'FEAR': 'fear',
                'JOY': 'joy',
                'NEUTRAL': 'neutral',
                'SADNESS': 'sadness',
                'SURPRISE': 'surprise'
            }
            
            emotions = {}
            for result in results:
                label = result['label'].upper()
                mapped_emotion = emotion_map.get(label, 'neutral')
                emotions[mapped_emotion] = result['score']
            
            # Normalize
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            self.logger.error(f"Text emotion classification error: {str(e)}")
            return {'neutral': 1.0}
    
    def record_audio(self, duration: int = 5) -> Optional[bytes]:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Raw audio data
        """
        try:
            with self.microphone as source:
                self.logger.info(f"Recording audio for {duration} seconds...")
                audio_data = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
                return audio_data.get_wav_data()
                
        except Exception as e:
            self.logger.error(f"Audio recording error: {str(e)}")
            return None
    
    def analyze_audio_chunk(self, audio_data: bytes = None) -> Dict:
        """
        Analyze emotion from audio chunk
        
        Args:
            audio_data: Raw audio data (if None, records new audio)
            
        Returns:
            Combined emotion probabilities
        """
        try:
            # Record audio if not provided
            if audio_data is None:
                audio_data = self.record_audio(duration=3)
                if audio_data is None:
                    return {'neutral': 1.0}
            
            # Convert to numpy array for feature extraction
            try:
                # Load audio data
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                y, sr = librosa.load(temp_file_path, sr=self.sample_rate)
                os.unlink(temp_file_path)
                
            except Exception as e:
                self.logger.warning(f"Could not process audio for features: {str(e)}")
                y, sr = np.array([]), self.sample_rate
            
            # Extract acoustic features and classify
            acoustic_emotions = {'neutral': 1.0}
            if len(y) > 0:
                features = self.extract_audio_features(y, sr)
                acoustic_emotions = self.classify_acoustic_emotion(features)
            
            # Speech to text and classify
            text = self.speech_to_text(audio_data)
            text_emotions = self.classify_text_emotion(text) if text else {'neutral': 1.0}
            
            # Combine acoustic and text emotions (weighted average)
            combined_emotions = {}
            acoustic_weight = 0.6
            text_weight = 0.4
            
            all_emotions = set(list(acoustic_emotions.keys()) + list(text_emotions.keys()))
            
            for emotion in all_emotions:
                acoustic_score = acoustic_emotions.get(emotion, 0.0)
                text_score = text_emotions.get(emotion, 0.0)
                combined_emotions[emotion] = (
                    acoustic_weight * acoustic_score + text_weight * text_score
                )
            
            # Normalize
            total = sum(combined_emotions.values())
            if total > 0:
                combined_emotions = {k: v/total for k, v in combined_emotions.items()}
            
            # Log results
            if text:
                self.logger.info(f"Transcribed text: '{text}'")
            self.logger.info(f"Voice emotion analysis: {combined_emotions}")
            
            return combined_emotions
            
        except Exception as e:
            self.logger.error(f"Audio analysis error: {str(e)}")
            return {'neutral': 1.0}
    
    def cleanup(self):
        """Clean up resources"""
        # Clean up any temporary files or connections
        pass
