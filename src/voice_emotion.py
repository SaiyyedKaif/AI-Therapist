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
                # Start with minimal neutral baseline (reduced from 0.5)
                emotions = {
                    'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 
                    'joy': 0.0, 'neutral': 0.02, 'sadness': 0.0, 'surprise': 0.0
                }
                
                if not features:
                    return emotions
                
                # Extract key features with defaults
                rms = features.get('rms_energy', 0)
                spectral_mean = features.get('spectral_centroid_mean', 0)
                spectral_std = features.get('spectral_centroid_std', 0)
                tempo = features.get('tempo', 90)
                zcr_mean = features.get('zcr_mean', 0)
                
                # High energy + high pitch = anger/excitement (lowered thresholds)
                if rms > 0.02 and spectral_mean > 1500:
                    emotions['anger'] += 0.4
                    emotions['joy'] += 0.2
                    emotions['neutral'] -= 0.15
                
                # Very high energy + very high pitch = strong anger
                if rms > 0.05 and spectral_mean > 2500:
                    emotions['anger'] += 0.3
                    emotions['neutral'] -= 0.1
                
                # Low energy + low pitch = sadness (more realistic thresholds)
                if rms < 0.015 and spectral_mean < 1200:
                    emotions['sadness'] += 0.5
                    emotions['neutral'] -= 0.2
                
                # Moderate-low energy + moderate pitch = neutral/calm is already low
                # So we boost it slightly only if truly flat
                if 0.015 <= rms <= 0.025 and 1200 <= spectral_mean <= 1800:
                    emotions['neutral'] += 0.05
                
                # High pitch variability = surprise/fear
                if spectral_std > 300:
                    emotions['surprise'] += 0.3
                    emotions['fear'] += 0.2
                    emotions['neutral'] -= 0.15
                
                # Very high variability = strong surprise
                if spectral_std > 500:
                    emotions['surprise'] += 0.2
                    emotions['neutral'] -= 0.1
                
                # Fast tempo = excitement/joy
                if tempo > 110:
                    emotions['joy'] += 0.3
                    emotions['neutral'] -= 0.1
                
                # Very fast = strong joy/excitement
                if tempo > 140:
                    emotions['joy'] += 0.2
                    emotions['neutral'] -= 0.1
                
                # Slow tempo = sadness
                if tempo < 85:
                    emotions['sadness'] += 0.3
                    emotions['neutral'] -= 0.1
                
                # Very slow = strong sadness
                if tempo < 70:
                    emotions['sadness'] += 0.2
                    emotions['neutral'] -= 0.1
                
                # High zero crossing rate = agitation/anger
                if zcr_mean > 0.1:
                    emotions['anger'] += 0.2
                    emotions['fear'] += 0.1
                    emotions['neutral'] -= 0.1
                
                # Low zero crossing rate = calm/sad
                if zcr_mean < 0.05:
                    emotions['sadness'] += 0.15
                
                # Disgust is harder to detect from acoustics alone
                # Use combination: moderate energy + irregular spectral pattern
                if 0.02 < rms < 0.04 and spectral_std > 400:
                    emotions['disgust'] += 0.2
                    emotions['neutral'] -= 0.1
                
                # First normalization
                total = sum(max(0, v) for v in emotions.values())
                if total > 0:
                    emotions = {k: max(0, v)/total for k, v in emotions.items()}
                
                # Suppress neutral if any non-neutral crosses threshold (same as text model)
                tau = 0.22
                non_neutral = {k: v for k, v in emotions.items() if k != 'neutral'}
                if non_neutral and max(non_neutral.values()) >= tau:
                    emotions['neutral'] = min(emotions['neutral'], 0.01)
                
                # Temperature scaling for non-neutral emotions (sharpen peaks)
                T = 0.8
                non_neutral_keys = [k for k in emotions if k != 'neutral']
                for k in non_neutral_keys:
                    emotions[k] = emotions[k] ** (1.0 / T)
                
                # Final normalization
                total = sum(emotions.values())
                if total > 0:
                    emotions = {k: v/total for k, v in emotions.items()}
                
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


# import numpy as np
# import librosa
# import librosa.display
# import speech_recognition as sr
# import logging
# from typing import Dict, Optional, List
# import io
# import wave
# import tempfile
# import os

# # Try to import transformers, but continue without it if unavailable
# try:
#     from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
#     TRANSFORMERS_AVAILABLE = True
# except ImportError as e:
#     logging.warning(f"Transformers library not available: {e}. Using acoustic-only emotion detection.")
#     TRANSFORMERS_AVAILABLE = False
#     pipeline = None
#     AutoTokenizer = None
#     AutoModelForSequenceClassification = None

# class VoiceEmotionDetector:
#     """
#     Voice emotion detection using audio features and pre-trained models
#     Analyzes both acoustic features and speech-to-text content
#     """
    
#     def __init__(self, model_name: str = 'j-hartmann/emotion-english-distilroberta-base'):
#         """
#         Initialize the voice emotion detector
        
#         Args:
#             model_name: HuggingFace model for emotion classification
#         """
#         self.model_name = model_name
#         self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
#         # Initialize speech recognition
#         self.recognizer = sr.Recognizer()
#         self.microphone = sr.Microphone()
        
#         # Initialize emotion classification pipeline
#         if not TRANSFORMERS_AVAILABLE:
#             logging.warning("Transformers library not available. Using acoustic-only emotion detection.")
#             self.emotion_classifier = None
#         else:
#             try:
#                 self.emotion_classifier = pipeline(
#                     "text-classification",
#                     model=model_name,
#                     tokenizer=model_name,
#                     top_k=None
#                 )
#             except Exception as e:
#                 logging.warning(f"Could not load emotion model: {e}. Using fallback.")
#                 self.emotion_classifier = None
        
#         # Audio processing parameters
#         self.sample_rate = 22050
#         self.frame_length = 2048
#         self.hop_length = 512
        
#         # Initialize logger
#         logging.basicConfig(level=logging.INFO)
#         self.logger = logging.getLogger(__name__)
        
#         # Calibrate microphone for ambient noise
#         self._calibrate_microphone()
    
#     def _calibrate_microphone(self):
#         """Calibrate microphone for ambient noise"""
#         try:
#             with self.microphone as source:
#                 self.recognizer.adjust_for_ambient_noise(source, duration=1)
#         except Exception as e:
#             self.logger.warning(f"Microphone calibration failed: {str(e)}")
    
#     def extract_audio_features(self, audio_data: np.ndarray, sr: int = None) -> Dict:
#         """
#         Extract acoustic features from audio data
        
#         Args:
#             audio_data: Audio time series
#             sr: Sample rate
            
#         Returns:
#             Dictionary of extracted features
#         """
#         try:
#             if sr is None:
#                 sr = self.sample_rate
            
#             features = {}
            
#             # Basic features
#             features['duration'] = len(audio_data) / sr
#             features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
            
#             # Spectral features
#             spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
#             features['spectral_centroid_mean'] = np.mean(spectral_centroids)
#             features['spectral_centroid_std'] = np.std(spectral_centroids)
            
#             spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
#             features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
#             # Zero crossing rate
#             zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
#             features['zcr_mean'] = np.mean(zcr)
#             features['zcr_std'] = np.std(zcr)
            
#             # MFCC features
#             mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
#             for i in range(13):
#                 features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
#                 features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
#             # Chroma features
#             chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
#             features['chroma_mean'] = np.mean(chroma)
#             features['chroma_std'] = np.std(chroma)
            
#             # Tempo
#             tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
#             features['tempo'] = tempo
            
#             return features
            
#         except Exception as e:
#             self.logger.error(f"Feature extraction error: {str(e)}")
#             return {}
    
#     def classify_acoustic_emotion(self, features: Dict) -> Dict:
#         """
#         Classify emotion based on acoustic features
        
#         Args:
#             features: Extracted audio features
            
#         Returns:
#             Emotion probabilities based on acoustic analysis
#         """
#         try:
#             # Start with minimal neutral baseline (reduced from 0.5)
#             emotions = {
#                 'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 
#                 'joy': 0.0, 'neutral': 0.02, 'sadness': 0.0, 'surprise': 0.0
#             }
            
#             if not features:
#                 return emotions
            
#             # Extract key features with defaults
#             rms = features.get('rms_energy', 0)
#             spectral_mean = features.get('spectral_centroid_mean', 0)
#             spectral_std = features.get('spectral_centroid_std', 0)
#             tempo = features.get('tempo', 90)
#             zcr_mean = features.get('zcr_mean', 0)
            
#             # High energy + high pitch = anger/excitement (lowered thresholds)
#             if rms > 0.02 and spectral_mean > 1500:
#                 emotions['anger'] += 0.4
#                 emotions['joy'] += 0.2
#                 emotions['neutral'] -= 0.15
            
#             # Very high energy + very high pitch = strong anger
#             if rms > 0.05 and spectral_mean > 2500:
#                 emotions['anger'] += 0.3
#                 emotions['neutral'] -= 0.1
            
#             # Low energy + low pitch = sadness (more realistic thresholds)
#             if rms < 0.015 and spectral_mean < 1200:
#                 emotions['sadness'] += 0.5
#                 emotions['neutral'] -= 0.2
            
#             # Moderate-low energy + moderate pitch = neutral/calm is already low
#             # So we boost it slightly only if truly flat
#             if 0.015 <= rms <= 0.025 and 1200 <= spectral_mean <= 1800:
#                 emotions['neutral'] += 0.05
            
#             # High pitch variability = surprise/fear
#             if spectral_std > 300:
#                 emotions['surprise'] += 0.3
#                 emotions['fear'] += 0.2
#                 emotions['neutral'] -= 0.15
            
#             # Very high variability = strong surprise
#             if spectral_std > 500:
#                 emotions['surprise'] += 0.2
#                 emotions['neutral'] -= 0.1
            
#             # Fast tempo = excitement/joy
#             if tempo > 110:
#                 emotions['joy'] += 0.3
#                 emotions['neutral'] -= 0.1
            
#             # Very fast = strong joy/excitement
#             if tempo > 140:
#                 emotions['joy'] += 0.2
#                 emotions['neutral'] -= 0.1
            
#             # Slow tempo = sadness
#             if tempo < 85:
#                 emotions['sadness'] += 0.3
#                 emotions['neutral'] -= 0.1
            
#             # Very slow = strong sadness
#             if tempo < 70:
#                 emotions['sadness'] += 0.2
#                 emotions['neutral'] -= 0.1
            
#             # High zero crossing rate = agitation/anger
#             if zcr_mean > 0.1:
#                 emotions['anger'] += 0.2
#                 emotions['fear'] += 0.1
#                 emotions['neutral'] -= 0.1
            
#             # Low zero crossing rate = calm/sad
#             if zcr_mean < 0.05:
#                 emotions['sadness'] += 0.15
            
#             # Disgust is harder to detect from acoustics alone
#             # Use combination: moderate energy + irregular spectral pattern
#             if 0.02 < rms < 0.04 and spectral_std > 400:
#                 emotions['disgust'] += 0.2
#                 emotions['neutral'] -= 0.1
            
#             # First normalization
#             total = sum(max(0, v) for v in emotions.values())
#             if total > 0:
#                 emotions = {k: max(0, v)/total for k, v in emotions.items()}
            
#             # Suppress neutral if any non-neutral crosses threshold (same as text model)
#             tau = 0.22
#             non_neutral = {k: v for k, v in emotions.items() if k != 'neutral'}
#             if non_neutral and max(non_neutral.values()) >= tau:
#                 emotions['neutral'] = min(emotions['neutral'], 0.01)
            
#             # Temperature scaling for non-neutral emotions (sharpen peaks)
#             T = 0.8
#             non_neutral_keys = [k for k in emotions if k != 'neutral']
#             for k in non_neutral_keys:
#                 emotions[k] = emotions[k] ** (1.0 / T)
            
#             # Final normalization
#             total = sum(emotions.values())
#             if total > 0:
#                 emotions = {k: v/total for k, v in emotions.items()}
            
#             return emotions
            
#         except Exception as e:
#             self.logger.error(f"Acoustic emotion classification error: {str(e)}")
#             return {'neutral': 1.0}
    
#     def speech_to_text(self, audio_data: bytes) -> str:
#         """
#         Convert speech to text using speech recognition
        
#         Args:
#             audio_data: Raw audio bytes
            
#         Returns:
#             Transcribed text
#         """
#         try:
#             # Create temporary wav file
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
#                 temp_file.write(audio_data)
#                 temp_file_path = temp_file.name
            
#             # Load audio file for speech recognition
#             with sr.AudioFile(temp_file_path) as source:
#                 audio = self.recognizer.record(source)
            
#             # Recognize speech
#             text = self.recognizer.recognize_google(audio)
            
#             # Clean up temporary file
#             os.unlink(temp_file_path)
            
#             return text
            
#         except sr.UnknownValueError:
#             self.logger.warning("Could not understand audio")
#             return ""
#         except sr.RequestError as e:
#             self.logger.error(f"Speech recognition error: {str(e)}")
#             return ""
#         except Exception as e:
#             self.logger.error(f"Speech to text error: {str(e)}")
#             return ""
    
#     def classify_text_emotion(self, text: str) -> Dict:
#         """
#         Classify emotion from text using transformer model
        
#         Args:
#             text: Input text
            
#         Returns:
#             Emotion probabilities
#         """
#         try:
#             if not text or not self.emotion_classifier:
#                 return {'neutral': 1.0}
            
#             results = self.emotion_classifier(text)[0]
            
#             # Convert to our emotion format
#             emotion_map = {
#                 'anger': 'anger',
#                 'disgust': 'disgust',
#                 'fear': 'fear',
#                 'joy': 'joy',
#                 'neutral': 'neutral',
#                 'sadness': 'sadness',
#                 'surprise': 'surprise'
#             }
            
#             emotions = {}
#             for result in results:
#                 label = result['label'].lower()
#                 mapped_emotion = emotion_map.get(label, 'neutral')
#                 emotions[mapped_emotion] = result['score']
            
#             # Normalize
#             total = sum(emotions.values())
#             if total > 0:
#                 emotions = {k: v/total for k, v in emotions.items()}
            
#             return emotions
            
#         except Exception as e:
#             self.logger.error(f"Text emotion classification error: {str(e)}")
#             return {'neutral': 1.0}
    
#     def get_emotion_description(self, emotions: Dict, transcribed_text: str = "") -> str:
#         """
#         Generate detailed emotion description from probabilities
        
#         Args:
#             emotions: Emotion probability distribution
#             transcribed_text: Optional transcribed text from speech
            
#         Returns:
#             Human-readable emotion analysis
#         """
#         # Get primary and secondary emotions
#         sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
#         primary = sorted_emotions[0]
#         secondary = sorted_emotions[1] if len(sorted_emotions) > 1 else None
        
#         # Emotion descriptions
#         emotion_descriptions = {
#             'joy': 'happiness, excitement, or pleasure',
#             'sadness': 'sadness, disappointment, or melancholy',
#             'anger': 'anger, frustration, or irritation',
#             'fear': 'fear, anxiety, or worry',
#             'surprise': 'surprise, shock, or astonishment',
#             'disgust': 'disgust, revulsion, or distaste',
#             'neutral': 'a calm or neutral state'
#         }
        
#         # Build description
#         description = f"**Primary Emotion:** {primary[0].capitalize()} ({primary[1]*100:.1f}%)\n"
#         description += f"The voice conveys {emotion_descriptions[primary[0]]}.\n\n"
        
#         # Add secondary if significant
#         if secondary and secondary[1] > 0.15:
#             description += f"**Secondary Emotion:** {secondary[0].capitalize()} ({secondary[1]*100:.1f}%)\n"
#             description += f"There are also traces of {emotion_descriptions[secondary[0]]}.\n\n"
        
#         # Add transcribed text if available
#         if transcribed_text:
#             description += f"**Transcribed Text:** \"{transcribed_text}\"\n\n"
        
#         # Add all emotion scores
#         description += "**Full Emotion Breakdown:**\n"
#         for emotion, score in sorted_emotions:
#             bar_length = int(score * 20)
#             bar = "█" * bar_length + "░" * (20 - bar_length)
#             description += f"- {emotion.capitalize()}: {bar} {score*100:.1f}%\n"
        
#         return description
    
#     def record_audio(self, duration: int = 5) -> Optional[bytes]:
#         """
#         Record audio from microphone
        
#         Args:
#             duration: Recording duration in seconds
            
#         Returns:
#             Raw audio data
#         """
#         try:
#             with self.microphone as source:
#                 self.logger.info(f"Recording audio for {duration} seconds...")
#                 audio_data = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
#                 return audio_data.get_wav_data()
#         except Exception as e:
#             self.logger.error(f"Audio recording error: {str(e)}")
#             return None
    
#     def analyze_audio_chunk(self, audio_data: bytes = None) -> Dict:
#         """
#         Analyze emotion from audio chunk
        
#         Args:
#             audio_data: Raw audio data (if None, records new audio)
            
#         Returns:
#             Dict with 'emotions', 'description', and 'transcribed_text' keys
#         """
#         try:
#             # Record audio if not provided
#             if audio_data is None:
#                 audio_data = self.record_audio(duration=3)
                
#             if audio_data is None:
#                 return {
#                     'emotions': {'neutral': 1.0},
#                     'description': 'Could not record audio.',
#                     'transcribed_text': ''
#                 }
            
#             # Load audio data
#             try:
#                 with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
#                     temp_file.write(audio_data)
#                     temp_file_path = temp_file.name
                
#                 y, sr = librosa.load(temp_file_path, sr=self.sample_rate)
#                 os.unlink(temp_file_path)
#             except Exception as e:
#                 self.logger.warning(f"Could not process audio for features: {str(e)}")
#                 y, sr = np.array([]), self.sample_rate
            
#             # Extract acoustic features and classify
#             acoustic_emotions = {'neutral': 1.0}
#             if len(y) > 0:
#                 features = self.extract_audio_features(y, sr)
#                 acoustic_emotions = self.classify_acoustic_emotion(features)
            
#             # Speech to text and classify
#             text = self.speech_to_text(audio_data)
#             text_emotions = self.classify_text_emotion(text) if text else {'neutral': 1.0}
            
#             # Combine acoustic and text emotions (weighted average)
#             acoustic_weight = 0.7  # Increased from 0.6 since acoustic is more reliable now
#             text_weight = 0.3
            
#             all_emotions = set(list(acoustic_emotions.keys()) + list(text_emotions.keys()))
#             combined_emotions = {}
            
#             for emotion in all_emotions:
#                 acoustic_score = acoustic_emotions.get(emotion, 0.0)
#                 text_score = text_emotions.get(emotion, 0.0)
#                 combined_emotions[emotion] = (acoustic_weight * acoustic_score + 
#                                              text_weight * text_score)
            
#             # Normalize
#             total = sum(combined_emotions.values())
#             if total > 0:
#                 combined_emotions = {k: v/total for k, v in combined_emotions.items()}
            
#             # Generate description
#             description = self.get_emotion_description(combined_emotions, text)
            
#             # Log results
#             if text:
#                 self.logger.info(f"Transcribed text: {text}")
#             self.logger.info(f"Voice emotion analysis: {combined_emotions}")
            
#             return {
#                 'emotions': combined_emotions,
#                 'description': description,
#                 'transcribed_text': text or ''
#             }
            
#         except Exception as e:
#             self.logger.error(f"Audio analysis error: {str(e)}")
#             return {
#                 'emotions': {'neutral': 1.0},
#                 'description': f'Error analyzing audio: {str(e)}',
#                 'transcribed_text': ''
#             }
    
#     def cleanup(self):
#         """Clean up resources"""
#         # Clean up any temporary files or connections
#         pass
