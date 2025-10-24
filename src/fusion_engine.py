import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta

class EmotionFusionEngine:
    """
    Multi-modal emotion fusion engine that combines predictions from
    facial, voice, and text emotion detectors using weighted voting
    """
    
    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the emotion fusion engine
        
        Args:
            default_weights: Default weights for each modality
        """
        # Default weights for each modality
        self.default_weights = default_weights or {
            'face': 0.4,
            'voice': 0.3,
            'text': 0.3
        }
        
        self.current_weights = self.default_weights.copy()
        
        # Standard emotion labels across all modalities
        self.emotion_labels = [
            'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        ]
        
        # Confidence thresholds
        self.confidence_threshold = 0.1
        self.fusion_threshold = 0.05
        
        # History for temporal fusion
        self.emotion_history = []
        self.history_window = 10  # Keep last 10 fusion results
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set custom weights for modalities
        
        Args:
            weights: Dictionary of modality weights
        """
        try:
            # Validate weights
            valid_modalities = {'face', 'voice', 'text'}
            
            for modality in weights:
                if modality not in valid_modalities:
                    self.logger.warning(f"Unknown modality: {modality}")
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in weights.items()}
                self.current_weights.update(normalized_weights)
                self.logger.info(f"Updated fusion weights: {self.current_weights}")
            else:
                self.logger.warning("Invalid weights provided, using defaults")
        
        except Exception as e:
            self.logger.error(f"Error setting weights: {str(e)}")
    
    def normalize_emotions(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize emotion probabilities to ensure they sum to 1
        
        Args:
            emotions: Raw emotion scores
            
        Returns:
            Normalized emotion probabilities
        """
        try:
            # Ensure all standard emotions are present
            normalized = {}
            for emotion in self.emotion_labels:
                normalized[emotion] = emotions.get(emotion, 0.0)
            
            # Add any additional emotions from input
            for emotion, score in emotions.items():
                if emotion not in normalized:
                    normalized[emotion] = score
            
            # Normalize to sum to 1
            total = sum(normalized.values())
            if total > 0:
                normalized = {k: v/total for k, v in normalized.items()}
            else:
                # If all zeros, set neutral to 1
                normalized = {k: 0.0 for k in normalized}
                normalized['neutral'] = 1.0
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing emotions: {str(e)}")
            return {'neutral': 1.0}
    
    def map_emotions(self, emotions: Dict[str, float], source_modality: str) -> Dict[str, float]:
        """
        Map emotions from different modalities to standard labels
        
        Args:
            emotions: Raw emotion predictions
            source_modality: Source modality ('face', 'voice', 'text')
            
        Returns:
            Mapped emotion dictionary
        """
        try:
            mapped_emotions = {}
            
            # Emotion mapping for different modalities
            emotion_mappings = {
                'face': {
                    'angry': 'anger',
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'happy': 'joy',
                    'sad': 'sadness',
                    'surprise': 'surprise',
                    'neutral': 'neutral'
                },
                'voice': {
                    'anger': 'anger',
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'joy': 'joy',
                    'neutral': 'neutral',
                    'sadness': 'sadness',
                    'surprise': 'surprise'
                },
                'text': {
                    'anger': 'anger',
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'joy': 'joy',
                    'neutral': 'neutral',
                    'sadness': 'sadness',
                    'surprise': 'surprise'
                }
            }
            
            modality_mapping = emotion_mappings.get(source_modality, {})
            
            # Map emotions
            for emotion, score in emotions.items():
                mapped_emotion = modality_mapping.get(emotion.lower(), emotion.lower())
                
                if mapped_emotion in mapped_emotions:
                    mapped_emotions[mapped_emotion] += score
                else:
                    mapped_emotions[mapped_emotion] = score
            
            return self.normalize_emotions(mapped_emotions)
            
        except Exception as e:
            self.logger.error(f"Error mapping emotions: {str(e)}")
            return self.normalize_emotions(emotions)
    
    def calculate_confidence_weights(self, modality_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate confidence-based weights for each modality
        
        Args:
            modality_data: Dictionary of modality predictions
            
        Returns:
            Confidence-adjusted weights
        """
        try:
            confidence_weights = {}
            
            for modality, emotions in modality_data.items():
                if emotions:
                    # Calculate confidence as max probability
                    max_confidence = max(emotions.values())
                    
                    # Calculate entropy as measure of certainty
                    entropy = -sum(p * np.log(p + 1e-10) for p in emotions.values() if p > 0)
                    max_entropy = np.log(len(emotions))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
                    
                    # Combine confidence metrics
                    confidence = max_confidence * (1 - normalized_entropy)
                    confidence_weights[modality] = confidence
                else:
                    confidence_weights[modality] = 0.0
            
            return confidence_weights
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence weights: {str(e)}")
            return {modality: 1.0 for modality in modality_data.keys()}
    
    def fuse_emotions(self, modality_data: Dict[str, Dict[str, float]], 
                     use_confidence_weighting: bool = True) -> Dict[str, float]:
        """
        Fuse emotions from multiple modalities
        
        Args:
            modality_data: Dictionary containing emotion predictions from each modality
            use_confidence_weighting: Whether to use confidence-based weighting
            
        Returns:
            Fused emotion probabilities
        """
        try:
            if not modality_data:
                return {'neutral': 1.0}
            
            # Normalize and map emotions for each modality
            processed_data = {}
            for modality, emotions in modality_data.items():
                if emotions:
                    mapped_emotions = self.map_emotions(emotions, modality)
                    processed_data[modality] = mapped_emotions
            
            if not processed_data:
                return {'neutral': 1.0}
            
            # Calculate weights
            if use_confidence_weighting:
                confidence_weights = self.calculate_confidence_weights(processed_data)
                
                # Combine base weights with confidence weights
                final_weights = {}
                total_confidence = sum(confidence_weights.values())
                
                for modality in processed_data:
                    base_weight = self.current_weights.get(modality, 0.0)
                    confidence_weight = confidence_weights.get(modality, 0.0)
                    
                    if total_confidence > 0:
                        confidence_factor = confidence_weight / total_confidence
                        final_weights[modality] = 0.5 * base_weight + 0.5 * confidence_factor
                    else:
                        final_weights[modality] = base_weight
            else:
                final_weights = {modality: self.current_weights.get(modality, 0.0) 
                               for modality in processed_data}
            
            # Normalize weights
            total_weight = sum(final_weights.values())
            if total_weight > 0:
                final_weights = {k: v/total_weight for k, v in final_weights.items()}
            
            # Fuse emotions using weighted average
            fused_emotions = {}
            
            # Initialize with all emotion categories
            for emotion in self.emotion_labels:
                fused_emotions[emotion] = 0.0
            
            # Weighted fusion
            for modality, emotions in processed_data.items():
                weight = final_weights.get(modality, 0.0)
                
                for emotion, score in emotions.items():
                    if emotion in fused_emotions:
                        fused_emotions[emotion] += weight * score
                    else:
                        fused_emotions[emotion] = weight * score
            
            # Normalize final result
            fused_emotions = self.normalize_emotions(fused_emotions)
            
            # Apply temporal smoothing if we have history
            if self.emotion_history:
                fused_emotions = self.apply_temporal_smoothing(fused_emotions)
            
            # Store in history
            self.emotion_history.append({
                'timestamp': datetime.now(),
                'emotions': fused_emotions.copy(),
                'modalities': processed_data.copy(),
                'weights': final_weights.copy()
            })
            
            # Trim history
            if len(self.emotion_history) > self.history_window:
                self.emotion_history = self.emotion_history[-self.history_window:]
            
            # Log fusion results
            self.logger.info(f"Emotion fusion result: {fused_emotions}")
            self.logger.debug(f"Fusion weights: {final_weights}")
            
            return fused_emotions
            
        except Exception as e:
            self.logger.error(f"Error in emotion fusion: {str(e)}")
            return {'neutral': 1.0}
    
    def apply_temporal_smoothing(self, current_emotions: Dict[str, float], 
                                smoothing_factor: float = 0.3) -> Dict[str, float]:
        """
        Apply temporal smoothing to reduce emotion jitter
        
        Args:
            current_emotions: Current emotion predictions
            smoothing_factor: Weight for current emotions vs. history
            
        Returns:
            Temporally smoothed emotions
        """
        try:
            if not self.emotion_history:
                return current_emotions
            
            # Get recent emotions (last 3 measurements)
            recent_history = self.emotion_history[-3:]
            
            # Calculate historical average
            historical_emotions = {}
            for emotion in self.emotion_labels:
                historical_emotions[emotion] = 0.0
            
            for record in recent_history:
                for emotion, score in record['emotions'].items():
                    if emotion in historical_emotions:
                        historical_emotions[emotion] += score
            
            # Average
            history_count = len(recent_history)
            if history_count > 0:
                historical_emotions = {k: v/history_count for k, v in historical_emotions.items()}
            
            # Blend current with historical
            smoothed_emotions = {}
            for emotion in self.emotion_labels:
                current_score = current_emotions.get(emotion, 0.0)
                historical_score = historical_emotions.get(emotion, 0.0)
                
                smoothed_emotions[emotion] = (
                    smoothing_factor * current_score + 
                    (1 - smoothing_factor) * historical_score
                )
            
            return self.normalize_emotions(smoothed_emotions)
            
        except Exception as e:
            self.logger.error(f"Error in temporal smoothing: {str(e)}")
            return current_emotions
    
    def get_dominant_emotion(self, emotions: Dict[str, float]) -> Dict[str, Union[str, float]]:
        """
        Get the dominant emotion and its confidence
        
        Args:
            emotions: Emotion probability dictionary
            
        Returns:
            Dictionary with dominant emotion info
        """
        try:
            if not emotions:
                return {'emotion': 'neutral', 'confidence': 1.0, 'certainty': 'low'}
            
            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Calculate certainty based on confidence gap
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            certainty = 'low'
            if len(sorted_emotions) > 1:
                confidence_gap = sorted_emotions[0][1] - sorted_emotions[1][1]
                if confidence_gap > 0.3:
                    certainty = 'high'
                elif confidence_gap > 0.15:
                    certainty = 'medium'
            elif dominant_emotion[1] > 0.7:
                certainty = 'high'
            
            return {
                'emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'certainty': certainty,
                'all_emotions': emotions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dominant emotion: {str(e)}")
            return {'emotion': 'neutral', 'confidence': 1.0, 'certainty': 'low'}
    
    def get_fusion_statistics(self) -> Dict:
        """
        Get statistics about fusion performance
        
        Returns:
            Dictionary with fusion statistics
        """
        try:
            if not self.emotion_history:
                return {'message': 'No fusion history available'}
            
            stats = {
                'total_fusions': len(self.emotion_history),
                'time_span': None,
                'most_common_emotion': None,
                'average_confidence': 0.0,
                'modality_usage': {}
            }
            
            if self.emotion_history:
                # Time span
                first_time = self.emotion_history[0]['timestamp']
                last_time = self.emotion_history[-1]['timestamp']
                stats['time_span'] = (last_time - first_time).total_seconds()
                
                # Most common emotion
                all_emotions = []
                total_confidence = 0.0
                
                for record in self.emotion_history:
                    dominant = max(record['emotions'].items(), key=lambda x: x[1])
                    all_emotions.append(dominant[0])
                    total_confidence += dominant[1]
                
                from collections import Counter
                emotion_counts = Counter(all_emotions)
                stats['most_common_emotion'] = emotion_counts.most_common(1)[0] if emotion_counts else None
                stats['average_confidence'] = total_confidence / len(self.emotion_history)
                
                # Modality usage
                modality_counts = {}
                for record in self.emotion_history:
                    for modality in record['modalities']:
                        modality_counts[modality] = modality_counts.get(modality, 0) + 1
                
                stats['modality_usage'] = modality_counts
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting fusion statistics: {str(e)}")
            return {'error': str(e)}
    
    def reset_history(self):
        """Reset emotion history"""
        self.emotion_history = []
        self.logger.info("Emotion fusion history reset")
    
    def cleanup(self):
        """Clean up resources"""
        self.reset_history()
