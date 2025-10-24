import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

class EmotionTracker:
    """
    Tracks emotion patterns and provides insights into emotional well-being over time
    """
    
    def __init__(self, max_history_days: int = 30):
        """
        Initialize the emotion tracker
        
        Args:
            max_history_days: Maximum days of history to keep
        """
        self.max_history_days = max_history_days
        self.emotion_data = []
        self.session_start_time = datetime.now()
        
        # Standard emotion categories
        self.emotion_labels = [
            'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        ]
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Wellness metrics
        self.wellness_thresholds = {
            'positive_emotions': ['joy'],
            'negative_emotions': ['anger', 'sadness', 'fear', 'disgust'],
            'neutral_emotions': ['neutral', 'surprise']
        }
    
    def add_emotion(self, emotions: Dict[str, float], timestamp: Optional[datetime] = None, 
                   metadata: Optional[Dict] = None):
        """
        Add emotion data point to tracking history
        
        Args:
            emotions: Dictionary of emotion probabilities
            timestamp: When the emotion was detected (defaults to now)
            metadata: Additional metadata about the detection
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Create emotion record
            emotion_record = {
                'timestamp': timestamp,
                'emotions': emotions.copy(),
                'dominant_emotion': dominant_emotion[0],
                'dominant_confidence': dominant_emotion[1],
                'session_time': (timestamp - self.session_start_time).total_seconds(),
                'metadata': metadata or {}
            }
            
            self.emotion_data.append(emotion_record)
            
            # Clean old data
            self._cleanup_old_data()
            
            self.logger.debug(f"Added emotion data point: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error adding emotion data: {str(e)}")
    
    def _cleanup_old_data(self):
        """Remove emotion data older than max_history_days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            
            # Keep only recent data
            self.emotion_data = [
                record for record in self.emotion_data
                if record['timestamp'] >= cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
    
    def get_emotion_history(self, hours: int = 24) -> List[Dict]:
        """
        Get emotion history for specified time period
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of emotion records
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_data = [
                record for record in self.emotion_data
                if record['timestamp'] >= cutoff_time
            ]
            
            return recent_data
            
        except Exception as e:
            self.logger.error(f"Error getting emotion history: {str(e)}")
            return []
    
    def get_dominant_emotions_timeline(self, hours: int = 24) -> List[Tuple[datetime, str, float]]:
        """
        Get timeline of dominant emotions
        
        Args:
            hours: Number of hours to include
            
        Returns:
            List of (timestamp, emotion, confidence) tuples
        """
        try:
            history = self.get_emotion_history(hours)
            
            timeline = [
                (record['timestamp'], record['dominant_emotion'], record['dominant_confidence'])
                for record in history
            ]
            
            return timeline
            
        except Exception as e:
            self.logger.error(f"Error creating emotions timeline: {str(e)}")
            return []
    
    def calculate_emotion_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Calculate emotion statistics for specified time period
        
        Args:
            hours: Time period in hours
            
        Returns:
            Dictionary with emotion statistics
        """
        try:
            history = self.get_emotion_history(hours)
            
            if not history:
                return {
                    'message': 'No emotion data available for specified period',
                    'time_period_hours': hours
                }
            
            # Initialize statistics
            stats = {
                'time_period_hours': hours,
                'total_detections': len(history),
                'emotion_distribution': {},
                'average_confidence': 0.0,
                'most_common_emotion': None,
                'emotion_changes': 0,
                'wellness_score': 0.0,
                'positive_emotion_ratio': 0.0,
                'negative_emotion_ratio': 0.0,
                'session_duration_minutes': 0.0
            }
            
            # Calculate session duration
            if history:
                start_time = min(record['timestamp'] for record in history)
                end_time = max(record['timestamp'] for record in history)
                stats['session_duration_minutes'] = (end_time - start_time).total_seconds() / 60
            
            # Emotion distribution and confidence
            emotion_counts = {}
            confidence_sum = 0.0
            
            for record in history:
                emotion = record['dominant_emotion']
                confidence = record['dominant_confidence']
                
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                confidence_sum += confidence
            
            # Calculate percentages
            total_detections = len(history)
            stats['emotion_distribution'] = {
                emotion: count / total_detections
                for emotion, count in emotion_counts.items()
            }
            
            stats['average_confidence'] = confidence_sum / total_detections
            stats['most_common_emotion'] = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate emotion changes (volatility)
            if len(history) > 1:
                changes = 0
                for i in range(1, len(history)):
                    if history[i]['dominant_emotion'] != history[i-1]['dominant_emotion']:
                        changes += 1
                stats['emotion_changes'] = changes
            
            # Wellness metrics
            positive_count = sum(
                emotion_counts.get(emotion, 0) 
                for emotion in self.wellness_thresholds['positive_emotions']
            )
            
            negative_count = sum(
                emotion_counts.get(emotion, 0)
                for emotion in self.wellness_thresholds['negative_emotions']
            )
            
            stats['positive_emotion_ratio'] = positive_count / total_detections
            stats['negative_emotion_ratio'] = negative_count / total_detections
            
            # Simple wellness score (0-100)
            # Higher positive emotions and lower negative emotions = higher score
            wellness_score = (
                stats['positive_emotion_ratio'] * 100 - 
                stats['negative_emotion_ratio'] * 50 +
                (1 - min(stats['emotion_changes'] / max(1, total_detections), 1)) * 20
            )
            stats['wellness_score'] = max(0, min(100, wellness_score))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating emotion statistics: {str(e)}")
            return {'error': str(e)}
    
    def get_emotion_trends(self, hours: int = 24, window_minutes: int = 30) -> Dict[str, List]:
        """
        Get emotion trends over time using sliding window
        
        Args:
            hours: Time period to analyze
            window_minutes: Window size for trend analysis
            
        Returns:
            Dictionary with trend data
        """
        try:
            history = self.get_emotion_history(hours)
            
            if not history:
                return {'message': 'No data available for trend analysis'}
            
            # Create time windows
            start_time = min(record['timestamp'] for record in history)
            end_time = max(record['timestamp'] for record in history)
            window_delta = timedelta(minutes=window_minutes)
            
            trends = {
                'timestamps': [],
                'dominant_emotions': [],
                'confidence_levels': [],
                'wellness_scores': []
            }
            
            current_time = start_time
            
            while current_time <= end_time:
                window_end = current_time + window_delta
                
                # Get data points in this window
                window_data = [
                    record for record in history
                    if current_time <= record['timestamp'] < window_end
                ]
                
                if window_data:
                    # Find most common emotion in window
                    emotion_counts = {}
                    confidence_sum = 0.0
                    
                    for record in window_data:
                        emotion = record['dominant_emotion']
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        confidence_sum += record['dominant_confidence']
                    
                    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                    avg_confidence = confidence_sum / len(window_data)
                    
                    # Calculate mini wellness score for window
                    positive_count = sum(
                        emotion_counts.get(emotion, 0)
                        for emotion in self.wellness_thresholds['positive_emotions']
                    )
                    negative_count = sum(
                        emotion_counts.get(emotion, 0)
                        for emotion in self.wellness_thresholds['negative_emotions']
                    )
                    
                    total_count = len(window_data)
                    pos_ratio = positive_count / total_count
                    neg_ratio = negative_count / total_count
                    
                    wellness_score = max(0, min(100, pos_ratio * 100 - neg_ratio * 50))
                    
                    trends['timestamps'].append(current_time)
                    trends['dominant_emotions'].append(dominant_emotion)
                    trends['confidence_levels'].append(avg_confidence)
                    trends['wellness_scores'].append(wellness_score)
                
                current_time += window_delta
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating emotion trends: {str(e)}")
            return {'error': str(e)}
    
    def get_wellness_insights(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get wellness insights and recommendations
        
        Args:
            hours: Time period for analysis
            
        Returns:
            Dictionary with wellness insights
        """
        try:
            stats = self.calculate_emotion_statistics(hours)
            
            if 'error' in stats or 'message' in stats:
                return stats
            
            insights = {
                'overall_wellness': 'unknown',
                'wellness_score': stats['wellness_score'],
                'key_insights': [],
                'recommendations': [],
                'emotional_stability': 'unknown',
                'dominant_mood_pattern': stats['most_common_emotion']
            }
            
            # Determine overall wellness
            if stats['wellness_score'] >= 70:
                insights['overall_wellness'] = 'good'
            elif stats['wellness_score'] >= 50:
                insights['overall_wellness'] = 'moderate'
            else:
                insights['overall_wellness'] = 'needs_attention'
            
            # Emotional stability analysis
            volatility_ratio = stats['emotion_changes'] / max(1, stats['total_detections'])
            
            if volatility_ratio < 0.3:
                insights['emotional_stability'] = 'stable'
            elif volatility_ratio < 0.6:
                insights['emotional_stability'] = 'moderate'
            else:
                insights['emotional_stability'] = 'volatile'
            
            # Generate insights
            if stats['positive_emotion_ratio'] > 0.6:
                insights['key_insights'].append("You're experiencing predominantly positive emotions today.")
            elif stats['negative_emotion_ratio'] > 0.6:
                insights['key_insights'].append("You're experiencing more negative emotions than usual.")
            
            if stats['emotion_changes'] > stats['total_detections'] * 0.5:
                insights['key_insights'].append("Your emotions have been changing frequently.")
            
            if stats['average_confidence'] > 0.8:
                insights['key_insights'].append("The emotion detection confidence has been high.")
            
            # Generate recommendations
            if insights['overall_wellness'] == 'needs_attention':
                insights['recommendations'].append("Consider taking breaks and practicing stress management techniques.")
            
            if insights['emotional_stability'] == 'volatile':
                insights['recommendations'].append("Try grounding exercises to help stabilize your emotional state.")
            
            if stats['negative_emotion_ratio'] > 0.4:
                insights['recommendations'].append("Focus on activities that bring you joy and relaxation.")
            
            if stats['session_duration_minutes'] > 60:
                insights['recommendations'].append("You've been actively using the system - remember to take regular breaks.")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating wellness insights: {str(e)}")
            return {'error': str(e)}
    
    def export_data(self, hours: int = 24) -> str:
        """
        Export emotion data to JSON format
        
        Args:
            hours: Hours of data to export
            
        Returns:
            JSON string of emotion data
        """
        try:
            history = self.get_emotion_history(hours)
            
            # Convert datetime objects to strings for JSON serialization
            export_data = []
            for record in history:
                export_record = record.copy()
                export_record['timestamp'] = record['timestamp'].isoformat()
                export_data.append(export_record)
            
            # Add metadata
            export_package = {
                'export_timestamp': datetime.now().isoformat(),
                'time_period_hours': hours,
                'total_records': len(export_data),
                'data': export_data
            }
            
            return json.dumps(export_package, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session
        
        Returns:
            Dictionary with session summary
        """
        try:
            current_time = datetime.now()
            session_duration = (current_time - self.session_start_time).total_seconds() / 60
            
            # Get all session data
            session_data = [
                record for record in self.emotion_data
                if record['timestamp'] >= self.session_start_time
            ]
            
            if not session_data:
                return {
                    'session_duration_minutes': session_duration,
                    'message': 'No emotions detected in current session'
                }
            
            # Calculate session statistics
            emotion_counts = {}
            total_confidence = 0.0
            
            for record in session_data:
                emotion = record['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence += record['dominant_confidence']
            
            summary = {
                'session_duration_minutes': session_duration,
                'total_detections': len(session_data),
                'emotions_detected': list(emotion_counts.keys()),
                'most_frequent_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0],
                'average_confidence': total_confidence / len(session_data),
                'emotion_distribution': {
                    emotion: count / len(session_data)
                    for emotion, count in emotion_counts.items()
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting session summary: {str(e)}")
            return {'error': str(e)}
    
    def reset(self):
        """Reset all tracking data and start new session"""
        self.emotion_data = []
        self.session_start_time = datetime.now()
        self.logger.info("Emotion tracker reset - new session started")
    
    def cleanup(self):
        """Clean up resources"""
        self.emotion_data = []
