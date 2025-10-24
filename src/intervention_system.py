import random
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

class InterventionSystem:
    """
    Personalized wellness intervention system that provides motivational quotes,
    music recommendations, and wellness tips based on detected emotions
    """
    
    def __init__(self):
        """Initialize the intervention system with content databases"""
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load intervention content
        self.quotes = self._load_quotes()
        self.music_recommendations = self._load_music_recommendations()
        self.wellness_tips = self._load_wellness_tips()
        self.breathing_exercises = self._load_breathing_exercises()
        self.activities = self._load_activities()
        
        # Intervention history for personalization
        self.intervention_history = []
        self.user_preferences = {}
        
    def _load_quotes(self) -> Dict[str, List[Dict[str, str]]]:
        """Load motivational quotes categorized by emotion"""
        
        return {
            'sadness': [
                {
                    'text': 'The wound is the place where the Light enters you.',
                    'author': 'Rumi'
                },
                {
                    'text': 'You are braver than you believe, stronger than you seem, and smarter than you think.',
                    'author': 'A.A. Milne'
                },
                {
                    'text': 'Every storm runs out of rain.',
                    'author': 'Maya Angelou'
                },
                {
                    'text': 'The way I see it, if you want the rainbow, you gotta put up with the rain.',
                    'author': 'Dolly Parton'
                },
                {
                    'text': 'Tomorrow will be better. But you have to make it through today first.',
                    'author': 'Unknown'
                }
            ],
            'anger': [
                {
                    'text': 'For every minute you are angry you lose sixty seconds of happiness.',
                    'author': 'Ralph Waldo Emerson'
                },
                {
                    'text': 'Holding on to anger is like grasping a hot coal with the intent of throwing it at someone else; you are the one who gets burned.',
                    'author': 'Buddha'
                },
                {
                    'text': 'The best revenge is massive success.',
                    'author': 'Frank Sinatra'
                },
                {
                    'text': 'You have power over your mind - not outside events. Realize this, and you will find strength.',
                    'author': 'Marcus Aurelius'
                },
                {
                    'text': 'Speak when you are angry and you will make the best speech you will ever regret.',
                    'author': 'Ambrose Bierce'
                }
            ],
            'fear': [
                {
                    'text': 'The cave you fear to enter holds the treasure you seek.',
                    'author': 'Joseph Campbell'
                },
                {
                    'text': 'Fear is only as deep as the mind allows.',
                    'author': 'Japanese Proverb'
                },
                {
                    'text': 'You gain strength, courage, and confidence by every experience in which you really stop to look fear in the face.',
                    'author': 'Eleanor Roosevelt'
                },
                {
                    'text': 'The only thing we have to fear is fear itself.',
                    'author': 'Franklin D. Roosevelt'
                },
                {
                    'text': 'Courage is not the absence of fear, but the mastery of it.',
                    'author': 'Mark Twain'
                }
            ],
            'joy': [
                {
                    'text': 'The most wasted of all days is one without laughter.',
                    'author': 'E.E. Cummings'
                },
                {
                    'text': 'Happiness is not something ready made. It comes from your own actions.',
                    'author': 'Dalai Lama'
                },
                {
                    'text': 'Joy is what happens to us when we allow ourselves to recognize how good things really are.',
                    'author': 'Marianne Williamson'
                },
                {
                    'text': 'Find ecstasy in life; the mere sense of living is joy enough.',
                    'author': 'Emily Dickinson'
                },
                {
                    'text': 'Today is a good day to have a good day.',
                    'author': 'Unknown'
                }
            ],
            'neutral': [
                {
                    'text': 'Every moment is a fresh beginning.',
                    'author': 'T.S. Eliot'
                },
                {
                    'text': 'The present moment is the only moment available to us, and it is the door to all moments.',
                    'author': 'Thich Nhat Hanh'
                },
                {
                    'text': 'Life is what happens when you are busy making other plans.',
                    'author': 'John Lennon'
                },
                {
                    'text': 'Be yourself; everyone else is already taken.',
                    'author': 'Oscar Wilde'
                },
                {
                    'text': 'The journey of a thousand miles begins with one step.',
                    'author': 'Lao Tzu'
                }
            ],
            'surprise': [
                {
                    'text': 'Life is full of surprises, and the quality of your life depends on how you handle those surprises.',
                    'author': 'Unknown'
                },
                {
                    'text': 'The unexpected is just another way of life saying, "Plot twist!"',
                    'author': 'Unknown'
                },
                {
                    'text': 'Embrace uncertainty. Some of the most beautiful chapters in our lives won\'t have a title until much later.',
                    'author': 'Bob Goff'
                }
            ],
            'disgust': [
                {
                    'text': 'Sometimes you need to step outside, get some air, and remind yourself of who you are and where you want to be.',
                    'author': 'Unknown'
                },
                {
                    'text': 'What lies behind us and what lies before us are tiny matters compared to what lies within us.',
                    'author': 'Ralph Waldo Emerson'
                },
                {
                    'text': 'You cannot control what happens to you, but you can control your attitude toward what happens to you.',
                    'author': 'Brian Tracy'
                }
            ]
        }
    
    def _load_music_recommendations(self) -> Dict[str, List[Dict[str, str]]]:
        """Load music recommendations categorized by emotion"""
        
        return {
            'sadness': [
                {
                    'genre': 'Ambient',
                    'mood': 'Melancholic & Healing',
                    'description': 'Soft ambient sounds to process emotions',
                    'examples': 'Brian Eno, Stars of the Lid, Tim Hecker'
                },
                {
                    'genre': 'Classical',
                    'mood': 'Contemplative',
                    'description': 'Gentle classical pieces for reflection',
                    'examples': 'Chopin Nocturnes, Debussy Clair de Lune'
                },
                {
                    'genre': 'Folk',
                    'mood': 'Comforting',
                    'description': 'Warm folk music for emotional support',
                    'examples': 'Nick Drake, Bon Iver, Iron & Wine'
                }
            ],
            'anger': [
                {
                    'genre': 'Classical',
                    'mood': 'Calming',
                    'description': 'Peaceful classical music to reduce tension',
                    'examples': 'Bach Air on G String, Pachelbel Canon'
                },
                {
                    'genre': 'Nature Sounds',
                    'mood': 'Grounding',
                    'description': 'Natural sounds to restore inner peace',
                    'examples': 'Ocean waves, forest sounds, rain'
                },
                {
                    'genre': 'Meditation Music',
                    'mood': 'Centering',
                    'description': 'Meditative sounds to find balance',
                    'examples': 'Tibetan bowls, soft instrumental'
                }
            ],
            'fear': [
                {
                    'genre': 'Uplifting Pop',
                    'mood': 'Encouraging',
                    'description': 'Positive, empowering songs',
                    'examples': 'Upbeat, motivational tracks'
                },
                {
                    'genre': 'Gospel/Spiritual',
                    'mood': 'Reassuring',
                    'description': 'Spiritual music for comfort and strength',
                    'examples': 'Amazing Grace, inspirational hymns'
                },
                {
                    'genre': 'Soft Rock',
                    'mood': 'Supportive',
                    'description': 'Gentle rock music with positive messages',
                    'examples': 'The Beatles, James Taylor'
                }
            ],
            'joy': [
                {
                    'genre': 'Upbeat Pop',
                    'mood': 'Celebratory',
                    'description': 'Energetic music to amplify happiness',
                    'examples': 'Feel-good pop hits, dance music'
                },
                {
                    'genre': 'Reggae',
                    'mood': 'Joyful',
                    'description': 'Relaxed, positive vibes',
                    'examples': 'Bob Marley, Jimmy Buffett'
                },
                {
                    'genre': 'World Music',
                    'mood': 'Vibrant',
                    'description': 'Diverse, uplifting world sounds',
                    'examples': 'Afrobeat, Latin music, Celtic'
                }
            ],
            'neutral': [
                {
                    'genre': 'Instrumental',
                    'mood': 'Focused',
                    'description': 'Background music for concentration',
                    'examples': 'Piano instrumentals, guitar fingerpicking'
                },
                {
                    'genre': 'Jazz',
                    'mood': 'Smooth',
                    'description': 'Relaxed jazz for a calm atmosphere',
                    'examples': 'Miles Davis, Bill Evans, Norah Jones'
                },
                {
                    'genre': 'Lo-fi Hip Hop',
                    'mood': 'Chill',
                    'description': 'Mellow beats for relaxation',
                    'examples': 'Study playlists, chillhop'
                }
            ],
            'surprise': [
                {
                    'genre': 'Electronic',
                    'mood': 'Exciting',
                    'description': 'Dynamic electronic music',
                    'examples': 'Upbeat EDM, synthwave'
                },
                {
                    'genre': 'World Fusion',
                    'mood': 'Adventurous',
                    'description': 'Unique blend of cultural sounds',
                    'examples': 'World fusion, experimental music'
                }
            ],
            'disgust': [
                {
                    'genre': 'Clean Instrumental',
                    'mood': 'Purifying',
                    'description': 'Pure, clean sounds to refresh',
                    'examples': 'Classical guitar, flute music'
                },
                {
                    'genre': 'Nature Sounds',
                    'mood': 'Cleansing',
                    'description': 'Natural sounds for mental cleansing',
                    'examples': 'Mountain streams, wind through trees'
                }
            ]
        }
    
    def _load_wellness_tips(self) -> Dict[str, List[str]]:
        """Load wellness tips categorized by emotion"""
        
        return {
            'sadness': [
                'Take a warm bath or shower to comfort yourself physically.',
                'Reach out to a trusted friend or family member for support.',
                'Practice gentle self-care activities like reading or listening to music.',
                'Go for a walk in nature to change your environment.',
                'Write in a journal to express your feelings.',
                'Allow yourself to feel the emotion without judgment.',
                'Engage in light physical activity like stretching or yoga.',
                'Practice gratitude by listing three things you appreciate.'
            ],
            'anger': [
                'Take slow, deep breaths to activate your parasympathetic nervous system.',
                'Count to ten before responding to reduce impulsive reactions.',
                'Step away from the situation if possible to gain perspective.',
                'Practice progressive muscle relaxation to release physical tension.',
                'Channel your energy into physical exercise like walking or running.',
                'Use "I" statements to express your feelings constructively.',
                'Practice mindfulness to observe your anger without being controlled by it.',
                'Consider the underlying need or value that\'s being threatened.'
            ],
            'fear': [
                'Practice grounding techniques: name 5 things you see, 4 you hear, 3 you touch.',
                'Challenge fearful thoughts with evidence-based reasoning.',
                'Break overwhelming tasks into smaller, manageable steps.',
                'Practice square breathing: inhale 4, hold 4, exhale 4, hold 4.',
                'Visualize yourself successfully handling the feared situation.',
                'Seek support from trusted friends, family, or professionals.',
                'Focus on what you can control in the present moment.',
                'Remember past times when you overcame difficult challenges.'
            ],
            'joy': [
                'Share your happiness with others to multiply the positive feeling.',
                'Capture the moment through photos, journaling, or mindful awareness.',
                'Use this positive energy to tackle something you\'ve been putting off.',
                'Practice gratitude for the people and circumstances that brought this joy.',
                'Plan future activities that might bring similar happiness.',
                'Pay it forward by doing something kind for someone else.',
                'Savor the feeling fully without rushing to the next thing.',
                'Create a memory or ritual to help you recall this moment later.'
            ],
            'neutral': [
                'Take this calm moment to check in with your body and mind.',
                'Set small, achievable goals for personal growth.',
                'Practice mindfulness meditation to stay present.',
                'Engage in a creative activity that interests you.',
                'Organize your space to create a sense of order.',
                'Learn something new through reading or online courses.',
                'Plan future activities that align with your values.',
                'Practice gentle movement like stretching or walking.'
            ],
            'surprise': [
                'Take a moment to fully process what just happened.',
                'Notice your body\'s response and breathe deeply.',
                'Consider what you can learn from this unexpected experience.',
                'Share the experience with someone if it feels appropriate.',
                'Use this moment of heightened awareness mindfully.',
                'Adapt your plans flexibly if needed.',
                'Find the opportunity or positive aspect in the surprise.',
                'Stay curious rather than immediately judging the situation.'
            ],
            'disgust': [
                'Remove yourself from the triggering situation if possible.',
                'Focus on clean, pleasant sensory experiences.',
                'Practice deep breathing with visualization of fresh air.',
                'Engage in activities that feel purifying or refreshing.',
                'Remind yourself of your values and what matters to you.',
                'Take a shower or wash your hands mindfully.',
                'Listen to uplifting music or look at beautiful images.',
                'Practice self-compassion for your emotional response.'
            ]
        }
    
    def _load_breathing_exercises(self) -> Dict[str, Dict[str, Any]]:
        """Load breathing exercises categorized by emotion"""
        
        return {
            'sadness': {
                'name': 'Gentle Heart Breathing',
                'duration': 300,  # 5 minutes
                'instructions': 'Place one hand on your heart. Breathe slowly and deeply, imagining warm, healing light flowing to your heart with each breath. Let each exhale release any heaviness you\'re carrying.',
                'pattern': 'Inhale for 4 counts, hold for 2, exhale for 6'
            },
            'anger': {
                'name': 'Cooling Breath',
                'duration': 180,  # 3 minutes
                'instructions': 'Imagine breathing in cool, calming air and exhaling hot, tense energy. Let each breath cool your inner fire and bring peace to your mind and body.',
                'pattern': 'Inhale for 4 counts, hold for 4, exhale for 8'
            },
            'fear': {
                'name': 'Grounding Square Breath',
                'duration': 240,  # 4 minutes
                'instructions': 'Visualize drawing a square with your breath. This equal-sided breathing pattern helps ground you and restore a sense of safety and control.',
                'pattern': 'Inhale for 4 counts, hold for 4, exhale for 4, hold for 4'
            },
            'joy': {
                'name': 'Energizing Breath',
                'duration': 120,  # 2 minutes
                'instructions': 'Breathe with enthusiasm and gratitude. Let each breath amplify your positive energy and spread joy throughout your entire being.',
                'pattern': 'Inhale for 3 counts, brief hold, exhale for 3 counts'
            },
            'neutral': {
                'name': 'Balanced Breath',
                'duration': 300,  # 5 minutes
                'instructions': 'Focus on natural, balanced breathing. This is your baseline - peaceful, steady, and centered. Use this time for gentle self-awareness.',
                'pattern': 'Inhale for 4 counts, exhale for 4 counts'
            }
        }
    
    def _load_activities(self) -> Dict[str, List[str]]:
        """Load recommended activities categorized by emotion"""
        
        return {
            'sadness': [
                'Take a warm, comforting shower or bath',
                'Call or text a supportive friend or family member',
                'Watch a favorite comfort movie or show',
                'Make yourself a hot cup of tea or cocoa',
                'Snuggle with a pet or soft blanket',
                'Look at photos of happy memories',
                'Do some gentle stretching or restorative yoga'
            ],
            'anger': [
                'Go for a brisk walk or run',
                'Do jumping jacks or other vigorous exercise',
                'Practice punching a pillow (safely)',
                'Write angry thoughts in a journal, then tear up the pages',
                'Take a cold shower to reset your system',
                'Listen to calming music',
                'Practice progressive muscle relaxation'
            ],
            'fear': [
                'Call someone who makes you feel safe',
                'Do a grounding exercise (5-4-3-2-1 senses)',
                'Engage in a familiar, comforting routine',
                'Write down your fears, then write rational responses',
                'Do something that makes you feel capable and strong',
                'Practice visualization of positive outcomes',
                'Take small, manageable action steps forward'
            ],
            'joy': [
                'Dance to your favorite upbeat music',
                'Share your good news with loved ones',
                'Take photos of things that make you smile',
                'Plan future activities you\'re excited about',
                'Do something creative or artistic',
                'Help someone else or volunteer',
                'Spend time outdoors enjoying nature'
            ],
            'neutral': [
                'Try a new hobby or skill',
                'Organize and declutter a space',
                'Read an interesting book or article',
                'Do a puzzle or play a brain game',
                'Plan your week or upcoming goals',
                'Practice a mindfulness meditation',
                'Take a contemplative walk'
            ]
        }
    
    def get_interventions(self, emotions: Dict[str, float], 
                         intervention_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get personalized interventions based on detected emotions
        
        Args:
            emotions: Dictionary of emotion probabilities
            intervention_types: List of specific intervention types to include
            
        Returns:
            Dictionary of recommended interventions
        """
        try:
            if not emotions:
                return {}
            
            # Determine dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name = dominant_emotion[0]
            confidence = dominant_emotion[1]
            
            # Default intervention types
            if intervention_types is None:
                intervention_types = ['quote', 'music', 'wellness_tip', 'breathing_exercise', 'activity']
            
            interventions = {}
            
            # Get quote
            if 'quote' in intervention_types and emotion_name in self.quotes:
                quotes = self.quotes[emotion_name]
                selected_quote = random.choice(quotes)
                interventions['quote'] = selected_quote
            
            # Get music recommendation
            if 'music' in intervention_types and emotion_name in self.music_recommendations:
                music_options = self.music_recommendations[emotion_name]
                selected_music = random.choice(music_options)
                interventions['music'] = selected_music
            
            # Get wellness tip
            if 'wellness_tip' in intervention_types and emotion_name in self.wellness_tips:
                tips = self.wellness_tips[emotion_name]
                selected_tip = random.choice(tips)
                interventions['wellness_tip'] = selected_tip
            
            # Get breathing exercise
            if 'breathing_exercise' in intervention_types and emotion_name in self.breathing_exercises:
                exercise = self.breathing_exercises[emotion_name]
                interventions['breathing_exercise'] = exercise
            
            # Get activity recommendation
            if 'activity' in intervention_types and emotion_name in self.activities:
                activities = self.activities[emotion_name]
                selected_activity = random.choice(activities)
                interventions['activity'] = selected_activity
            
            # Store intervention history for personalization
            self.intervention_history.append({
                'timestamp': datetime.now(),
                'emotion': emotion_name,
                'confidence': confidence,
                'interventions': interventions.copy()
            })
            
            # Limit history size
            if len(self.intervention_history) > 50:
                self.intervention_history = self.intervention_history[-50:]
            
            self.logger.info(f"Generated interventions for {emotion_name}: {list(interventions.keys())}")
            
            return interventions
            
        except Exception as e:
            self.logger.error(f"Error generating interventions: {str(e)}")
            return {}
    
    def get_personalized_interventions(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """
        Get interventions personalized based on user history and preferences
        
        Args:
            emotions: Dictionary of emotion probabilities
            
        Returns:
            Personalized intervention recommendations
        """
        try:
            # Get base interventions
            interventions = self.get_interventions(emotions)
            
            if not self.intervention_history:
                return interventions
            
            # Analyze user preferences from history
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Find similar past situations
            similar_interventions = [
                record for record in self.intervention_history[-20:]  # Last 20 records
                if record['emotion'] == dominant_emotion
            ]
            
            # If we have history for this emotion, potentially adjust recommendations
            if similar_interventions:
                # Could implement more sophisticated personalization here
                # For now, we'll add a personalization note
                interventions['personalization_note'] = f"Based on your history with {dominant_emotion}, this approach has been helpful before."
            
            return interventions
            
        except Exception as e:
            self.logger.error(f"Error generating personalized interventions: {str(e)}")
            return self.get_interventions(emotions)
    
    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get statistics about intervention usage"""
        
        try:
            if not self.intervention_history:
                return {'message': 'No intervention history available'}
            
            from collections import Counter
            
            # Count emotions addressed
            emotions_addressed = [record['emotion'] for record in self.intervention_history]
            emotion_counts = Counter(emotions_addressed)
            
            # Count intervention types used
            intervention_types = []
            for record in self.intervention_history:
                intervention_types.extend(record['interventions'].keys())
            
            type_counts = Counter(intervention_types)
            
            stats = {
                'total_interventions': len(self.intervention_history),
                'emotions_addressed': dict(emotion_counts),
                'intervention_types_used': dict(type_counts),
                'most_common_emotion': emotion_counts.most_common(1)[0] if emotion_counts else None,
                'most_used_intervention': type_counts.most_common(1)[0] if type_counts else None
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating intervention stats: {str(e)}")
            return {'error': str(e)}
    
    def reset_history(self):
        """Reset intervention history"""
        self.intervention_history = []
        self.user_preferences = {}
        self.logger.info("Intervention history reset")
    
    def cleanup(self):
        """Clean up resources"""
        self.reset_history()
