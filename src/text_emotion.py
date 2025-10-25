import numpy as np
import logging
from typing import Dict, List, Optional
import re

# Try to import transformers and torch, but continue without them if unavailable
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers library not available: {e}. Using rule-based fallback.")
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None

class TextEmotionDetector:
    """
    Text emotion detection using pre-trained BERT-based models
    Analyzes emotional content in text input
    """
    
    def __init__(self, model_name: str = 'j-hartmann/emotion-english-distilroberta-base'):
        """
        Initialize the text emotion detector
        
        Args:
            model_name: HuggingFace model for emotion classification
        """
        self.model_name = model_name
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers library not available. Using rule-based emotion detection.")
            self.emotion_classifier = None
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Create pipeline for easier inference
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    top_k=None,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.logger.info(f"Loaded text emotion model: {model_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load emotion model: {str(e)}")
                self.emotion_classifier = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for emotion analysis
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
            
            # Remove excessive punctuation
            text = re.sub(r'([.!?]){2,}', r'\1', text)
            
            # Remove excessive capitalization (but preserve some for emotion detection)
            words = text.split()
            processed_words = []
            for word in words:
                if word.isupper() and len(word) > 3:
                    # Keep one uppercase version for emphasis detection
                    processed_words.append(word.lower() + "_EMPHASIS")
                else:
                    processed_words.append(word)
            
            text = ' '.join(processed_words)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Text preprocessing error: {str(e)}")
            return text
    
    def extract_text_features(self, text: str) -> Dict:
        """
        Extract linguistic features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        try:
            features = {}
            
            # Basic metrics
            features['length'] = len(text)
            features['word_count'] = len(text.split())
            features['sentence_count'] = len(re.split(r'[.!?]+', text))
            features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Punctuation analysis
            features['exclamation_count'] = text.count('!')
            features['question_count'] = text.count('?')
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            
            # Emotional indicators
            positive_words = [# Core emotions
    'happy', 'joy', 'love', 'excited', 'great', 'wonderful', 'amazing', 'good', 'best', 'excellent',
    
    # Synonyms & related adjectives
    'cheerful', 'delighted', 'content', 'pleased', 'ecstatic', 'elated', 'thrilled', 'satisfied',
    'glad', 'blissful', 'overjoyed', 'radiant', 'grateful', 'optimistic', 'hopeful', 'joyful',
    'vibrant', 'fantastic', 'superb', 'outstanding', 'splendid', 'marvelous', 'awesome',
    'impressive', 'fabulous', 'terrific', 'incredible', 'phenomenal', 'brilliant', 'spectacular',
    'admirable', 'commendable', 'beautiful', 'cute', 'lovely', 'inspiring', 'heartwarming',
    
    # Slang and casual positive terms
    'cool', 'dope', 'lit', 'fire', 'awesome', 'sweet', 'rad', 'goated', 'based', 'wholesome',
    'slay', 'pog', 'epic', 'win', 'yay', 'woah', 'wow',
    
    # Positive verbs and states
    'appreciate', 'enjoy', 'celebrate', 'achieve', 'succeed', 'improve', 'smile', 'laugh', 
    'progress', 'accomplish', 'grow', 'learn', 'help', 'support', 'care', 'respect', 'love',
    
    # Emotional intensifiers
    'absolutely', 'definitely', 'truly', 'deeply', 'completely', 'immensely', 'extremely', 
    'highly', 'really', 'super', 'ultra', 'mega', 'giga', 'fantastically',
    
    # Emoji cues
    '😊', '😄', '😁', '😃', '🤩', '🥰', '😍', '😘', '👍', '👏', '💪', '💖', '🎉', '✨', '❤️', '🌟', '🙌'] 
            
            negative_words = [
    # Core emotions
    'sad', 'angry', 'hate', 'terrible', 'awful', 'bad', 'worst', 'horrible', 'disgusting', 'fear',
    
    # Synonyms & related adjectives
    'depressed', 'unhappy', 'miserable', 'upset', 'frustrated', 'annoyed', 'irritated', 'mad',
    'furious', 'enraged', 'resentful', 'offended', 'heartbroken', 'hopeless', 'helpless',
    'devastated', 'lonely', 'disappointed', 'embarrassed', 'ashamed', 'guilty', 'insecure',
    'tired', 'bored', 'sick', 'gross', 'ugly', 'painful', 'displeased', 'worse', 'dreadful',
    'nasty', 'horrid', 'insulting', 'unbearable', 'tragic', 'annoying',
    
    # Slang and casual negative terms
    'meh', 'lame', 'cringe', 'trash', 'mid', 'sus', 'bs', 'fml', 'yuck', 'ew', 'ugh', 'damn', 
    'wtf', 'worst', 'fail', 'down', 'broke', 'lost',
    
    # Negative verbs and states
    'cry', 'suffer', 'struggle', 'fail', 'lose', 'hurt', 'hate', 'fear', 'ignore', 'break', 
    'ruin', 'complain', 'regret', 'blame', 'destroy', 'reject', 'avoid', 'curse',
    
    # Emotional intensifiers
    'completely', 'totally', 'utterly', 'deeply', 'extremely', 'horribly', 'terribly', 'awfully',
    'insanely', 'unbelievably', 'painfully',
    
    # Emoji cues
    '😢', '😭', '😞', '😔', '😠', '😡', '🤬', '💔', '👎', '😩', '😤', '😫', '😖', '😣', '😨', '😰'
]

            
            text_lower = text.lower()
            features['positive_word_count'] = sum(1 for word in positive_words if word in text_lower)
            features['negative_word_count'] = sum(1 for word in negative_words if word in text_lower)
            
            # Emphasis indicators
            features['caps_word_count'] = len([word for word in text.split() if word.isupper() and len(word) > 2])
            features['repeated_letters'] = len(re.findall(r'(.)\1{2,}', text))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            return {}
    
    def analyze_text(self, text: str, include_features: bool = True) -> Dict:
        """
        Analyze emotion in text
        
        Args:
            text: Input text to analyze
            include_features: Whether to include feature-based analysis
            
        Returns:
            Dictionary of emotion probabilities
        """
        

        try:
            if not text or not text.strip():
                return {'neutral': 1.0}
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Use transformer model for primary emotion detection
            if self.emotion_classifier:
                try:
                    results = self.emotion_classifier(processed_text)
                    
                    # Convert results to emotion dictionary
                    emotions = {}
                    
                    # Map model labels to our emotion categories
                    label_mapping = {
                        'ANGER': 'anger',
                        'DISGUST': 'disgust',
                        'FEAR': 'fear',
                        'JOY': 'joy',
                        'NEUTRAL': 'neutral',
                        'SADNESS': 'sadness',
                        'SURPRISE': 'surprise',
                        # Additional mappings for different model outputs
                        'anger': 'anger',
                        'disgust': 'disgust',
                        'fear': 'fear',
                        'joy': 'joy',
                        'neutral': 'neutral',
                        'sadness': 'sadness',
                        'surprise': 'surprise',
                        'happy': 'joy',
                        'love': 'joy'
                    }
                    
                    for result in results:
                        label = result['label']
                        score = result['score']
                        
                        # Map to our emotion categories
                        mapped_emotion = label_mapping.get(label.lower())
                        if mapped_emotion is None:
                            continue
                        if mapped_emotion in emotions:
                            emotions[mapped_emotion] += score
                        else:
                            emotions[mapped_emotion] = score
                    
                    # Ensure all emotion categories are present
                    for emotion in self.emotion_labels:
                        if emotion not in emotions:
                            emotions[emotion] = 0.0
                    
                    # Normalize probabilities
                    total = sum(emotions.values())
                    if total > 0:
                        emotions = {k: v/total for k, v in emotions.items()}
                    
                except Exception as e:
                    self.logger.warning(f"Transformer model inference failed: {str(e)}")
                    emotions = self._fallback_emotion_analysis(text)
            else:
                emotions = self._fallback_emotion_analysis(text)
            
            # Enhance with feature-based analysis if requested
            if include_features:
                features = self.extract_text_features(text)
                feature_emotions = self._feature_based_emotion_analysis(features)
                
                # Combine transformer and feature-based results (weighted)
                transformer_weight = 0.8
                feature_weight = 0.2
                
                combined_emotions = {}
                for emotion in self.emotion_labels:
                    transformer_score = emotions.get(emotion, 0.0)
                    feature_score = feature_emotions.get(emotion, 0.0)
                    combined_emotions[emotion] = (
                        transformer_weight * transformer_score + 
                        feature_weight * feature_score
                    )
                
                # Normalize
                total = sum(combined_emotions.values())
                if total > 0:
                    combined_emotions = {k: v/total for k, v in combined_emotions.items()}
                
                emotions = combined_emotions
            
            # Log analysis results
            self.logger.info(f"Text emotion analysis for '{text[:50]}...': {emotions}")
            
            return emotions
            
        except Exception as e:
            self.logger.error(f"Text emotion analysis error: {str(e)}")
            return {'neutral': 1.0}
    
    def _fallback_emotion_analysis(self, text: str) -> Dict:
        """
        Fallback emotion analysis using rule-based approach
        
        Args:
            text: Input text
            
        Returns:
            Emotion probabilities
        """
        emotions = {emotion: 0.0 for emotion in self.emotion_labels}
        emotions['neutral'] = 0.02  # Default baseline
        
        text_lower = text.lower()
        
        # Define emotion keywords
        emotion_keywords ={
    'joy': [
        # Core
        'happy', 'joy', 'excited', 'love', 'wonderful', 'amazing', 'great', 'fantastic', 'excellent', 'cheerful',
        
        # Synonyms & variations
        'delighted', 'elated', 'ecstatic', 'thrilled', 'blissful', 'pleased', 'content', 'satisfied',
        'grateful', 'glad', 'overjoyed', 'euphoric', 'radiant', 'optimistic', 'hopeful', 'joyful',
        'vibrant', 'smiling', 'laughing', 'celebrating', 'fulfilled',
        
        # Slang / colloquial
        'lit', 'dope', 'fire', 'goated', 'based', 'pog', 'win', 'awesome', 'yay', 'woah', 'wow', 'slay', 'cool',
        
        # Emoji cues
        '😊', '😄', '😁', '😃', '🤩', '🥰', '😍', '😘', '🎉', '✨', '❤️', '💖', '👍', '👏', '🙌', '💪', '🌟'
    ],

    'sadness': [
        # Core
        'sad', 'depressed', 'unhappy', 'miserable', 'crying', 'tears', 'lonely', 'heartbroken',
        
        # Synonyms & variations
        'melancholy', 'sorrowful', 'down', 'blue', 'dejected', 'gloomy', 'disheartened', 'hopeless',
        'grieving', 'despair', 'broken', 'upset', 'hurt', 'regretful', 'pained', 'downcast',
        'mournful', 'wistful', 'forlorn', 'troubled', 'lost', 'bitter', 'let down',
        
        # Slang / colloquial
        'fml', 'lowkey sad', 'emo', 'ugh', 'down bad', 'cried', 'meh',
        
        # Emoji cues
        '😢', '😭', '😞', '😔', '😟', '😕', '💔', '😩', '😫', '😣', '😖'
    ],

    'anger': [
        # Core
        'angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated', 'frustrated',
        
        # Synonyms & variations
        'enraged', 'livid', 'fuming', 'resentful', 'outraged', 'offended', 'infuriated',
        'bitter', 'provoked', 'agitated', 'vengeful', 'disgusted', 'indignant', 'hostile',
        'heated', 'mad', 'snappy', 'cross', 'grumpy', 'irate', 'peeved',
        
        # Slang / colloquial
        'salty', 'triggered', 'pissed', 'raging', 'mad af', 'wtf', 'damn', 'smh',
        
        # Emoji cues
        '😠', '😡', '🤬', '😤', '👿', '💢', '😾'
    ],

    'fear': [
        # Core
        'scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'frightened',
        
        # Synonyms & variations
        'fearful', 'petrified', 'horrified', 'uneasy', 'tense', 'alarmed', 'shaky', 'apprehensive',
        'distressed', 'startled', 'stressed', 'insecure', 'paranoid', 'intimidated', 'timid',
        'on edge', 'dreadful', 'alarmed', 'restless',
        
        # Slang / colloquial
        'spooked', 'scared af', 'creeped out', 'yikes', 'uh oh', 'wtf',
        
        # Emoji cues
        '😨', '😰', '😱', '😖', '😧', '😬', '🥶', '😳', '😥'
    ],

    'surprise': [
        # Core
        'surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow',
        
        # Synonyms & variations
        'startled', 'stunned', 'speechless', 'bewildered', 'dumbfounded', 'astounded',
        'confused', 'flabbergasted', 'perplexed', 'incredulous', 'startling',
        'unexpectedly', 'woah', 'what', 'omg',
        
        # Slang / colloquial
        'no way', 'holy', 'wtf', 'whoa', 'dang', 'bruh', 'yo',
        
        # Emoji cues
        '😲', '😯', '😮', '😳', '🤯', '🤔', '😶', '🙀'
    ],

    'disgust': [
        # Core
        'disgusting','disgusted' ,'gross', 'revolting', 'awful', 'terrible', 'horrible', 'nasty',
        
        # Synonyms & variations
        'repulsive', 'abhorrent', 'sickening', 'vile', 'foul', 'hideous', 'offensive',
        'detestable', 'loathsome', 'hateful', 'yucky', 'unclean', 'filthy', 'distasteful',
        'displeased', 'repelled', 'repugnant', 'nauseated', 'sick', 'vomit', 'ew',
        
        # Slang / colloquial
        'eww', 'yuck', 'grossed out', 'nasty af', 'cringe', 'trash', 'mid',
        
        # Emoji cues
        '🤢', '🤮', '😷', '😖', '😫', '😣', '👎'
    ]
}

        
        # Score based on keyword presence
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotions[emotion] = score * 0.6
                emotions['neutral'] -= score * 0.3
        
        # Adjust based on punctuation and capitalization
        if '!' in text:
            emotions['surprise'] += 0.1
            emotions['joy'] += 0.1
            emotions['anger'] += 0.1
        
        if '?' in text:
            emotions['surprise'] += 0.05
        
        if text.isupper():
            emotions['anger'] += 0.2
            emotions['surprise'] += 0.1
        
        # Normalize
        total = sum(max(0, v) for v in emotions.values())
        if total > 0:
            emotions = {k: max(0, v)/total for k, v in emotions.items()}
        
        return emotions
    
    def _feature_based_emotion_analysis(self, features: Dict) -> Dict:
            """
            Emotion analysis based on linguistic features

            Args:
                features: Extracted text features
                
            Returns:
                Emotion probabilities based on features
            """
            emotions = {emotion: 0.0 for emotion in self.emotion_labels}
            emotions['neutral'] = 0.02  # Reduced from 0.2 to minimize neutral dominance

            if not features:
                return emotions

            # High exclamation usage = excitement/anger
            if features.get('exclamation_count', 0) > 0:
                emotions['joy'] += 0.3 * features['exclamation_count']
                emotions['anger'] += 0.2 * features['exclamation_count']
                emotions['neutral'] -= 0.1

            # High question usage = confusion/surprise
            if features.get('question_count', 0) > 0:
                emotions['surprise'] += 0.2 * features['question_count']
                emotions['neutral'] -= 0.05

            # High uppercase ratio = anger/excitement
            if features.get('uppercase_ratio', 0) > 0.3:
                emotions['anger'] += 0.3
                emotions['surprise'] += 0.1
                emotions['neutral'] -= 0.2

            # Positive/negative word balance
            pos_count = features.get('positive_word_count', 0)
            neg_count = features.get('negative_word_count', 0)

            if pos_count > neg_count:
                emotions['joy'] += 0.3 * (pos_count - neg_count)
                emotions['neutral'] -= 0.2
            elif neg_count > pos_count:
                emotions['sadness'] += 0.3 * (neg_count - pos_count)
                emotions['anger'] += 0.15 * (neg_count - pos_count)
                emotions['neutral'] -= 0.2

            # Short, abrupt messages = anger
            if features.get('word_count', 0) < 5 and features.get('uppercase_ratio', 0) > 0.2:
                emotions['anger'] += 0.2
                emotions['neutral'] -= 0.1

            # Very long messages = detailed emotional expression
            if features.get('word_count', 0) > 50:
                # Slightly increase all emotions except neutral
                for emotion in emotions:
                    if emotion != 'neutral':
                        emotions[emotion] += 0.02
                emotions['neutral'] -= 0.1

            # First normalize to ensure all values are non-negative
            total = sum(max(0, v) for v in emotions.values())
            if total > 0:
                emotions = {k: max(0, v)/total for k, v in emotions.items()}

            # Suppress neutral if any non-neutral emotion crosses threshold
            tau = 0.22
            non_neutral = {k: v for k, v in emotions.items() if k != 'neutral'}
            if non_neutral and max(non_neutral.values()) >= tau:
                emotions['neutral'] = min(emotions['neutral'], 0.01)

            # Temperature scaling to sharpen non-neutral emotions
            T = 0.8  # <1 sharpens peaks
            non_neutral_keys = [k for k in emotions if k != 'neutral']
            for k in non_neutral_keys:
                emotions[k] = emotions[k] ** (1.0 / T)

            # Final normalization after suppression and scaling
            s = sum(emotions.values())
            if s > 0:
                emotions = {k: v/s for k, v in emotions.items()}

            return emotions



