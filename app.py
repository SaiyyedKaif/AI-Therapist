import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading
import queue
import time
import base64
from io import BytesIO
from PIL import Image

# Import our custom modules
from src.face_emotion import FaceEmotionDetector
from src.voice_emotion import VoiceEmotionDetector
from src.text_emotion import TextEmotionDetector
from src.fusion_engine import EmotionFusionEngine
from src.intervention_system import InterventionSystem
from src.emotion_tracker import EmotionTracker
from utils.video_utils import VideoProcessor
from utils.audio_utils import AudioProcessor

# Configure page
st.set_page_config(
    page_title="AI Therapist - Multi-Modal Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = None
if 'face_detector' not in st.session_state:
    st.session_state.face_detector = FaceEmotionDetector()
if 'voice_detector' not in st.session_state:
    st.session_state.voice_detector = VoiceEmotionDetector()
if 'text_detector' not in st.session_state:
    st.session_state.text_detector = TextEmotionDetector()
if 'fusion_engine' not in st.session_state:
    st.session_state.fusion_engine = EmotionFusionEngine()
if 'intervention_system' not in st.session_state:
    st.session_state.intervention_system = InterventionSystem()
if 'emotion_tracker' not in st.session_state:
    st.session_state.emotion_tracker = EmotionTracker()
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'audio_active' not in st.session_state:
    st.session_state.audio_active = False

def main():
    # Header
    st.title("🧠 AI Therapist - Multi-Modal Emotion Detection")
    st.markdown("**Real-time wellness support through facial, voice, and text emotion analysis**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Modality weights
        st.subheader("Emotion Detection Weights")
        face_weight = st.slider("Facial Expression Weight", 0.0, 1.0, 0.4, 0.1)
        voice_weight = st.slider("Voice Tone Weight", 0.0, 1.0, 0.3, 0.1)
        text_weight = st.slider("Text Analysis Weight", 0.0, 1.0, 0.3, 0.1)
        
        # Update fusion engine weights
        st.session_state.fusion_engine.set_weights({
            'face': face_weight,
            'voice': voice_weight,
            'text': text_weight
        })
        
        st.divider()
        
        # Session controls
        st.subheader("Session Controls")
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.emotion_history = []
            st.session_state.current_emotion = None
            st.session_state.emotion_tracker.reset()
            st.rerun()
        
        if st.button("📊 Export Data", use_container_width=True):
            if st.session_state.emotion_history:
                df = pd.DataFrame(st.session_state.emotion_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"emotion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create tabs for different input modes
        tab1, tab2, tab3 = st.tabs(["📹 Camera", "🎤 Voice", "✍️ Text"])
        
        with tab1:
            st.subheader("Facial Emotion Detection")
            
            # Camera controls
            camera_col1, camera_col2 = st.columns(2)
            with camera_col1:
                start_camera = st.button("📸 Start Camera", use_container_width=True)
            with camera_col2:
                stop_camera = st.button("⏹️ Stop Camera", use_container_width=True)
            
            if start_camera:
                st.session_state.camera_active = True
            if stop_camera:
                st.session_state.camera_active = False
            
            # Camera feed placeholder
            camera_placeholder = st.empty()
            
            if st.session_state.camera_active:
                try:
                    # Initialize camera
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Process frame for emotion detection
                            face_emotions = st.session_state.face_detector.detect_emotions(frame)
                            
                            # Draw emotion detection results on frame
                            if face_emotions:
                                frame = st.session_state.face_detector.draw_emotions(frame, face_emotions)
                            
                            # Convert frame to RGB and display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                            
                            # Update emotion data
                            if face_emotions:
                                dominant_emotion = max(face_emotions[0]['emotions'].items(), key=lambda x: x[1])
                                update_emotion_data('face', {dominant_emotion[0]: dominant_emotion[1]})
                    
                    cap.release()
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                    st.session_state.camera_active = False
            else:
                camera_placeholder.info("📸 Click 'Start Camera' to begin facial emotion detection")
        
        with tab2:
            st.subheader("Voice Emotion Analysis")
            
            # Voice controls
            voice_col1, voice_col2 = st.columns(2)
            with voice_col1:
                start_recording = st.button("🎙️ Start Recording", use_container_width=True)
            with voice_col2:
                stop_recording = st.button("⏹️ Stop Recording", use_container_width=True)
            
            if start_recording:
                st.session_state.audio_active = True
                st.info("🎙️ Recording... Speak naturally for emotion analysis")
            
            if stop_recording:
                st.session_state.audio_active = False
                st.success("✅ Recording stopped. Analyzing audio...")
                
                # Simulate audio processing (in real implementation, this would process actual audio)
                try:
                    voice_emotions = st.session_state.voice_detector.analyze_audio_chunk()
                    if voice_emotions:
                        update_emotion_data('voice', voice_emotions)
                        st.success("🎵 Voice emotion analysis complete")
                except Exception as e:
                    st.error(f"Voice analysis error: {str(e)}")
            
            # Audio visualization placeholder
            if st.session_state.audio_active:
                st.info("🎵 Audio visualization would appear here in real-time")
            else:
                st.info("🎤 Click 'Start Recording' to begin voice emotion analysis")
        
        with tab3:
            st.subheader("Text Emotion Analysis")
            
            # Text input
            user_text = st.text_area(
                "Share your thoughts or feelings:",
                placeholder="Type here to analyze the emotional content of your text...",
                height=150
            )
            
            analyze_text_btn = st.button("🔍 Analyze Text", use_container_width=True)
            
            if analyze_text_btn and user_text.strip():
                try:
                    with st.spinner("Analyzing text emotions..."):
                        text_emotions = st.session_state.text_detector.analyze_text(user_text)
                        if text_emotions:
                            update_emotion_data('text', text_emotions)
                            
                            # Display text analysis results
                            st.success("✅ Text analysis complete")
                            
                            # Show emotion breakdown
                            emotion_df = pd.DataFrame(list(text_emotions.items()), 
                                                    columns=['Emotion', 'Confidence'])
                            emotion_df = emotion_df.sort_values('Confidence', ascending=False)
                            
                            fig = px.bar(emotion_df, x='Emotion', y='Confidence',
                                       title="Text Emotion Analysis Results")
                            st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Text analysis error: {str(e)}")
            
            elif analyze_text_btn:
                st.warning("⚠️ Please enter some text to analyze")
    
    with col2:
        # Current emotion display
        st.subheader("🎯 Current Emotional State")
        
        if st.session_state.current_emotion:
            emotion_data = st.session_state.current_emotion
            
            # Display dominant emotion
            dominant_emotion = max(emotion_data['emotions'].items(), key=lambda x: x[1])
            
            # Emotion indicator
            emotion_colors = {
                'happy': '#4CAF50', 'sad': '#2196F3', 'angry': '#F44336',
                'fear': '#9C27B0', 'surprise': '#FF9800', 'disgust': '#795548',
                'neutral': '#607D8B', 'joy': '#FFEB3B', 'love': '#E91E63'
            }
            
            color = emotion_colors.get(dominant_emotion[0], '#607D8B')
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h3 style="color: {color}; margin: 0;">{dominant_emotion[0].title()}</h3>
                <p style="margin: 5px 0; font-size: 18px;">Confidence: {dominant_emotion[1]:.1%}</p>
                <p style="margin: 0; font-size: 14px;">Detected: {emotion_data['timestamp'].strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Modality contributions
            st.write("**Modality Contributions:**")
            for modality, emotions in emotion_data.get('modalities', {}).items():
                if emotions:
                    top_emotion = max(emotions.items(), key=lambda x: x[1])
                    st.write(f"• {modality.title()}: {top_emotion[0]} ({top_emotion[1]:.1%})")
        
        else:
            st.info("🤖 No emotion detected yet. Try using camera, voice, or text input.")
        
        st.divider()
        
        # Intervention recommendations
        st.subheader("💡 Wellness Interventions")
        
        if st.session_state.current_emotion:
            interventions = st.session_state.intervention_system.get_interventions(
                st.session_state.current_emotion['emotions']
            )
            
            if interventions:
                # Quote
                if 'quote' in interventions:
                    st.markdown("**🌟 Inspirational Quote:**")
                    st.markdown(f"*\"{interventions['quote']['text']}\"*")
                    st.markdown(f"— {interventions['quote']['author']}")
                
                # Music recommendation
                if 'music' in interventions:
                    st.markdown("**🎵 Music Recommendation:**")
                    music = interventions['music']
                    st.write(f"• **Genre:** {music['genre']}")
                    st.write(f"• **Mood:** {music['mood']}")
                    st.write(f"• **Suggestion:** {music['description']}")
                
                # Wellness tip
                if 'wellness_tip' in interventions:
                    st.markdown("**🧘 Wellness Tip:**")
                    st.info(interventions['wellness_tip'])
                
                # Breathing exercise
                if 'breathing_exercise' in interventions:
                    st.markdown("**🫁 Breathing Exercise:**")
                    exercise = interventions['breathing_exercise']
                    st.write(f"**{exercise['name']}**")
                    st.write(exercise['instructions'])
                    
                    if st.button("🧘 Start Guided Breathing", use_container_width=True):
                        breathing_exercise_timer(exercise)
        
        else:
            st.info("💡 Personalized interventions will appear based on your detected emotions.")
    
    # Emotion history visualization
    st.divider()
    st.subheader("📈 Emotion History & Patterns")
    
    if st.session_state.emotion_history:
        # Create dataframe from history
        history_df = pd.DataFrame(st.session_state.emotion_history)
        
        # Time series plot
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotion timeline
            fig = px.line(history_df, x='timestamp', y='confidence', color='emotion',
                         title="Emotion Confidence Over Time")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Emotion distribution
            emotion_counts = history_df['emotion'].value_counts()
            fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                        title="Emotion Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent emotions table
        st.subheader("Recent Emotion Detections")
        recent_emotions = history_df.tail(10)[['timestamp', 'emotion', 'confidence', 'modality']]
        recent_emotions['timestamp'] = recent_emotions['timestamp'].dt.strftime('%H:%M:%S')
        recent_emotions['confidence'] = recent_emotions['confidence'].apply(lambda x: f"{x:.1%}")
        st.dataframe(recent_emotions, use_container_width=True, hide_index=True)
    
    else:
        st.info("📊 Emotion history will be displayed here as you use the system.")

def update_emotion_data(modality, emotions):
    """Update emotion data from different modalities"""
    try:
        # Store individual modality data
        timestamp = datetime.now()
        
        # Add to history for each detected emotion
        for emotion, confidence in emotions.items():
            st.session_state.emotion_history.append({
                'timestamp': timestamp,
                'emotion': emotion,
                'confidence': confidence,
                'modality': modality
            })
        
        # Fuse emotions if we have multiple modalities
        current_emotions = {modality: emotions}
        
        # Get recent emotions from other modalities (within last 5 seconds)
        recent_time = timestamp - timedelta(seconds=5)
        
        # Collect recent emotions from other modalities
        for record in reversed(st.session_state.emotion_history[-20:]):  # Check last 20 records
            if record['timestamp'] >= recent_time and record['modality'] != modality:
                mod = record['modality']
                if mod not in current_emotions:
                    current_emotions[mod] = {}
                current_emotions[mod][record['emotion']] = record['confidence']
        
        # Fuse the emotions
        fused_emotions = st.session_state.fusion_engine.fuse_emotions(current_emotions)
        
        # Update current emotion state
        st.session_state.current_emotion = {
            'emotions': fused_emotions,
            'modalities': current_emotions,
            'timestamp': timestamp
        }
        
        # Update emotion tracker
        st.session_state.emotion_tracker.add_emotion(fused_emotions, timestamp)
        
    except Exception as e:
        st.error(f"Error updating emotion data: {str(e)}")

def breathing_exercise_timer(exercise):
    """Display a guided breathing exercise"""
    duration = exercise.get('duration', 60)  # Default 1 minute
    
    placeholder = st.empty()
    
    for i in range(duration, 0, -1):
        if i > duration // 2:
            phase = "Breathe In"
            color = "#4CAF50"
        else:
            phase = "Breathe Out"
            color = "#2196F3"
        
        placeholder.markdown(f"""
        <div style="background-color: {color}20; padding: 40px; border-radius: 15px; text-align: center;">
            <h2 style="color: {color}; margin: 0;">{phase}</h2>
            <h1 style="color: {color}; margin: 10px 0;">{i}</h1>
            <p>Seconds remaining</p>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
    
    placeholder.success("🎉 Breathing exercise complete! You should feel more relaxed now.")

if __name__ == "__main__":
    main()
