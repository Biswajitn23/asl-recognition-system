import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import time
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="👋",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sentence-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stats-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        data = joblib.load("youtube_asl_model.joblib")
        return data["model"], data["le"]
    except:
        st.error("Model not found! Please train the model first.")
        return None, None

@st.cache_resource
def init_mediapipe():
    """Initialize MediaPipe hands"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=0
    )
    return hands, mp_hands

def preprocess_landmarks(hand_landmarks):
    """Extract and normalize hand landmarks"""
    lm = []
    for l in hand_landmarks.landmark:
        lm.append([l.x, l.y, l.z])
    lm = np.array(lm)
    wrist = lm[0].copy()
    lm = lm - wrist
    max_val = np.max(np.abs(lm))
    if max_val > 0:
        lm = lm / max_val
    return lm.flatten()

def main():
    # Header
    st.markdown('<p class="main-header">👋 ASL Sign Language Recognition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time American Sign Language Detection using AI</p>', unsafe_allow_html=True)
    
    # Load model
    model, le = load_model()
    if model is None:
        st.stop()
    
    # Initialize MediaPipe
    hands, mp_hands = init_mediapipe()
    mp_draw = mp.solutions.drawing_utils
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        smoothing_window = st.slider("Smoothing Window", 5, 20, 15)
        stable_count = st.slider("Stable Count", 3, 15, 6)
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.info(f"**Trained Signs:** {len(le.classes_)}")
        for sign in le.classes_:
            st.write(f"• {sign}")
        
        st.markdown("---")
        st.header("📖 Instructions")
        st.write("""
        1. Allow camera access
        2. Show ASL signs to camera
        3. System will recognize signs
        4. Build sentences with gestures
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        run = st.checkbox("Start Camera", value=True)
        FRAME_WINDOW = st.image([])
        
    with col2:
        st.subheader("🎯 Prediction")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        st.subheader("💬 Sentence")
        sentence_placeholder = st.empty()
        
        col_clear, col_copy = st.columns(2)
        with col_clear:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        with col_copy:
            copy_button = st.button("📋 Copy", use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Stats")
        stats_placeholder = st.empty()
    
    # Initialize session state
    if 'sentence' not in st.session_state:
        st.session_state.sentence = ""
    if 'preds' not in st.session_state:
        st.session_state.preds = deque(maxlen=smoothing_window)
    if 'last_spoken' not in st.session_state:
        st.session_state.last_spoken = None
    if 'last_append_time' not in st.session_state:
        st.session_state.last_append_time = 0
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Clear button logic
    if clear_button:
        st.session_state.sentence = ""
        st.session_state.last_spoken = None
        st.session_state.preds.clear()
    
    # Camera feed
    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera!")
            st.stop()
        
        frame_skip = 0
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            st.session_state.frame_count += 1
            frame = cv2.flip(frame, 1)
            
            # Process every 3rd frame for better performance
            frame_skip += 1
            if frame_skip % 3 != 0:
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=False)
                time.sleep(0.01)  # Small delay to reduce CPU usage
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            current_pred = None
            current_conf = 0.0
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                
                feats = preprocess_landmarks(hand)
                probs = model.predict_proba([feats])[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]
                pred_label = le.inverse_transform([pred_idx])[0]
                
                if conf >= confidence_thresh:
                    st.session_state.preds.append(pred_label)
                    current_pred = pred_label
                    current_conf = conf
                else:
                    st.session_state.preds.append(None)
                
                # Smoothing
                if len(st.session_state.preds) > 0:
                    most_common, count = Counter(st.session_state.preds).most_common(1)[0]
                    if most_common is not None and count >= stable_count:
                        now = time.time()
                        if (st.session_state.last_spoken != most_common) or \
                           (now - st.session_state.last_append_time > 1.2):
                            st.session_state.sentence += most_common + " "
                            st.session_state.last_spoken = most_common
                            st.session_state.last_append_time = now
                        
                        current_pred = most_common
                        current_conf = conf
                        
                        # Draw on frame
                        cv2.putText(frame, f"Sign: {most_common}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.putText(frame, f"Confidence: {current_conf:.1%}", (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                st.session_state.preds.append(None)
                cv2.putText(frame, "No hand detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display sentence on frame
            if st.session_state.sentence:
                cv2.putText(frame, f"Sentence: {st.session_state.sentence[:30]}", (10, frame.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Update display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame, channels="RGB", use_container_width=False)
            
            # Update predictions
            if current_pred:
                prediction_placeholder.markdown(
                    f'<div class="prediction-box">🤟 {current_pred.upper()}</div>', 
                    unsafe_allow_html=True
                )
                confidence_placeholder.progress(current_conf)
            else:
                prediction_placeholder.markdown(
                    '<div class="prediction-box">✋ Waiting...</div>', 
                    unsafe_allow_html=True
                )
                confidence_placeholder.progress(0.0)
            
            # Update sentence
            sentence_placeholder.markdown(
                f'<div class="sentence-box">{st.session_state.sentence or "Start signing..."}</div>',
                unsafe_allow_html=True
            )
            
            # Update stats
            stats_placeholder.markdown(f"""
            <div class="stats-box">
            <b>📊 Statistics</b><br>
            Frames Processed: {st.session_state.frame_count}<br>
            Current Buffer: {len(st.session_state.preds)}/{smoothing_window}<br>
            Words Built: {len(st.session_state.sentence.split())}<br>
            </div>
            """, unsafe_allow_html=True)
            
            # Small delay
            time.sleep(0.03)
        
        cap.release()
    else:
        st.info("👆 Check 'Start Camera' to begin recognition")

if __name__ == "__main__":
    main()