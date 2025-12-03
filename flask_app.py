from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import time

app = Flask(__name__)

# Load model
try:
    model_data = joblib.load('youtube_asl_model.joblib')
    model = model_data['model']
    le = model_data['label_encoder']
    print(f"✅ Model loaded: {len(le.classes_)} signs")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    le = None

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

# Global variables
sentence = ""
preds = deque(maxlen=10)
last_spoken = None
last_append_time = 0

def preprocess_landmarks(hand_landmarks):
    lm = []
    for l in hand_landmarks.landmark:
        lm.append([l.x, l.y, l.z])
    lm = np.array(lm)
    wrist = lm[0]
    lm = lm - wrist
    distances = np.linalg.norm(lm, axis=1)
    scale = np.max(distances)
    if scale > 0:
        lm = lm / scale
    return lm.flatten()

def generate_frames():
    global sentence, preds, last_spoken, last_append_time
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_skip = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        frame_skip += 1
        
        if frame_skip % 2 == 0 and model:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                
                feats = preprocess_landmarks(hand)
                probs = model.predict_proba([feats])[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]
                pred_label = le.inverse_transform([pred_idx])[0]
                
                if conf >= 0.4:
                    preds.append(pred_label)
                    
                    if len(preds) >= 4:
                        counter = Counter(preds)
                        most_common, count = counter.most_common(1)[0]
                        
                        if most_common is not None and count >= 4:
                            now = time.time()
                            if (last_spoken != most_common) or (now - last_append_time > 1.5):
                                sentence += most_common + " "
                                last_spoken = most_common
                                last_append_time = now
                                preds.clear()
                            
                            cv2.putText(frame, f"Sign: {most_common}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            cv2.putText(frame, f"Conf: {conf:.1%}", (10, 70), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if sentence:
            cv2.putText(frame, f"Text: {sentence[:30]}", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    global sentence
    return jsonify({'sentence': sentence})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence, preds, last_spoken
    sentence = ""
    preds.clear()
    last_spoken = None
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
