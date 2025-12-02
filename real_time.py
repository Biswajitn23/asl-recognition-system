# real_time.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import pyttsx3
import time

MODEL_PATH = "youtube_asl_model.joblib"  # Updated to use YouTube-trained model
CONFIDENCE_THRESH = 0.5
SMOOTHING_WINDOW = 12   # number of recent predictions to consider
STABLE_COUNT = 9        # if a label appears at least this many times in the window, we treat it as stable

# load
data = joblib.load(MODEL_PATH)
model = data["model"]
le = data["le"]

# TTS init
engine = pyttsx3.init()
engine.setProperty('rate', 160)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
preds = deque(maxlen=SMOOTHING_WINDOW)
last_spoken = None
sentence = ""
last_append_time = 0
APPEND_COOLDOWN = 0.9  # seconds before appending same char again

def preprocess_landmarks(hand_landmarks):
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

print("Starting realtime. Press 'q' to quit, 'c' to clear sentence.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        feats = preprocess_landmarks(hand)
        probs = model.predict_proba([feats])[0]
        pred_idx = np.argmax(probs)
        conf = probs[pred_idx]
        pred_label = le.inverse_transform([pred_idx])[0]

        if conf >= CONFIDENCE_THRESH:
            preds.append(pred_label)
        else:
            preds.append(None)

        # smoothing: check most common in deque
        most_common, count = Counter(preds).most_common(1)[0]
        if most_common is not None and count >= STABLE_COUNT:
            # stable prediction
            now = time.time()
            # append to sentence if cooldown passed
            if (last_spoken != most_common) or (now - last_append_time > APPEND_COOLDOWN):
                sentence += most_common
                last_spoken = most_common
                last_append_time = now
                # speak
                engine.say(most_common)
                engine.startLoop(False)  # non-blocking
                engine.iterate()
                engine.endLoop()
        # display current prediction and confidence
        display = f"Pred: {most_common}  conf:{conf:.2f}"
        cv2.putText(frame, display, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    else:
        preds.append(None)
        cv2.putText(frame, "No hand", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # show assembled sentence
    cv2.putText(frame, "Output: " + sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.imshow("Sign2Text", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence = ""
        last_spoken = None

cap.release()
cv2.destroyAllWindows()
