#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")
model = joblib.load('svm_alphabet_model.pkl')

def normalize_landmarks(landmarks_63):
    pts = np.array(landmarks_63).reshape(21, 3)
    pts = pts - pts[0]
    max_val = np.max(np.abs(pts))
    if max_val > 0: pts = pts / max_val
    return pts.flatten()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Buffer for temporal smoothing
prediction_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.extend([lm.x, lm.y, lm.z])
            normalized = normalize_landmarks(raw_landmarks)

            # Predict
            landmarks_array = np.array(normalized).reshape(1, -1)
            prediction = model.predict(landmarks_array)[0]
            probabilities = model.predict_proba(landmarks_array)[0]
            confidence = max(probabilities) * 100

            # Confidence Threshold (only trust if > 70%)
            if confidence > 70:
                prediction_buffer.append(prediction)
            else:
                prediction_buffer.append("...")

            # Get most common prediction in the last 10 frames
            if len(prediction_buffer) > 0:
                most_common = Counter(prediction_buffer).most_common(1)[0][0]
            else:
                most_common = "..."

            # Draw UI
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            text_color = (0, 255, 0) if most_common != "..." else (0, 255, 255)
            cv2.putText(frame, f"Letter: {most_common}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    else:
        prediction_buffer.clear()
        cv2.putText(frame, "No Hand Detected", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Live Alphabet Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()


# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import warnings
warnings.filterwarnings('ignore')

# 1. Load the trained model
print("Loading model...")
# Ensure this matches the model file you saved in your training notebook
model = joblib.load('svm_alphabet_model.pkl') 

# 2. Normalization function (Must match training exactly)
def normalize_landmarks(landmarks_63):
    pts = np.array(landmarks_63).reshape(21, 3)
    pts = pts - pts[0] # Center at wrist
    max_val = np.max(np.abs(pts))
    if max_val > 0: 
        pts = pts / max_val # Scale invariant
    return pts.flatten()

# 3. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 4. Buffer for temporal smoothing (Stops flickering)
prediction_buffer = deque(maxlen=5)

# 5. Start Webcam
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.extend([lm.x, lm.y, lm.z])

            # Normalize
            normalized = normalize_landmarks(raw_landmarks)

            # Predict
            landmarks_array = np.array(normalized).reshape(1, -1)
            prediction = model.predict(landmarks_array)[0]
            probabilities = model.predict_proba(landmarks_array)[0]
            confidence = max(probabilities) * 100

            # Confidence Threshold (40%)
            if confidence > 40:
                prediction_buffer.append(prediction)
            else:
                prediction_buffer.append("...")

            # Smoothing Logic (Get most common in last 5 frames)
            if len(prediction_buffer) > 0:
                most_common = Counter(prediction_buffer).most_common(1)[0][0]
            else:
                most_common = "..."

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine text color (Green for confident, Yellow for unsure)
            text_color = (0, 255, 0) if most_common != "..." else (0, 255, 255)

            # Draw Text
            cv2.putText(frame, f"Letter: {most_common}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    else:
        # If no hand is detected
        prediction_buffer.clear()
        cv2.putText(frame, "No Hand Detected", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show Window
    cv2.imshow('Live Alphabet Test', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Webcam closed.")


# In[ ]:




