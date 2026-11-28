#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import mediapipe as mp
import os

# --- CHANGE THIS FOR EACH LETTER ---
CURRENT_LETTER = 'B' 
SAVE_FOLDER = r"D:\Signverse\MY_WEBCAM_DATA"

save_path = os.path.join(SAVE_FOLDER, CURRENT_LETTER)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = len(os.listdir(save_path)) 
print(f"Saving to: {save_path}")
print("Press 's' to SAVE (only works if hand is detected). Press 'q' to QUIT.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the screen so you can see them
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show text on screen
    color = (0, 255, 0) if hand_detected else (0, 0, 255) # Green if hand, Red if no hand
    status = "HAND DETECTED" if hand_detected else "NO HAND"
    cv2.putText(frame, f"{status} | Letter: {CURRENT_LETTER} | Saved: {count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Webcam Capture", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        if hand_detected:
            img_name = f"{count}.jpg"
            cv2.imwrite(os.path.join(save_path, img_name), frame)
            print(f"Saved {img_name}")
            count += 1
        else:
            print("Cannot save! No hand detected in frame.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()


# In[ ]:




