#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip list')


# In[3]:


import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def normalize_landmarks(landmarks_63):
    pts = np.array(landmarks_63).reshape(21, 3)
    pts = pts - pts[0] # Center at wrist
    max_val = np.max(np.abs(pts))
    if max_val > 0: 
        pts = pts / max_val # Scale invariant
    return pts.flatten()

def augment_landmarks(pts_flat, angle_deg=15, noise_std=0.015):
    pts = pts_flat.reshape(21, 3).copy()
    # Random rotation (simulates tilting hand)
    theta = np.radians(np.random.uniform(-angle_deg, angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2] @ rot_matrix.T
    # Add noise (simulates shaky hands)
    pts += np.random.normal(0, noise_std, pts.shape)
    return pts.flatten()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ---> YOUR DATA PATH <---
dataset_path = r"D:\Signverse\MY_WEBCAM_DATA" 

data = []
labels = []

print("Processing images and augmenting...")

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if not os.path.isdir(folder_path): continue

    print(f"Processing letter: {folder_name}")

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.extend([lm.x, lm.y, lm.z])

            normalized = normalize_landmarks(raw_landmarks)

            # Save original
            data.append(normalized)
            labels.append(folder_name)

            # Create 8 augmented versions to boost accuracy
            for _ in range(8):
                data.append(augment_landmarks(normalized))
                labels.append(folder_name)

hands.close()

# Save to CSV
df = pd.DataFrame(data)
df['Label'] = labels
df.to_csv('custom_alphabet_data.csv', index=False)

print("=====================================")
print(f"Successfully created {len(data)} total rows.")
print("Saved to 'custom_alphabet_data.csv'")


# In[4]:


import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def normalize_landmarks(landmarks_63):
    pts = np.array(landmarks_63).reshape(21, 3)
    pts = pts - pts[0]
    max_val = np.max(np.abs(pts))
    if max_val > 0: pts = pts / max_val
    return pts.flatten()

def augment_landmarks(pts_flat, angle_deg=15, noise_std=0.015):
    pts = pts_flat.reshape(21, 3).copy()
    theta = np.radians(np.random.uniform(-angle_deg, angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2] @ rot_matrix.T
    pts += np.random.normal(0, noise_std, pts.shape)
    return pts.flatten()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# YOUR TWO DATASET PATHS
dataset_paths = [
    r"D:\Signverse\DATA",              
    r"D:\Signverse\MY_WEBCAM_DATA"     
]

data = []
labels = []
image_paths = []

print("Processing and Augmenting all data...")

for dataset_path in dataset_paths:
    if not os.path.exists(dataset_path): 
        print(f"Warning: Folder not found -> {dataset_path}")
        continue

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path)
                if img is None: continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    raw_landmarks = []
                    for lm in hand_landmarks.landmark:
                        raw_landmarks.extend([lm.x, lm.y, lm.z])

                    normalized = normalize_landmarks(raw_landmarks)

                    # Save Original
                    data.append(normalized)
                    labels.append(folder_name)
                    image_paths.append(image_path)

                    # Save 8 Augmented Versions
                    for _ in range(8):
                        data.append(augment_landmarks(normalized))
                        labels.append(folder_name)
                        image_paths.append(image_path + "_aug")

hands.close()

df = pd.DataFrame(data)
df['Label'] = labels
df['Image_Path'] = image_paths
df.to_csv('master_custom_data.csv', index=False)

print("=====================================")
print(f"Total rows created: {len(data)}")
print("Saved to master_custom_data.csv")
print("=====================================")


# In[2]:


import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Normalization Function
def normalize_landmarks(landmarks_63):
    pts = np.array(landmarks_63).reshape(21, 3)
    pts = pts - pts[0] # Center relative to wrist
    max_val = np.max(np.abs(pts))
    if max_val > 0: 
        pts = pts / max_val # Scale invariant
    return pts.flatten()

# 2. Augmentation Function
def augment_landmarks(pts_flat, angle_deg=15, noise_std=0.015):
    pts = pts_flat.reshape(21, 3).copy()
    # Random 2D rotation
    theta = np.radians(np.random.uniform(-angle_deg, angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2] @ rot_matrix.T
    # Add slight noise
    pts += np.random.normal(0, noise_std, pts.shape)
    return pts.flatten()

# 3. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 4. Define your dataset paths
dataset_paths = [
    r"D:\Signverse\DATA",               # Your friends' images
    r"D:\Signverse\MY_WEBCAM_DATA"      # Your webcam images (A-D + new letters)
]

data = []
labels = []
image_paths = []

print("Processing and Augmenting all data...")

# 5. Main Loop
for dataset_path in dataset_paths:
    if not os.path.exists(dataset_path): 
        print(f"Warning: Folder not found -> {dataset_path}")
        continue

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Read image
                img = cv2.imread(image_path)
                if img is None: continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                # If hand is detected
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    raw_landmarks = []
                    for lm in hand_landmarks.landmark:
                        raw_landmarks.extend([lm.x, lm.y, lm.z])

                    # A. Normalize the landmarks
                    normalized = normalize_landmarks(raw_landmarks)

                    # B. Save the original image data
                    data.append(normalized)
                    labels.append(folder_name)
                    image_paths.append(image_path)

                    # C. Save 8 Augmented (rotated/noisy) versions
                    for _ in range(8):
                        data.append(augment_landmarks(normalized))
                        labels.append(folder_name)
                        image_paths.append(image_path + "_aug")

hands.close()

# 6. Save to CSV
df = pd.DataFrame(data)
df['Label'] = labels
df['Image_Path'] = image_paths
df.to_csv('master_custom_data.csv', index=False)

print("=====================================")
print(f"SUCCESS: Total rows created = {len(data)}")
print("Saved to master_custom_data.csv")
print("=====================================")


# In[1]:


import sys
import importlib.metadata as metadata

print("Python executable:", sys.executable)
print("Python version:", sys.version)

packages = [
    "mediapipe",
    "opencv-contrib-python",
    "opencv-python",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "protobuf",
    "jupyter"
]

print("\nINSTALLED PACKAGES")

for package in packages:
    try:
        print(package, ":", metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, ": NOT INSTALLED")

print("\nMEDIAPIPE TEST")

try:
    import mediapipe as mp
    print("MediaPipe imported successfully")
    print("Solutions available:", hasattr(mp, "solutions"))
except Exception as e:
    print("MediaPipe error:", repr(e))


# In[ ]:


import cv2, numpy as np, pandas as pd, mediapipe as mp, os

PERSON = "classmate1"          # change for each person
cols = list(pd.read_csv('master_custom_data.csv', nrows=0).columns)
feat = [c for c in cols if c not in ('Label', 'Image_Path')]

def normalize_landmarks(l):
    pts = np.array(l).reshape(21, 3); pts = pts - pts[0]
    m = np.max(np.abs(pts)); return (pts / m if m > 0 else pts).flatten()

hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
cap, counts = cv2.VideoCapture(0), {}
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)                       # same as live code
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.putText(frame, str(counts), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('Collect', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
    if 97 <= k <= 122 and res.multi_hand_landmarks:
        L = chr(k).upper(); counts[L] = counts.get(L, 0) + 1
        raw = [v for p in res.multi_hand_landmarks[0].landmark for v in (p.x, p.y, p.z)]
        row = dict(zip(feat, normalize_landmarks(raw)))
        row.update(Label=L, Image_Path=f"{PERSON}_{L}_{counts[L]}")
        pd.DataFrame([row]).to_csv('new_people_data.csv', mode='a', index=False,
                                   header=not os.path.exists('new_people_data.csv'))
cap.release(); cv2.destroyAllWindows()


# In[ ]:


import pandas as pd, numpy as np, joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

old, new = pd.read_csv('master_custom_data.csv'), pd.read_csv('new_people_data.csv')
feat = [c for c in old.columns if c not in ('Label', 'Image_Path')]

def mirror(X):
    M = X.copy().reshape(len(X), 21, 3); M[:, :, 0] *= -1
    return M.reshape(len(X), -1)

Xo, yo = old[feat].values, old['Label'].values
pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale', probability=True))
pipe.fit(np.vstack([Xo, mirror(Xo)]), np.concatenate([yo, yo]))   # mirror = left/right hands

pred = pipe.predict(new[feat].values)
print("Accuracy on NEW people:", accuracy_score(new['Label'], pred))
print(classification_report(new['Label'], pred))

