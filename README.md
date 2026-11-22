Signverse is a Final Year Project (FYP) focused on bridging the communication gap between the deaf and hearing communities. It is a lightweight, real-time American Sign Language (ASL) recognition system that translates static fingerspelling and dynamic words into English text using a standard webcam.

📖 Project Overview
The system bypasses the need for heavy, pixel-based Convolutional Neural Networks (CNNs) by utilizing MediaPipe to extract 3D hand landmarks. This makes the application exceptionally fast and allows it to run smoothly on standard laptops (CPU) without requiring a dedicated GPU.

The architecture is split into two distinct pipelines:

Static Alphabet (A-Z): Uses a Support Vector Machine (SVM) to recognize stationary hand shapes.

Dynamic Words: Uses a Long Short-Term Memory (LSTM) neural network to recognize the temporal motion of 50 common ASL words.

✨ Key Features
Real-Time Inference: Operates at 15+ FPS on a standard webcam.

Scale & Translation Invariance: Utilizes wrist-centered coordinate normalization so the system works regardless of hand size or distance from the camera.

Data Augmentation: Mathematically rotates and jitters landmark data during training to simulate different hand angles.

Temporal Smoothing: Implements a sliding window buffer and confidence thresholding to prevent "flickering" predictions.

Dual-Mode Operation: Dynamically switches between static alphabet recognition and dynamic word recognition.

🏗️ System Architecture
text
[ Webcam Input ]
       │
       ▼
[ OpenCV Image Processing ]
       │
       ▼
[ MediaPipe Hands ]
 (Extracts 21 landmarks / 63 coordinates)
       │
       ▼
[ Coordinate Normalization ]
 (Wrist-centered, Scale-invariant)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ Static Pipeline ]                       [ Dynamic Pipeline ]
 30-frame buffer is static                 30-frame buffer shows motion
       │                                         │
       ▼                                         ▼
[ SVM Classifier ]                        [ LSTM Neural Network ]
 (Predicts A-Z)                           (Predicts 50 common words)
       │                                         │
       └───────────────────┬─────────────────────┘
                           ▼
                [ English Text Output ]
📂 Project Structure
The project is modularized into Jupyter Notebooks to separate data preparation, training, and deployment:

text
Signverse/
│
├── datasets/                  # Excluded from Git (Too large)
│   ├── raw_videos/            # Original WLASL/ASL videos
│   └── landmarks/             # Extracted .npy files
│
├── models/                    # Excluded from Git
│   ├── svm_alphabet.pkl       # Trained SVM model
│   └── lstm_words.h5          # Trained LSTM model
│
├── notebooks/
│   ├── 0_Dataset_Acquisition.ipynb      # Script to download/curate video dataset
│   ├── 1_Data_Preprocessing.ipynb       # Extracts & normalizes MediaPipe landmarks
│   ├── 2_SVM_Alphabet_Training.ipynb    # Trains the static alphabet model
│   ├── 3_LSTM_Preprocessing.ipynb       # Formats sequences for dynamic words
│   ├── 4_LSTM_Training.ipynb            # Trains the TensorFlow LSTM model
│   └── 5_Live_Webcam_App.ipynb          # Final real-time deployment app
│
├── requirements.txt           # Pinned dependencies for all teammates
└── README.md
🛠️ Installation & Environment Setup
⚠️ CRITICAL ENVIRONMENT WARNING:
TensorFlow (for LSTM) and MediaPipe (for landmark extraction) often conflict over the protobuf library. To prevent dependency crashes, Python 3.11 is highly recommended. You must install the pinned versions listed below.

1. Prerequisites
Python 3.11 installed on your system.

A working webcam.

2. Step-by-Step Installation
Clone the repository and install the dependencies:

bash
git clone https://github.com/your-username/Signverse.git
cd Signverse
Install the specific versions to avoid the protobuf conflict:

bash
pip install numpy==1.26.4 pandas matplotlib scikit-learn joblib
pip install opencv-python mediapipe==0.10.14
pip install tensorflow==2.15.0 protobuf==3.20.3
pip install jupyter ipykernel
(Alternatively, if a requirements.txt is provided, simply run pip install -r requirements.txt)

🚀 Usage Guide
Run the notebooks in sequential order. Ensure your Jupyter Kernel is set to Python 3.11.

Data Preparation:

Run 0_Dataset_Acquisition.ipynb to download and filter the dataset to the target 50 words.

Run 1_Data_Preprocessing.ipynb to extract the MediaPipe landmarks and save them as .npy files.

Static Alphabet Training:

Run 2_SVM_Alphabet_Training.ipynb. This trains the model and saves svm_alphabet.pkl.

Dynamic Word Training:

Run 3_LSTM_Preprocessing.ipynb to organize the landmarks into 30-frame sequences.

Run 4_LSTM_Training.ipynb to train the neural network and save lstm_words.h5.

Real-Time Deployment:

Run 5_Live_Webcam_App.ipynb. The webcam will open, and the system will begin predicting signs in real-time.

📊 Dataset Information
To ensure the model generalizes well to different signers, backgrounds, and lighting conditions, this project utilizes a curated subset of the WLASL (Word-Level American Sign Language) dataset.

Vocabulary: 50 common ASL phrases (e.g., HELLO, THANK YOU, HELP, WATER).

Signers: Multiple independent signers to prevent overfitting.

Data Leakage Prevention: Train/Test splits are done by signer rather than random frames to ensure true accuracy.

🚧 Limitations & Future Work
Current Limitations (Scope of Bachelor's FYP):

Isolated Signs: The system recognizes isolated words, not continuous, grammatical ASL sentences.

Hands Only: The current MediaPipe pipeline only extracts hand landmarks. It does not process facial expressions or body posture (which are vital for full ASL grammar).

Vocabulary Size: Limited to 50 words (with a stretch goal of 100).

Future Work:

Implement MediaPipe Holistic to include facial and pose landmarks.

Transition from LSTM to Transformers for better sequence modeling.

Expand vocabulary to 500+ words using larger datasets.

Develop a mobile app wrapper using TensorFlow Lite.
