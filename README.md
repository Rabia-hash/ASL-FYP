# ASL-FYP
# American Sign Language (ASL) Translator - FYP

## 📖 Project Overview
This project is a Final Year Project (FYP) focused on American Sign Language (ASL) recognition and translation. The system captures video input, processes the hand gestures using computer vision, and uses a Long Short-Term Memory (LSTM) neural network to predict and translate the signs in real-time.

## 📂 Project Structure
The project is divided into several Jupyter Notebooks for modularity:
*   `0_Datacollect.ipynb`: Script used to collect and save raw image/keypoint data for the ASL gestures.
*   `1_Data Preprocessing.ipynb`: Cleans, normalizes, and prepares the collected data for training.
*   `LSTM preprocessing.ipynb`: Formats the sequential data specifically for the LSTM model.
*   `LSTM_2_Train.ipynb` & `Model Training.ipynb`: Contains the architecture, training loops, and evaluation of the LSTM model.
*   `Webcam.ipynb`: Loads the trained model and runs real-time inference using the device webcam.

## 🛠️ Environment & Requirements
To run this project locally, you will need **Python 3.12 or 3.11** and **Jupyter Notebook**.

### Required Libraries
You can install the necessary dependencies using pip. (Note: If you have a dedicated `requirements.txt`, you can update this section later).

```bash
pip install numpy pandas matplotlib scikit-learn
pip install tensorflow # or tensorflow-gpu if you have an NVIDIA GPU
pip install opencv-python mediapipe # Assuming MediaPipe is used for hand tracking
pip install jupyter
