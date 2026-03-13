# Real-Time Facial Emotion Detection using AI

## Overview
This project is a **Real-Time Facial Emotion Detection System** built using **Python, OpenCV, and Deep Learning**.  
The system detects a human face through a webcam and predicts the emotion in real time using a **Convolutional Neural Network (CNN)** model.

It also displays the **confidence percentage** and a **real-time emotion probability graph**.

---

## Features
- Real-time face detection using OpenCV
- Emotion recognition using Deep Learning (CNN)
- Emotion confidence percentage display
- Real-time emotion probability graph
- Works with laptop webcam
- Detects multiple emotions

---

## Emotions Detected
The system can detect the following emotions:

- Happy
- Sad
- Angry
- Surprise
- Fear
- Disgust
- Neutral

---

## Technologies Used
- Python
- OpenCV
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Project Structure

```
Emotion-Detection-AI
│
├── emotion_detector.py
├── emotion_model.hdf5
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

---

## Installation

### 1 Clone the Repository

```
git clone https://github.com/yourusername/Emotion-Detection-AI.git
```

### 2 Navigate to Project Folder

```
cd Emotion-Detection-AI
```

### 3 Install Required Libraries

```
pip install -r requirements.txt
```

### 4 Run the Program

```
python emotion_detector.py
```

---

## How It Works

1. The webcam captures video frames.
2. OpenCV detects the human face using Haar Cascade.
3. The detected face is preprocessed and passed to the CNN model.
4. The model predicts the emotion.
5. The predicted emotion and confidence percentage are displayed on the screen.
6. A real-time graph visualizes the emotion probabilities.

---

## Example Output

- Face detection with bounding box
- Emotion label with confidence percentage
- Real-time emotion probability graph

---

## Applications

- Human Computer Interaction
- Mental Health Monitoring
- Customer Experience Analysis
- Smart Surveillance Systems
- AI Assistants

---

## Future Improvements

- Improve model accuracy with larger datasets
- Add GUI interface
- Deploy as a web application
- Support multiple face tracking

---


## License

This project is open-source and available under the MIT License.
