# Gesture-Based System Control and Sign Language to Speech Conversion

A real-time AI-powered assistive system that combines touchless computer control with sign language to speech conversion using only a standard webcam.

Developed as a research-oriented project at MIT World Peace University, this system leverages MediaPipe hand tracking, Machine Learning, and Computer Vision to create an accessible human-computer interaction framework for users with hearing or motor impairments.

---

## Overview

Traditional computer interaction devices such as keyboards and mice create accessibility barriers for users with physical disabilities, while sign language communication remains inaccessible to most non-signers.

This project solves both problems using a unified real-time pipeline capable of:

- Hand gesture based cursor control
- Sign language recognition
- Real-time speech synthesis
- Touchless interaction
- Webcam-only operation without specialized hardware

The system uses MediaPipe Hands for 21-point landmark extraction and a K-Nearest Neighbors (KNN) classifier for gesture recognition.

---

## Key Features

### Gesture-Based System Control
- Cursor movement using hand tracking
- Left click detection
- Right click detection
- Drag and hold functionality
- Smooth cursor motion using EMA filtering

### Sign Language to Speech
- Real-time sign recognition
- Sentence buffer generation
- Offline text-to-speech synthesis using pyttsx3
- Multi-token sentence accumulation

### Real-Time Performance
- 21–35 FPS real-time execution
- Mean inference latency of 28 ms
- Multi-threaded architecture for smooth operation

---

## System Architecture

The workflow of the system:

1. Webcam captures live video frames
2. MediaPipe extracts 21 hand landmarks
3. Landmark coordinates are normalized
4. Feature vectors are passed into a KNN classifier
5. Classified gestures trigger:
   - Cursor control actions
   - Sign language token generation
6. Recognized tokens are converted into speech output

The same landmark extraction pipeline powers both interaction modes simultaneously.

---

## Tech Stack

### Languages
- Python

### Libraries & Frameworks
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- pyttsx3
- PyAutoGUI
- Threading

### Concepts Used
- Computer Vision
- Human-Computer Interaction (HCI)
- Machine Learning
- Real-Time Gesture Recognition
- Assistive Technology

---

## Machine Learning Pipeline

### Feature Extraction
- 21 hand landmarks detected per frame
- Wrist-relative normalization
- 42-dimensional feature vector generation

### Classification
The system uses a K-Nearest Neighbors (KNN) classifier because:
- Landmark vectors form compact clusters
- Lightweight inference
- No GPU requirement
- Fast real-time predictions

---

## Supported Gestures

| Gesture | Action |
|---|---|
| Index Finger | Move Cursor |
| Thumb + Index Pinch | Left Click |
| Long Pinch | Drag |
| Two Fingers | Right Click |
| Open Palm | Speak Sentence |
| Three Fingers | Clear Buffer |
| Closed Fist | Sign Recognition |

---

## Performance Metrics

| Metric | Result |
|---|---|
| Accuracy | 94.2% |
| Precision | 93.1% |
| Recall | 94.8% |
| F1-Score | 0.940 |
| FPS | 21–35 FPS |
| Inference Latency | 28 ms |

The system was trained and evaluated on a custom dataset containing approximately 1400 labeled gesture samples collected using a standard webcam.

---

## Project Structure

```bash
Gesture-Based-System-Control/
│
├── dataset/                  # Gesture dataset
├── model/                    # Trained ML model
├── outputs/                  # Output screenshots/results
├── src/
│   ├── feature_extraction.py
│   ├── classifier.py
│   ├── cursor_control.py
│   ├── speech_module.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Gesture-Based-System-Control.git
cd Gesture-Based-System-Control
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Linux / Mac
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python main.py
```

---

## Experimental Results

The system achieved stable real-time performance under standard indoor lighting conditions using only a consumer-grade laptop webcam.

Observations:
- Robust landmark tracking
- Smooth cursor interaction
- Reliable sentence accumulation
- Low-latency speech generation
- Minimal frame drops due to multi-threading

Minor classification errors were observed under:
- Poor lighting
- Fast hand motion
- Finger occlusion
- Side-angle gestures

---

## Research Contribution

Unlike many existing systems that focus solely on:
- sign language recognition, or
- gesture-based cursor control

this project integrates both functionalities into a single lightweight assistive framework without requiring expensive sensors or specialized hardware.

---

## Future Improvements

- Larger sign vocabulary
- Transformer/LSTM based sequence models
- Multi-language speech synthesis
- Improved low-light robustness
- Mobile deployment
- Personalized gesture calibration

---

## Research Paper

This project was developed as part of an academic research initiative at:

Department of Electrical & Electronics Engineering  
MIT World Peace University  
Pune, India

Title:
**Gesture-Based System Control and Sign Language to Speech Conversion**

---

## Authors

### Smit Bhagwat
B.Tech ECE-AIML  
MIT-WPU Pune

### Unnati Bhanushali
Department of Electrical & Electronics Engineering  
MIT-WPU Pune

---

## License

This project is intended for educational, research, and accessibility purposes.

MIT License