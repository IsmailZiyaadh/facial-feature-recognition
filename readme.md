# Facial Feature Detection System

This project is a facial feature detection system built using Python, Dlib’s 68-point facial landmark model, and a Tkinter GUI. It detects and labels major facial features in an image and allows users to view and save the results.

## Features

- Detects and highlights:
  - Jawline
  - Mouth
  - Nose
  - Left and Right Eyes
  - Left and Right Eyebrows
  - Face Outline
- Color-coded bounding boxes and feature labels
- Visual legend added to processed image
- User-friendly GUI for image selection and saving
- Error handling for invalid images or no face detected

## How to Run

### 1. Create a virtual environment: python -m venv venv
### 2. Activate it: 
       - Windows Command Prompt: venv\Scripts\activate
       - Windows PowerShell: .\venv\Scripts\activate
       - Git Bash on Windows: source venv/Scripts/activate
       - Mac/Linux: source venv/bin/activate
### 3. Install dependencies: pip install -r requirements.txt
### 4. Run the program: python main.py

## Directory Structure
.
├── main.py
├── face_feature.py
├── data/
│   └── shape_predictor_68_face_landmarks.dat
├── test_images/
├── results/
├── requirements.txt
└── README.md

## Dependencies
dlib==19.24.6
numpy==2.2.4
opencv-python==4.11.0.86
pillow==11.1.0
