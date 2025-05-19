# DrowzyDetect
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/mohamedmofid/DrowzyDetect)

DrowzyDetect is a real-time driver drowsiness detection system that uses a webcam to monitor the driver's eyes. It employs computer vision techniques to determine if the driver is becoming drowsy and triggers an audible alert.

## Overview

This project provides two primary methods for detecting drowsiness:
1.  **Using a TFLite model (`app.py`):** This approach utilizes a pre-trained TensorFlow Lite model to classify the state of each eye (open or closed).
2.  **Using Eye Aspect Ratio (EAR) (`main.py`):** This method calculates the Eye Aspect Ratio using facial landmarks detected by MediaPipe. A low EAR value for a sustained period indicates closed eyes.

When drowsiness is detected (eyes closed for a predefined number of consecutive frames), an alert sound is played to warn the driver.

## Features

*   Real-time drowsiness detection using a standard webcam.
*   Face and eye landmark detection using MediaPipe.
*   Two distinct drowsiness detection algorithms:
    *   TFLite-based eye state classification.
    *   Eye Aspect Ratio (EAR) calculation.
*   Audible alert system using Pygame for sound playback.
*   Visual feedback on the video stream showing eye status and drowsiness alerts.

## How it Works

### 1. TFLite Eye State Model (`app.py`)
This script uses MediaPipe Face Mesh to detect facial landmarks and identify the regions corresponding to the left and right eyes.
-   The eye regions are cropped from the video frame.
-   Each eye crop is preprocessed (resized to 24x24, converted to grayscale, normalized, and dimensions expanded) to match the input requirements of the `eye_state_model.tflite`.
-   The TFLite interpreter predicts the state of each eye (0 for closed, 1 for open).
-   If both eyes are detected as closed for a specified number of consecutive frames (`CLOSED_FRAMES_LIMIT`), a drowsiness alert is triggered, and a sound is played.

### 2. Eye Aspect Ratio (EAR) (`main.py`)
This script utilizes the MediaPipe FaceLandmarker task to obtain detailed 3D facial landmarks.
-   The Eye Aspect Ratio (EAR) is calculated for both the left and right eyes using specific landmark points. EAR is a ratio of distances between vertical and horizontal eye landmarks.
-   An average EAR is computed from both eyes.
-   If the average EAR falls below a predefined threshold (`EAR_THRESHOLD`) for a specified number of consecutive frames (`CLOSED_FRAMES_LIMIT`), it signifies potential drowsiness, and an alert message is displayed on the screen.
-   The script also visualizes the detected facial landmarks.

## Project Structure

```
mohamedmofid-drowzydetect/
├── app.py                     # Main application using TFLite model for eye state
├── main.py                    # Main application using Eye Aspect Ratio (EAR)
├── requirments.txt            # Python dependencies
├── assets/
│   └── mixkit-residential-burglar-alert-1656.wav # Alert sound
├── models/
│   ├── eye_state_model.tflite # TFLite model for eye state classification
│   └── face_landmarker_v2_with_blendshapes.task # MediaPipe face landmarker model
└── utils/
    ├── __init__.py
    ├── ear.py                 # Function to calculate Eye Aspect Ratio
    └── visuals.py             # Functions for drawing landmarks
```

## Prerequisites

*   Python 3.x
*   A webcam connected to your computer.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohamedmofid/DrowzyDetect.git
    cd DrowzyDetect
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirments.txt
    ```

## Usage

You can run either of the main scripts to start the drowsiness detection system.

*   **To run the TFLite-based eye state detection (`app.py`):**
    ```bash
    python app.py
    ```

*   **To run the EAR-based detection (`main.py`):**
    ```bash
    python main.py
    ```

Press the `ESC` key to close the application window and stop the script. The system will display the webcam feed, overlaying eye status information and drowsiness alerts if detected. An audible alert will sound if drowsiness persists.

## Key Dependencies

The `requirments.txt` file lists all Python packages. Key dependencies include:

*   `opencv-python` (cv2): For webcam access and image processing.
*   `mediapipe`: For facial landmark detection.
*   `tflite_runtime`: For running the TensorFlow Lite model (used in `app.py`).
*   `pygame`: For playing the alert sound.
*   `numpy`: For numerical operations.

Make sure these are installed correctly by following the setup instructions.
