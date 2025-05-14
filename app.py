import cv2 as cv
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the alert sound
alert_sound = pygame.mixer.Sound("assets/mixkit-residential-burglar-alert-1656.wav")
# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="models/eye_state_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Eye indices for cropping
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 387, 385, 263, 380, 373]

# Drowsiness detection settings
CLOSED_FRAMES = 0
CLOSED_FRAMES_LIMIT = 20
ALERT_PLAYING = False  # To prevent overlapping sounds

# Webcam setup
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def preprocess_eye(eye_image):
    eye_image = cv.resize(eye_image, (24, 24))  # Resize to match model input
    eye_image = cv.cvtColor(eye_image, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    eye_image = eye_image / 255.0  # Normalize to [0, 1]
    eye_image = np.expand_dims(eye_image, axis=-1)  # Add channel dimension
    eye_image = np.expand_dims(eye_image, axis=0)  # Batch dimension
    eye_image = eye_image.astype(np.float32)  # TFLite requires float32
    return eye_image

def predict_eye_state(eye_image):
    interpreter.set_tensor(input_details[0]['index'], eye_image)
    interpreter.invoke()  # Run inference
    output = interpreter.get_tensor(output_details[0]['index'])
    return int(output[0][0] > 0.5)  # Return 1 for open, 0 for closed

def extract_eye(frame, landmarks, eye_indices):
    h, w = frame.shape[:2]
    eye_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_min = min([coord[0] for coord in eye_coords])
    x_max = max([coord[0] for coord in eye_coords])
    y_min = min([coord[1] for coord in eye_coords])
    y_max = max([coord[1] for coord in eye_coords])
    eye_crop = frame[y_min:y_max, x_min:x_max]
    return eye_crop

def play_alert_sound():
    if not pygame.mixer.get_busy():  # Play sound only if not already playing
        alert_sound.play()

def stop_alert_sound():
    pygame.mixer.stop()  # Stop any ongoing sound

print("Webcam opened, press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Extract left and right eye crops
        left_eye_crop = extract_eye(frame, landmarks, LEFT_EYE_INDICES)
        right_eye_crop = extract_eye(frame, landmarks, RIGHT_EYE_INDICES)

        try:
            left_eye_state = predict_eye_state(preprocess_eye(left_eye_crop))
            right_eye_state = predict_eye_state(preprocess_eye(right_eye_crop))

            # Check for drowsiness
            if left_eye_state == 0 and right_eye_state == 0:
                CLOSED_FRAMES += 1
                if CLOSED_FRAMES >= CLOSED_FRAMES_LIMIT:
                    cv.putText(frame, "DROWSINESS DETECTED!", (30, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    print("DROWSINESS DETECTED!")
                    # Play alert sound in a separate thread
                    play_alert_sound()
            else:
                stop_alert_sound()
                CLOSED_FRAMES = 0

            # Display eye states
            cv.putText(frame, f"Left Eye: {'Open' if left_eye_state else 'Closed'}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame, f"Right Eye: {'Open' if right_eye_state else 'Closed'}", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            print("Error during eye prediction:", e)

    # Show frame
    cv.imshow("Driver Drowsiness Detection", frame)

    # Exit on ESC
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
