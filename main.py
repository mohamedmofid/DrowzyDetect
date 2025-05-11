import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import time


from utils.ear import calculate_EAR
from utils.visuals import draw_landmarks_on_image  # keep this if you have it

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 387, 385, 263, 380, 373]

EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 0
CLOSED_FRAMES_LIMIT = 20  # how many consecutive frames eyes must be closed


# Setup model and options
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path='models/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.VIDEO,  # important for webcam
    result_callback=None  # we’ll use synchronous processing here
)

# Create the FaceLandmarker object
detector = vision.FaceLandmarker.create_from_options(options)

# Open webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened, press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to RGB and mediapipe format
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect landmarks
    # detection_result = detector.detect(mp_image)

    timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    ear_avg = 1  # Default to a non-drowsy state

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        ear_left = calculate_EAR(landmarks, LEFT_EYE_INDICES, frame.shape[1], frame.shape[0])
        ear_right = calculate_EAR(landmarks, RIGHT_EYE_INDICES, frame.shape[1], frame.shape[0])
        ear_avg = (ear_left + ear_right) / 2

    

    if ear_avg < EAR_THRESHOLD:
        CLOSED_FRAMES += 1
        if CLOSED_FRAMES >= CLOSED_FRAMES_LIMIT:
            cv.putText(frame, "DROWSINESS DETECTED!", (30, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            print('DROWSINESS DETECTED!')
    else:
        CLOSED_FRAMES = 0

    # Draw landmarks
    annotated_frame = draw_landmarks_on_image(frame_rgb, detection_result)

    # Convert RGB to BGR for display
    cv.imshow('Driver Drowsiness Detection', cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR))

    # Exit on ESC
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
