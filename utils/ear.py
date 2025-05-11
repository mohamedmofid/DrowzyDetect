import numpy as np

def calculate_EAR(landmarks, indices, image_width, image_height):
    # Convert landmark indices to 2D pixel coordinates
    def _get_point(i):
        lm = landmarks[i]
        return np.array([lm.x * image_width, lm.y * image_height])
    
    p1 = _get_point(indices[0])
    p2 = _get_point(indices[1])
    p3 = _get_point(indices[2])
    p4 = _get_point(indices[3])
    p5 = _get_point(indices[4])
    p6 = _get_point(indices[5])

    # EAR calculation
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2 * np.linalg.norm(p1 - p4)
    ear = vertical / horizontal
    return ear
