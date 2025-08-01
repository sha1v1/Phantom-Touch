import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def get_fingertip(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        return (x, y), hand_landmarks
    return None, None



def track_fingertip_motion(prev_gray, gray, prev_point, current_point, tap_in_progress):
    tap_detected = False
    next_point = None

    if prev_gray is not None and prev_point is not None:
        next_point, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_point, None,
            winSize=(50, 50),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        if status[0][0] == 1:
            x1, y1 = prev_point[0][0]
            x2, y2 = next_point[0][0]
            velocity = np.hypot(x2 - x1, y2 - y1)

            if velocity > 20:
                tap_in_progress = True
            elif tap_in_progress and velocity < 3:
                tap_detected = True
                tap_in_progress = False

    return next_point, tap_detected, tap_in_progress
