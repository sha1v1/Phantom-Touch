import cv2
import numpy as np
from hand_tracker import get_fingertip, track_fingertip_motion
from utils import draw_fingertip, fft_ripple

cap = cv2.VideoCapture(0)

prev_gray = None
prev_point = None
tap_in_progress = False
ripples = []  # List of tuples: (position, age)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  #mirror image (for testing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tip, _ = get_fingertip(frame)

    if tip:
        draw_fingertip(frame, tip)
        current_point = np.array([[tip]], dtype=np.float32)

        next_point, tap_detected, tap_in_progress = track_fingertip_motion( prev_gray, gray, prev_point, current_point, tap_in_progress)

        if tap_detected:
            print("Tap Detected!")
            ripples.append((tip, 0))  #add new ripple at this location

        prev_point = current_point
        prev_gray = gray.copy()

    #apply and update ripple effects
    ripple_frame = frame.copy()
    updated_ripples = []
    for center, age in ripples:
        if age < 30:  # ripple lasts ~1 sec
            ripple_frame = fft_ripple(ripple_frame, center, age)
            updated_ripples.append((center, age + 1))
    ripples = updated_ripples

    cv2.imshow("phantom-touch", ripple_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
