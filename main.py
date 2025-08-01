import cv2
import numpy as np
from hand_tracker import get_fingertip, track_fingertip_motion
from utils import draw_fingertip

cap = cv2.VideoCapture(0)

prev_gray = None
prev_point = None
tap_in_progress = False
tap_marker = None
tap_timer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tip, _ = get_fingertip(frame)

    if tip:
        draw_fingertip(frame, tip)
        current_point = np.array([[tip]], dtype=np.float32)

        #optical flow and tap detction
        next_point, tap_detected, tap_in_progress = track_fingertip_motion(
            prev_gray, gray, prev_point, current_point, tap_in_progress
        )

        #draw line of motion
        if next_point is not None:
            x1, y1 = prev_point[0][0]
            x2, y2 = next_point[0][0]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        if tap_detected:
            print("Tap Detected!")
            tap_marker = tip
            tap_timer = 90  # 3 sec

        prev_point = current_point
        prev_gray = gray.copy()

    #draw a circle if tap was recently detected
    if tap_timer > 0 and tap_marker:
        cv2.circle(frame, tap_marker, 40, (0, 0, 255), 3)
        tap_timer -= 1

    cv2.imshow("phantom-touch", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
