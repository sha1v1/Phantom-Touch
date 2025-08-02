import cv2
import numpy as np
from hand_tracker import get_fingertip, track_fingertip_motion
from utils import draw_fingertip, fft_ripple, detect_edges_near_tip

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

    if tip is not None:
        edge_strength, edge_patch, (ex, ey) = detect_edges_near_tip(gray, tip)

        if edge_strength > 50:
            print("Fingertip visible")
            draw_fingertip(frame, tip)
            # cv2.imshow("Edges around fingertip", edge_patch)
            cv2.rectangle(frame, (ex, ey), (ex + 50, ey + 50), (255, 255, 0), 1)

            current_point = np.array([[tip]], dtype=np.float32)
            next_point, tap_detected, tap_in_progress = track_fingertip_motion(prev_gray, gray, prev_point, current_point, tap_in_progress)

            if tap_detected and edge_strength > 100:
                print("Tap Detected!")
                ripples.append((tip, 0))
                if len(ripples) > 3:
                    ripples.pop(0)

            prev_point = current_point
            prev_gray = gray.copy()
        else:
            print("Fingertip hidden or weak (low edge strength)")
    else:
        print("Fingertip not detected")

    #apply and update ripple effects
    ripple_frame = frame.copy()
    updated_ripples = []
    for center, age in ripples:
        if age < 15:  
            ripple_frame = fft_ripple(ripple_frame, center, age)
            updated_ripples.append((center, age + 1))
    ripples = updated_ripples

    cv2.imshow("phantom-touch", ripple_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
