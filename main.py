import numpy as np
import cv2 as cv
from collections import deque

cap = cv.VideoCapture(0, cv.CAP_ANY)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev = None
area_history = deque(maxlen=5)
pos_history = deque(maxlen=5)
tap_cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    if prev is None:
        prev = gray
        continue

    diff = cv.absdiff(prev, gray)
    _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_cnt = max(contours, key=cv.contourArea, default=None)
    tap_detected = False

    if largest_cnt is not None and cv.contourArea(largest_cnt) > 2000:
        area = cv.contourArea(largest_cnt)
        M = cv.moments(largest_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        area_history.append(area)
        pos_history.append((cx, cy))

        #draw
        cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
        cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        #tap detection
        if len(area_history) == 5 and len(pos_history) == 5 and tap_cooldown == 0:
            area_growth = area_history[-1] - area_history[0]
            dx = pos_history[-1][0] - pos_history[0][0]
            dy = pos_history[-1][1] - pos_history[0][1]
            movement = np.hypot(dx, dy)

            #idea: big area increase, some motion, then no growth
            if area_growth > 5000 and movement > 30:
                tap_detected = True
                tap_cooldown = 15  #cooldown to handle false positives

    if tap_cooldown > 0:
        tap_cooldown -= 1

    tap_markers = deque()  #stores tuples of(x, y, remaining_lifetime)
    TAP_MARKER_LIFETIME = 90 

    if tap_detected:
        cv.putText(frame, "Tap Detected!", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tap_markers.append((cx, cy, TAP_MARKER_LIFETIME))
    #draw and update tap markers

    updated_markers = deque()
    for x, y, life in tap_markers:
        if life > 0:
            cv.circle(frame, (x, y), 15, (0, 0, 255), 2)  #red circle
            updated_markers.append((x, y, life - 1))
    tap_markers = updated_markers



    cv.imshow('phantom-touch', frame)
    prev = gray

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
