import cv2
import numpy as np
from hand_tracker import get_fingertip, track_fingertip_motion
from utils import draw_fingertip, fft_ripple

cap = cv2.VideoCapture(0)

prev_gray = None
prev_point = None
tap_in_progress = False
ripples = []

# Constants
MAX_AGE = 60  #frames
RIPPLE_SIZE = 500  #FFT patch size
TINT_COLOR = np.array([0, 0, 255], dtype=np.float32)  #red colored glow

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
            ripples.append((tip, 0))  #in format (position, age)

        prev_point = current_point
        prev_gray = gray.copy()

    #render riipples
    ripple_mask = np.zeros_like(frame, dtype=np.float32)
    updated_ripples = []

    for (center, age) in ripples:
        intensity = max(0, (1 - age / MAX_AGE) ** 2)
        if intensity <= 0:
            continue

        ripple = fft_ripple(wave_freq=0.015 + age * 0.002, size=RIPPLE_SIZE, phase=0.3)
        h, w = ripple.shape
        cx, cy = center
        x1, x2 = cx - w // 2, cx + w // 2
        y1, y2 = cy - h // 2, cy + h // 2

        #if ripple goes out of bounds
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue

        #color tint (this is for testing mostly)
        colored = np.stack([ripple * c for c in TINT_COLOR], axis=-1)
        ripple_mask[y1:y2, x1:x2] += colored * intensity

        updated_ripples.append((center, age + 1))

    ripples = updated_ripples

    # Blend ripple with base frame
    ripple_mask = np.clip(ripple_mask, 0, 255).astype(np.uint8)
    frame = cv2.add(frame, ripple_mask)

    cv2.imshow("phantom-touch", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
