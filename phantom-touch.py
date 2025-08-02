import cv2
import numpy as np
import sys
from hand_tracker import get_fingertip, track_fingertip_motion
from utils import draw_fingertip, fft_ripple, detect_edges_near_tip

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phantom_touch.py input_video.mp4")
        return

    input_path = sys.argv[1]
    output_path = "output.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_gray = None
    prev_point = None
    tap_in_progress = False
    ripples = []  # List of tuples: (position, age)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tip, _ = get_fingertip(frame)

        if tip is not None:
            edge_strength, _, _ = detect_edges_near_tip(gray, tip)

            if edge_strength > 80:
                draw_fingertip(frame, tip)
                current_point = np.array([[tip]], dtype=np.float32)

                next_point, tap_detected, tap_in_progress = track_fingertip_motion(
                    prev_gray, gray, prev_point, current_point, tap_in_progress
                )

                if tap_detected and edge_strength > 100:
                    ripples.append((tip, 0))
                    if len(ripples) > 3:
                        ripples.pop(0)

                prev_point = current_point
                prev_gray = gray.copy()

        # Apply and update ripple effects
        ripple_frame = frame.copy()
        updated_ripples = []
        for center, age in ripples:
            if age < 15:
                ripple_frame = fft_ripple(ripple_frame, center, age)
                updated_ripples.append((center, age + 1))
        ripples = updated_ripples

        out.write(ripple_frame)

    cap.release()
    out.release()
    print(f"Done. Output saved to {output_path}")

if __name__ == "__main__":
    main()
