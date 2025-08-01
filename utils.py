import cv2

def draw_fingertip(frame, point, color=(0, 0, 255)):
    if point:
        cv2.circle(frame, point, 8, color, -1)
