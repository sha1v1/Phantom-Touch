import cv2
import numpy as np

def draw_fingertip(frame, point, color=(0, 0, 255)):
    if point:
        cv2.circle(frame, point, 8, color, -1)


def fft_ripple(frame, center, age):
    h, w = frame.shape[:2]
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dx = X - center[0]
    dy = Y - center[1]
    dist = np.sqrt(dx**2 + dy**2)

    # parameters
    wave_speed = 3.0
    decay = 0.03
    freq = 0.3

    offset = np.sin(dist * freq - age * wave_speed) * np.exp(-dist * decay) * 8

    # apply displacement
    map_x = (X + dx / dist * offset).astype(np.float32)
    map_y = (Y + dy / dist * offset).astype(np.float32)

    displaced = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return displaced