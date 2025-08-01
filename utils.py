import cv2
import numpy as np

def draw_fingertip(frame, point, color=(0, 0, 255)):
    if point:
        cv2.circle(frame, point, 8, color, -1)


def fft_ripple(wave_freq=0.03, size=400, phase=0):
    h, w = size, size
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Circular wavefront: sine of distance with animated phase
    wave = np.sin(dist * wave_freq * 2 * np.pi + phase)

    # Normalize
    wave -= wave.min()
    wave /= wave.max()

    # Falloff mask
    max_radius = size // 2
    falloff = np.clip(1 - dist / max_radius, 0, 1)
    ripple = wave * (falloff ** 3)

    return ripple.astype(np.float32)