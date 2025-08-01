import cv2
import numpy as np

def draw_fingertip(frame, point, color=(0, 0, 255)):
    if point:
        cv2.circle(frame, point, 8, color, -1)


def fft_ripple(wave_freq=0.05, size=200):
    h, w = size, size
    y = np.fft.fftfreq(h).reshape(-1, 1)
    x = np.fft.fftfreq(w).reshape(1, -1)
    radius = np.sqrt((x)**2 + (y)**2)

    wave = np.sin(2 * np.pi * radius / wave_freq)
    wave[radius == 0] = 0

    ripple = np.fft.ifft2(np.fft.ifftshift(wave)).real
    ripple -= ripple.min()
    ripple /= ripple.max()

    # circular falloff 
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    falloff = np.clip(1 - dist / (w // 2), 0, 1)
    ripple *= falloff ** 2  #smooth edges

    return ripple.astype(np.float32)