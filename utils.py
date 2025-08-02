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

    max_age = 50

    # parameters
    t = age / max_age

    #the ripples fade out progressively, hence the changing prarams
    base_freq = 0.03         #for wider ripple spacing
    base_amplitude = 25      #max distortion

    freq = base_freq * (1 - t + 0.3)      
    amplitude = base_amplitude * t        
    decay = 0.01
    wave_speed = 2.0

    ripple = np.sin(dist * freq - age * wave_speed)
    offset = ripple * np.exp(-dist * decay) * amplitude

    #direction of displacement (normalized)
    dx_norm = dx / dist
    dy_norm = dy / dist

    map_x = (X + dx_norm * offset).astype(np.float32)
    map_y = (Y + dy_norm * offset).astype(np.float32)

    displaced = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return displaced