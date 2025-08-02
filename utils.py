import cv2
import numpy as np

def draw_fingertip(frame, point, color=(0, 0, 255)):
    if point:
        cv2.circle(frame, point, 8, color, -1)

def detect_edges_near_tip(gray, tip, size=25):
    x, y = tip
    h, w = gray.shape

    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w, x + size)
    y2 = min(h, y + size)

    patch = gray[y1:y2, x1:x2]
    edges = cv2.Canny(patch, 50, 150)

    #count edge pixels
    edge_strength = np.count_nonzero(edges)

    return edge_strength, edges, (x1, y1)


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
    base_freq = 0.0001         #for wider ripple spacing
    base_amplitude = 120      #max distortion

    freq = base_freq * (1 - t + 0.3)      
    amplitude = base_amplitude * t        
    decay = 0.01
    wave_speed = 2.0

    ripple = np.sin(dist * freq - age * wave_speed)
    base_offset = ripple * np.exp(-dist * decay)

    #gaussian envelope
    sigma = 80  #wider=smoother ripple
    gaussian = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    #apply gaussian modulated offset
    offset = base_offset * gaussian * amplitude

    #directin of displacement (normalized)
    dx_norm = dx / (dist + 1e-5)
    dy_norm = dy / (dist + 1e-5)

    map_x = (X + dx_norm * offset).astype(np.float32)
    map_y = (Y + dy_norm * offset).astype(np.float32)

    #warp
    displaced = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
 

    #create a colored halo around the ripple
    color_mask = np.abs(ripple) * gaussian * 255 * t
    color_mask = np.clip(color_mask, 0, 255).astype(np.uint8)

    #create rgb halo
    color_ripple = np.zeros_like(frame)
    color_ripple[:, :, 0] = color_mask            #blue channel
    color_ripple[:, :, 1] = color_mask // 2       #teal 

    #overlay color ripple onto distorted image
    colored_ripple = cv2.addWeighted(displaced, 1.0, color_ripple, 0.4, 0)

    return colored_ripple
