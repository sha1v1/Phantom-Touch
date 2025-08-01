import cv2
from hand_tracker import get_fingertip
from utils import draw_fingertip

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tip, landmarks = get_fingertip(frame)
    if tip:
        draw_fingertip(frame, tip)

    cv2.imshow("phantom-touch", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
