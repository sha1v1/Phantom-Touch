import numpy as np
import cv2 as cv

# if on a mac, turn off bluetooth (otherwise it connects to the iphone camera lol)

cap = cv.VideoCapture(0,  cv.CAP_ANY)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
prev = None
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if prev is None:
        prev = gray
        continue
    diff = cv.absdiff(prev, gray)
    _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) < 1000: continue
        cv.drawContours(frame, [cnt], 0, (0,255,0), 3)

    # Display the resulting frame
    cv.imshow('phantom-touch', frame)
    
    prev = gray

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()