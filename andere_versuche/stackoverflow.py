# Source - https://stackoverflow.com/a/71957500
# Posted by fmw42
# Retrieved 2026-03-04, License - CC BY-SA 4.0
# Refactored for continuous camera stream processing

import cv2
import numpy as np

# initialize camera capture
cap = cv2.VideoCapture(0)  # 0 for default camera

# set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# morphology kernels
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 29))

# threshold values
lower = (0, 0, 0)
upper = (20, 20, 20)
min_area = 1000

print("Press 'q' to quit the camera stream")

while True:
    ret, img = cap.read()
    
    if not ret:
        print("Failed to read frame from camera")
        break
    
    # median blur
    median = cv2.medianBlur(img, 5)
    
    # threshold on black
    thresh = cv2.inRange(median, lower, upper)
    
    # apply morphology open and close
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close)
    
    # filter contours on area
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    result = img.copy()
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            cv2.drawContours(result, [c], -1, (0, 0, 255), 2)
    
    # display results
    cv2.imshow("threshold", thresh)
    cv2.imshow("morphology", morph)
    cv2.imshow("result", result)
    
    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
