import cv2
import numpy as np

# initialize camera capture
image = cv2.VideoCapture(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, threshold=100, minLineLength=50, maxLineGap=10)


# Draw the lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Lines Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()