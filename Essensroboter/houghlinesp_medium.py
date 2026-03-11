import PIL
import PIL.Image as Image
import cv2         #for canny edge detection and hough line detection algorithm
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
   # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",gray_frame)
    #image = cv2.imread('90_2.jpeg')
   # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    cv2.imshow("edges", edges)

    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    #print(lines)
    angles = []

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(gray, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))  # find angle of line connecting (0,0) to (x,y) from +ve x axis
        #print('Tilt angle for x1, y1, x2, y2 {} is {}'.format([x1, y1, x2, y2], angle))
        angles.append(angle)
        median_angle = np.median(angles)
        print('Angle (- is ccw, + is cw):', median_angle)
    time.sleep(1)

    key = cv2.waitKey(1)
    if key == 27:
        break