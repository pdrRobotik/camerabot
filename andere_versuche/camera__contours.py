import cv2
import numpy as np

lower = np.array([0, 0, 0])      # Dark end of black color dectection params
upper = np.array([100, 100, 100])   # Adjust based on how dark your line is


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0) # change to 1 if you have a second camera, etc.

if vc.isOpened(): # try to get the first frame
    res = vc.read()
    rval = res[0]
    frame = res[1]
else:
    rval = False
    print("Error: Could not open video stream") 

while rval:
    # Lese zuerst das nächste Frame
    res = vc.read()
    rval = res[0]
    frame = res[1] 
    if not rval:
        break

    # median blur
    median = cv2.medianBlur(frame, 5) #donoise for detection
    
    # threshold on black
    # thresh = cv2.inRange(median, lower, upper) switched to gray due to lighting 
    #cv2.imshow("threshold", thresh)

    # compute edges from the current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,1,1,apertureSize = 3)

    # Größe aus dem Frame bestimmen (Breite = shape[1], Höhe = shape[0])
    h, w = frame.shape[:2]
    cx = w // 2

    # Linie auf das Frame zeichnen (vor dem Anzeigen)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 3)

    # Hough lines (use cv2 and guard if no lines found)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)


    # if lines is not None:
    #     for line in lines:
    #         rho,theta = line[0]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))
    #      
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

    # Dann Bild anzeigen und auf Taste warten
    cv2.imshow("camerafeed", frame)
    cv2.imshow("canny", edges)
    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
cv2.destroyWindow("preview")
