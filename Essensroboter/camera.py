import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0) # change to 1 if you have a second camera, etc.

if vc.isOpened(): # try to get the first frame
    res = vc.read()
    rval = res[0]
    frame = res[1]
else:
    rval = False

while rval:
    # Lese zuerst das nächste Frame
    res = vc.read()
    rval = res[0]
    frame = res[1] 
    if not rval:
        break

    # Größe aus dem Frame bestimmen (Breite = shape[1], Höhe = shape[0])
    h, w = frame.shape[:2]
    cx = w // 2

    # Linie auf das Frame zeichnen (vor dem Anzeigen)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 5)

    # Dann Bild anzeigen und auf Taste warten
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
cv2.destroyWindow("preview")
