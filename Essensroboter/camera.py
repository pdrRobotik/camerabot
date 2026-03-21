import cv2
import numpy as np

lower = np.array([0, 0, 0])      # Dark end of black color dectection params
upper = np.array([100, 100, 100])   # Adjust based on how dark your line is

edges_prev = None


def canny_image_process(frame, height):
    global edges_prev
    
    median = cv2.medianBlur(frame, 5) #donoise for detection
    
    # roi config
 
    region_of_interest = frame[int(height*0.3):, :]#nur untere Hälfte des Bildes betrachten, da oben rauschen, 0.5 anpassbar

    
    
    #grayscale
    hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)#grayscale für bessere kanten erkennung, da farbe nicht relevant
    v_channel = hsv[:, :, 2] #nur v channel für helligkeit, da schwarz-weiß linien erkannt werden sollen

    roi_blurred = cv2.GaussianBlur(v_channel, (5, 5), 0)

    #canny edge detection with dynamic thresholds based on median
    median = np.median(roi_blurred)
    lower = int(max(0, 0.5 * median))
    upper = int(min(255, 1.3 * median))
    edges = cv2.Canny(roi_blurred, lower, upper, apertureSize=3)

    if edges_prev is not None:
        edges = cv2.addWeighted(edges, 0.7, edges_prev, 0.3, 0) #mit vorherigem frame mischen für stabilere linien

    edges_prev = edges.copy() # Speichern des aktuellen Frames für den nächsten Durchlauf

    lines = cv2.HoughLinesP(
        edges, #input canny edges
        1,
        np.pi / 180, 
        threshold=50, #bei rauschen höher setzen, bei klaren Linien niedriger
        minLineLength=100, #kleine lienen ignorieren, anpassbar
        maxLineGap=50 #erkungslücken in Linien überbrücken, anpassbar
        )
    return lines, edges 

def draw_middle_line(frame):
    h, w = frame.shape[:2]
    cx = w // 2
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 3) #grüne mittellinie zeichnen

def draw_hough_lines(frame, lines, height):

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1+int(height*0.5)), (x2, y2+int(height*0.5)), (0, 0, 255), 3) #rote Linien zeichnen


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
    

    height, width = frame.shape[:2]
    lines, edges = canny_image_process(frame, height)

    draw_middle_line(frame)
    
    draw_hough_lines(frame, lines, height)


    # Dann Bild anzeigen und auf Taste warten
    cv2.imshow("camerafeed", frame)
    cv2.imshow("canny", edges)
    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
cv2.destroyWindow("preview")