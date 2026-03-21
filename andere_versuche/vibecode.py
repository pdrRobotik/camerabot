import cv2
import numpy as np
from collections import deque

# Camera setup
cv2.namedWindow("Line Follower")
vc = cv2.VideoCapture(0)  # change to 1 if you have a second camera, etc.

if not vc.isOpened():
    print("Error: Could not open video stream")
    exit()

# Set camera resolution for better performance (optional)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Parameters for line detection
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200
BLUR_KERNEL = 5

# Region of interest (lower part of frame where the tape is visible)
ROI_TOP = 240  # Start from middle of frame downwards
ROI_HEIGHT = 240  # Height of ROI

while True:
    res = vc.read()
    rval = res[0]
    frame = res[1]
    
    if not rval:
        break
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    goal_x = w // 2  # Goal line is in the center of the frame
    
    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(frame, BLUR_KERNEL)
    
    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2, apertureSize=3)
    
    # Create ROI (Region of Interest) - focus on lower part of frame
    roi_edges = edges[ROI_TOP:ROI_TOP + ROI_HEIGHT, :]
    roi_frame = frame[ROI_TOP:ROI_TOP + ROI_HEIGHT, :]
    
    # Find contours in the ROI
    contours, hierarchy = cv2.findContours(roi_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize line position as None
    line_x = None
    line_width = None
    line_found = False
    
    # Process contours to find the tape line
    if contours:
        # Filter contours by area (ignore very small noise)
        min_area = 50
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if significant_contours:
            # Sort contours by area (largest first)
            sorted_contours = sorted(significant_contours, key=cv2.contourArea, reverse=True)
            
            # Take the top 2-3 contours (in case the line is split into left and right edges)
            top_contours = sorted_contours[:3]
            
            # Calculate the combined centroid of all selected contours
            total_m10 = 0
            total_m01 = 0
            total_m00 = 0
            all_moments = []
            
            for contour in top_contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    all_moments.append(M)
                    total_m10 += M["m10"]
                    total_m01 += M["m01"]
                    total_m00 += M["m00"]
            
            # Calculate combined centroid
            if total_m00 > 0:
                cx = int(total_m10 / total_m00)
                cy = int(total_m01 / total_m00)
                
                # Calculate the width as the distance between leftmost and rightmost contours
                x_positions = []
                widths = []
                for contour in top_contours:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    x_positions.append(x)
                    widths.append(w_rect)
                
                if x_positions:
                    leftmost = min(x_positions)
                    rightmost = max(x_positions) + max(widths)
                    line_width = rightmost - leftmost
                else:
                    line_width = 0
                
                line_x = cx
                line_found = True
                
                # Draw all contours on the ROI frame
                cv2.drawContours(roi_frame, top_contours, -1, (0, 255, 0), 2)
                
                # Draw a circle at the center of the line
                cv2.circle(roi_frame, (cx, cy), 5, (0, 255, 0), -1)
    
    # Draw the goal line (center of frame) on the ROI
    goal_line_y_start = 0
    goal_line_y_end = ROI_HEIGHT
    cv2.line(roi_frame, (goal_x, goal_line_y_start), (goal_x, goal_line_y_end), (255, 0, 0), 3)
    
    # Calculate deviation and print to terminal
    if line_found and line_x is not None:
        deviation = line_x - goal_x  # Positive = line to the right, Negative = line to the left
        
        # Print the deviation (this can be used to control the steppers)
        print(f"Line detected | X: {line_x:3d} | Goal: {goal_x:3d} | Deviation: {deviation:4d} px | Width: {line_width:3d} px")
        
        # Draw deviation information on the frame
        dev_text = f"Dev: {deviation:+d} px"
        cv2.putText(roi_frame, dev_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw a vertical line at the detected line position
        cv2.line(roi_frame, (line_x, 0), (line_x, ROI_HEIGHT), (0, 255, 255), 2)
    else:
        print("Line not detected")
        cv2.putText(roi_frame, "No line detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the full frame with goal line
    cv2.line(frame, (goal_x, ROI_TOP), (goal_x, ROI_TOP + ROI_HEIGHT), (255, 0, 0), 2)
    
    # Show windows
    cv2.imshow("Line Follower", roi_frame)
    cv2.imshow("Full Frame", frame)
    cv2.imshow("Canny Edges", roi_edges)
    
    # Press ESC to exit
    key = cv2.waitKey(20)
    if key == 27:  # ESC key
        break

vc.release()
cv2.destroyAllWindows()
print("Line follower stopped")
