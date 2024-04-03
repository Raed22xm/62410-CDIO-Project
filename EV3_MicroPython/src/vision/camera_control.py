import cv2 
import numpy as np

# Open a handle to the USB camera
#choosing index 1 because we use a laptop with an external usb camera , that's why we choose the second camera 

def open_camera(camera_index=1): 
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"No camera found at index {camera_index}- please reconnect your USB-camera")
        return None
    
    return cap
        
        
# if you have other cameras 
# Attempt to open the first few indices. This can be deleted it just checks if there are multiple cameras and choose the active one 
'''
 for index in range(0, 4):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"No camera found at index {index}")
        cap.release()
        continue
    else:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Test Frame from Camera Index {index}", frame)
            cv2.waitKey(1000)  # Display each frame for 1000 milliseconds
            print(f"Camera found at index {index}")
        cap.release()

cv2.destroyAllWindows() 
'''
class DetectedCircles:
    def __init__(self,x,y, radius):
        self.x=x
        self.y = y
        self.radius = radius
'''class DetectedRobot:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h'''    
class DetectedRobot:
    def __init__(self, vertices):
        self.vertices = vertices  # A list of three points [(x1, y1), (x2, y2), (x3, y3)] cause it is a triangle
        
# Create trackbars for color change
# The arguments are: trackbarName, windowName, value, count, onChange

#def detect(cap,robot_template):
def detect(cap):
 cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

 balls_list = [] # List to store the deteced balls 
 robots_list = []
 fields_list = []
 #robot_position = None 
 while True:
    # Capturing frames
    ret, frame = cap.read()
    if not ret:
        break
    
    #dimensions
    #width = int(cap.get(3))
    #height = int(cap.get(4))
    #image = np.zeros(frame.shape, np.uint8)
    #smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Convert to grayscale for thresholding and circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects using the classifier
    objects = cascade_classifier.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw green rectangles

    # Display the resulting frame
    cv2.imshow('frame', frame)
            #balls_list.append(DetectedCircles(i[0], i[1], i[2]))
            
    #Field detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask1, mask2)    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_contour = []
    for contour in contours:
    # Approximate the contour to reduce the number of points
     epsilon = 0.05 * cv2.arcLength(contour, True)
     approx = cv2.approxPolyDP(contour, epsilon, True)
      # Check if the contour has 4 vertices and a significant area to be considered a rectangle
     if len(approx) == 4 and cv2.contourArea(contour) > 1 :  # You can adjust the area threshold as necessary
        rectangle_contour.append(contour)
    
    # Find the largest contour for better detection Test 
    '''largest_contour = max(contours, key=cv2.contourArea)'''
    # Assuming the field is the largest rectangle, find the max area rectangle
    if rectangle_contour:
     largest_rectangle = max(rectangle_contour, key=cv2.contourArea)
     # Draw contours on the original image
     cv2.drawContours(frame, rectangle_contour, -1, (255, 255, 255), 3)
     #fields_list.append((i[0], i[1], i[2]))
    else: 
     print("No field found")
    '''cv2.putText(frame, "field", text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 255), 1, cv2.LINE_AA)'''
                
    #Robot detection using color and a triangle
    '''gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)'''
    lower_yellow = np.array([20, 100, 100]) 
    upper_yellow = np.array([35, 255, 255])
    maskRobot = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskRobot = cv2.morphologyEx(maskRobot, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


    contours, _ = cv2.findContours(maskRobot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_triangle_area = 0
    largest_triangle_approx = None
    # Loop over the contours
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # If the polygon has 3 vertices, it is a triangle
        if len(approx) == 3:
            area = cv2.contourArea(cnt)
            if area > largest_triangle_area:
             largest_triangle_area = area
             largest_triangle_approx = approx
            if largest_triangle_approx is not None:
            # Draw the contour of the triangle
             cv2.drawContours(frame, [largest_triangle_approx], -1, (0, 255, 0), 3)
             cv2.putText(frame, "Robot", (largest_triangle_approx[0][0][0], largest_triangle_approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             robots_list.append(DetectedRobot(largest_triangle_approx))
            
    # Display and Detect
    cv2.imshow('Frame with Detected Objects-', frame)
    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
    # When everything done, release the capture and destroy all windows
    #cap.release()
    #cv2.destroyAllWindows() 

 #return balls_list, robots_list #,fields_list
 
 

def main():
    #Main function to control the camera and detect objects (balls-obstacles and the field)
    cap = open_camera()
    '''robot_template = cv2.imread('C:\\Users\\fathi\\Documents\\GitHub\\62410-CDIO-Project\\EV3_MicroPython\\data\\images\\prototype.png') 
    #if cap is not None and robot_template is not None:
        balls,robot = detect(cap, robot_template) '''
    if cap is not None:
        detect(cap)
        #balls = detect(cap)
    #print(f"Detected {len(balls)} balls.")
    #for index,ball in enumerate(balls):
     #print(f"Ball {index} at ({ball.x}, {ball.y}) with radius {ball.radius}.")
    '''if robot:
     print(f"Detected robot at ({robot.x}, {robot.y}).")
     else:
        print("No robot detected.")'''
 
   
        
if __name__ == "__main__":
 main()