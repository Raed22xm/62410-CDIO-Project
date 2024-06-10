import cv2
import numpy as np
import math

def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(robot_center, ball_center):
    # Calculate the angle between the robot's heading and the ball position
    delta_x = ball_center[0] - robot_center[0]
    delta_y = ball_center[1] - robot_center[1]
    return math.atan2(delta_y, delta_x) * 180 / math.pi

def detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 0, 100)
    cv2.imshow("Edges", edges)
    # Find contours in the edge image for ball detection
    ball_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter ball contours based on area and circularity
    min_ball_area = 30
    max_ball_area = 200
    min_ball_circularity = 0.5
    detected_balls = []
    for contour in ball_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if area > min_ball_area and area < max_ball_area and circularity > min_ball_circularity:
                if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[1]:
                    detected_balls.append((x, y, radius))
    return detected_balls

def detect_black_and_yellow_robots(frame, rect_bottom_left, rect_top_right, min_robot_area):
    # Define the lower and upper boundaries for the yellow and black color ranges
    lower_yellow = np.array([20, 50, 200])
    upper_yellow = np.array([40, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 255, 50])
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create masks for yellow and black regions
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    # Find contours for yellow and black regions
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter yellow and black contours based on position and size within the specified rectangle
    yellow_robot = None
    black_robot = None
    for contour in yellow_contours:
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        area = w * h
        if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[1] and area > min_robot_area:
            yellow_robot = (top_left, bottom_right)
    for contour in black_contours:
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        area = w * h
        if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[1] and area > min_robot_area:
            if yellow_robot is None or not (top_left[0] > yellow_robot[1][0] and bottom_right[0] < yellow_robot[0][0]):
                black_robot = (top_left, bottom_right)
    return yellow_robot, black_robot

def detect_table_tennis_balls_and_robots():
    # Open the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Define the dimensions of the rectangle for ball detection
    rect_bottom_left = (20, 20)
    rect_top_right = (600, 450)  # Initial values (adjust as needed)
    min_area = 500
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        # Detect table tennis balls
        detected_balls = detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right)
        # Detect black robots
        yellow_robot, black_robot = detect_black_and_yellow_robots(frame, rect_bottom_left, rect_top_right, min_area)
        # Draw detected circles for balls and add text
        for (x, y, radius) in detected_balls:
            center = (int(x), int(y))
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            # Add text for ball position
            cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Draw rectangles for robots
        if yellow_robot is not None:
            top_left, bottom_right = yellow_robot
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        if black_robot is not None:
            top_left, bottom_right = black_robot
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        # Display the frame with detected balls and robots
        cv2.imshow("Table Tennis Ball and Robot Detection", frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the camera and destroy windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to detect table tennis balls and robots from the camera
detect_table_tennis_balls_and_robots()
