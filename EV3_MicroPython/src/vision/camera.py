import cv2
import numpy as np
import math
import json
import os

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(robot_center, ball_center):
    delta_x = ball_center[0] - robot_center[0]
    delta_y = ball_center[1] - ball_center[1]
    return math.atan2(delta_y, delta_x) * 180 / math.pi

def detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 0, 100)
    ball_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                    detected_balls.append((center[0], center[1], radius))
    return detected_balls

def detect_robots(hsv, frame_width):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_robot = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask_robot = cv2.morphologyEx(mask_robot, cv2.MORPH_OPEN, kernel)
    mask_robot = cv2.morphologyEx(mask_robot, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Robot Mask", mask_robot)

    contours, _ = cv2.findContours(mask_robot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_robot_area = 100  
    max_robot_area = frame_width * frame_width // 2
    largest_triangle_area = 0
    largest_triangle_approx = None

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            area = cv2.contourArea(cnt)
            if min_robot_area < area < max_robot_area:
                if area > largest_triangle_area:
                    largest_triangle_area = area
                    largest_triangle_approx = approx

    if largest_triangle_approx is not None:
        print(f"Detected Robot with area: {largest_triangle_area}")

    return largest_triangle_approx

def detect_field(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Field Mask", mask)
    rectangle_contour = []
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 1:
            rectangle_contour.append(contour)
    if rectangle_contour:
        largest_rectangle = max(rectangle_contour, key=cv2.contourArea)
        cv2.drawContours(frame, [largest_rectangle], -1, (255, 255, 255), 3)
        return largest_rectangle
    else:
        print("No field found")
        return None

def detect_table_tennis_balls_and_robots():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    robots_list = []  
    field_coords = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_width = frame.shape[1]
        field_contour = detect_field(frame)
        if field_contour is not None:
            x, y, w, h = cv2.boundingRect(field_contour)
            rect_bottom_left = (x, y)
            rect_top_right = (x + w, y + h)
            field_coords = field_contour.reshape(-1, 2).tolist()
        else:
            rect_bottom_left = (20, 20)
            rect_top_right = (600, 450)
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_robot = detect_robots(hsv, frame_width)
        if detected_robot is not None:
            print("Robot Detected, drawing contours and text...")
            cv2.drawContours(frame, [detected_robot], -1, (0, 255, 0), 3)
            robot_center = (detected_robot[0][0][0], detected_robot[0][0][1])
            cv2.putText(frame, "Robot", robot_center, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            robots_list.append(robot_center)
        else:
            print("No robot detected.")
        detected_balls = detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right)
       
        ball_positions = []
        for (x, y, radius) in detected_balls:
            center = (int(x), int(y))
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, (center[0], center[1]), 2, (0, 0, 255), 3)
            circle_text = f"({center[0]}, {center[1]}), Radius: {radius}"
            text_position = (center[0] - radius, center[1] + radius + 2)
            cv2.putText(frame, circle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            ball_positions.append((center[0], center[1], radius))
        
        os.makedirs('EV3_MicroPython/data/balls_positions', exist_ok=True)
        with open('EV3_MicroPython/data/balls_positions/balls_positions.json', 'w') as f:
            json.dump(ball_positions, f)

        os.makedirs('EV3_MicroPython/data/robots_positions', exist_ok=True)
        with open('EV3_MicroPython/data/robots_positions/robots_positions.json', 'w') as f:
            json.dump(robots_list, f)
        
        if field_coords:
            os.makedirs('EV3_MicroPython/data/field_positions', exist_ok=True)
            with open('EV3_MicroPython/data/field_positions/field_positions.json', 'w') as f:
                json.dump(field_coords, f)
        
        cv2.imshow("Table Tennis Ball and Robot Detection", frame)
        print("Detected Ball Positions:", ball_positions)
        print("Detected Robot Positions:", robots_list)
        print("Detected Field Coordinates:", field_coords)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_table_tennis_balls_and_robots()
