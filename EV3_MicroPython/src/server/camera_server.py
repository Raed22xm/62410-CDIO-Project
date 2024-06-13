import cv2
import numpy as np
import math
import socket

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(robot_center, ball_center):
    delta_x = ball_center[0] - robot_center[0]
    delta_y = ball_center[1] - robot_center[1]
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
                    detected_balls.append((x, y, radius))
    return detected_balls

def detect_black_and_yellow_robots(frame, rect_bottom_left, rect_top_right, min_robot_area):
    lower_yellow = np.array([20, 50, 200])
    upper_yellow = np.array([40, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 255, 50])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def detect_obstacles(hsv):
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plus_sign_contours = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:  
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 8 and len(approx) <= 14:
            plus_sign_contours.append(approx)

    return plus_sign_contours

def detect_and_communicate():
    cap = cv2.VideoCapture(0)  # Use the correct camera index if you have multiple cameras
    server_address = ('192.168.1.253', 47725)  # Change to a new port number
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Reuse the address
        server_socket.bind(server_address)
        server_socket.listen(1)
        print('Server listening on {}:{}'.format(*server_address))

        connection, client_address = server_socket.accept()
        with connection:
            print('Connected to:', client_address)
            rect_bottom_left = (20, 20)
            rect_top_right = (600, 450)  # Initial values (adjust as needed)
            min_area = 500
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                detected_balls = detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right)
                yellow_robot, black_robot = detect_black_and_yellow_robots(frame, rect_bottom_left, rect_top_right, min_area)
                field_contour = detect_field(frame)
                obstacles = detect_obstacles(hsv)

                for (x, y, radius) in detected_balls:
                    center = (int(x), int(y))
                    cv2.circle(frame, center, radius, (0, 255, 0), 2)
                    cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if yellow_robot is not None:
                    top_left, bottom_right = yellow_robot
                    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                if black_robot is not None:
                    top_left, bottom_right = black_robot
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

                for contour in obstacles:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)

                ball_center = detected_balls[0][:2] if detected_balls else None
                robot_center = yellow_robot[0] if yellow_robot else (black_robot[0] if black_robot else None)

                if ball_center and robot_center:
                    command = '{},{},{},{}'.format(ball_center[0], ball_center[1], robot_center[0], robot_center[1])
                    try:
                        connection.sendall(command.encode())
                        print("Sent command: {}".format(command))
                    except BrokenPipeError:
                        print("Connection closed by client")
                        break

                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_communicate()
