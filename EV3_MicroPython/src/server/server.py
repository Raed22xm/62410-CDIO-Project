import socket
from ev3dev2.motor import EV3Brick, Motor, DriveBase
from  ev3dev2.motor import Port, Direction


HOST = '172.21.229.121'  
PORT = 65432     

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


server_socket.bind((HOST, PORT))


server_socket.listen()

print(f"Server listening on {HOST}:{PORT}")


conn, addr = server_socket.accept()
print(f"Connected by {addr}")


ev3 = EV3Brick()
left_motor = Motor(Port.B, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.C, Direction.COUNTERCLOCKWISE)
gripper_motor = Motor(Port.A)
golfBot = DriveBase(left_motor, right_motor, wheel_diameter=26, axle_track=115)

try:
    while True:
     
        data = conn.recv(1024)
        if not data:
            break
        command = data.decode()
        print(f"Received {command}")


        if command == 'move_forward':
            golfBot.on_for_degrees(left_motor, 100, 100, brake=False)
        elif command == 'move_backward':
            golfBot.on_for_degrees(left_motor, -100, 100, brake=False)
        elif command == 'grip':
            gripper_motor.on_for_seconds(50, 2, brake=True)
        elif command == 'release':
            gripper_motor.on_for_seconds(-50, 2, brake=True)

        conn.sendall(b'Command received')

finally:

    conn.close()
    server_socket.close()
