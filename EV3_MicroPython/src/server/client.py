import socket
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent, MoveTank

def start_client():
    server_address = ('192.168.1.253',47725)  # Change to the new port number

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(10)  # Set a timeout for the socket
    try:
        print("Attempting to connect")
        client_socket.connect(server_address)
        print('Connected to server:', server_address)

        tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)

        while True:
            try:
                data = client_socket.recv(1024).decode()
                print("Received data:", data)
                if ',' in data:
                    ball_x, ball_y, ev3_x, ev3_y = map(int, data.split(','))
                    # Calculate direction and send appropriate command
                    if ball_x > ev3_x:
                        command = "right"
                    elif ball_x < ev3_x:
                        command = "left"
                    else:
                        command = "forward"
                    
                    print("Command: {}".format(command))
                    if command == "forward":
                        tank_drive.on_for_seconds(SpeedPercent(50), SpeedPercent(50), 2)
                    elif command == "backward":
                        tank_drive.on_for_seconds(SpeedPercent(-50), SpeedPercent(-50), 2)
                    elif command == "left":
                        tank_drive.on_for_seconds(SpeedPercent(-50), SpeedPercent(50), 1)
                    elif command == "right":
                        tank_drive.on_for_seconds(SpeedPercent(50), SpeedPercent(-50), 1)

                    client_socket.sendall("Moved {}".format(command).encode())
                elif data == "exit":
                    break
                else:
                    print("Unknown command")
                    client_socket.sendall("Unknown command".encode())
            except socket.timeout:
                print('Connection timed out')
                break
    except Exception as e:
        print('An error occurred:', e)
    finally:
        print('Closing connection')
        client_socket.close()

if __name__ == '__main__':
    start_client()
