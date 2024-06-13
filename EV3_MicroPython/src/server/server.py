import socket

def start_server():
    server_address = ('0.0.0.0',47725)  # Use 0.0.0.0 to listen on all network interfaces

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(server_address)
        server_socket.listen(1)
        print('Server listening on {}:{}'.format(*server_address))

        connection, client_address = server_socket.accept()
        with connection:
            print('Connected to:', client_address)
            while True:
                command = input("Enter command for EV3 (forward/backward/left/right/exit): ").strip()
                connection.sendall(command.encode())
                if command == "exit":
                    break
                data = connection.recv(1024)
                print('Received:', data.decode())

if __name__ == '__main__':
    start_server()
