import socket

# Define the server address and port
HOST = '192.168.1.100'  # The IP address of your laptop
PORT = 65432            # The same port as used by the server

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((HOST, PORT))

# Send data to the server
client_socket.sendall(b'Hello Laptop!')

# Receive data from the server
data = client_socket.recv(1024)
print(f"Received {data.decode()}")

# Close the connection
client_socket.close()
