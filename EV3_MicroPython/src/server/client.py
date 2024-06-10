import socket


HOST = '192.168.1.100'  
PORT = 65432            


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


client_socket.connect((HOST, PORT))


client_socket.sendall(b'Hello Laptop!')


data = client_socket.recv(1024)
print(f"Received {data.decode()}")


client_socket.close()
