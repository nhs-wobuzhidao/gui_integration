import socket

# ESP32_IP = '192.168.0.135'
ESP32_IP = '192.168.0.204'

PORT = 80

def connect_to_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ESP32_IP, PORT))
        print("Connected Wrench to server.")
        
        try:
            while True:
                data = s.recv(1024)
                print(data.decode(), end='')
        except KeyboardInterrupt:
            print("Disconnected from server.")

if __name__ == '__main__':
    connect_to_server()