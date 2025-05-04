#this script is a simple client class for rcssserver

import socket

# TESTING PURPOSES
import time

class Client:
    def __init__(self, server='localhost', port=6000):
        self.server = server
        self.port = port
        self.sock = None
        self.create_socket()

    def create_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.sock.bind(('', 0))
        print(f"Connected to {self.server}:{self.port}")

    def send_command(self, message):
        self.sock.sendto(message.encode(), (self.server, self.port))

    def get_sensordata(self):
        try:
            data, addr = self.sock.recvfrom(8192)
            return data.decode(), addr
        except socket.error:
            return None, None
    
    def die(self): #kill off client
        self.sock.close()



#TESTING PURPOSES 
if __name__ == "__main__":
    client = Client(server='localhost', port=6000)
    
    time.sleep(2)
    client.send_command("(init T1)")
    for i in range(100):
        time.sleep(0.1)
        print(client.get_sensordata())
        client.send_command("(turn 5)")
    client.die()
