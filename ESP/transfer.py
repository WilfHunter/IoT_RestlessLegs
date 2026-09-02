import socket
import struct

class UDPTransmitter:
    def __init__(self, host="192.168.68.100", port=5005):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_stream_tick(self, mpu_data, audio_bytes):
        """Sends one line of data (24 bytes MPU + Audio)"""
        try:
            # 6 floats MPU (24 bytes) + raw audio (PCM)
            packet = struct.pack("6f", *mpu_data) + audio_bytes
            self.sock.sendto(packet, (self.host, self.port))
        except:
            pass
