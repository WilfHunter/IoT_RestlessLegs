import machine
import struct

class MPU6050:
    def __init__(self, sda_pin=10, scl_pin=11, addr=0x68):
        self.i2c = machine.I2C(0, sda=machine.Pin(sda_pin), scl=machine.Pin(scl_pin), freq=400000)
        self.addr = addr
        
        # Wake up sensor (PWR_MGMT_1 = 0)
        self.i2c.writeto_mem(self.addr, 0x6B, b'\x00')
        
        # FORCE (PWR_MGMT_2 = 0)
        self.i2c.writeto_mem(self.addr, 0x6C, b'\x00')
        
    def read_accel_gyro(self):
        """Reads 6 axes with index"""
        try:
            data = self.i2c.readfrom_mem(self.addr, 0x3B, 14)
            vals = struct.unpack(">hhhhhhh", data)
            return [vals[0]/16384.0, vals[1]/16384.0, vals[2]/16384.0, 
                    vals[4]/131.0,   vals[5]/131.0,   vals[6]/131.0]
        except:
            return [0.0] * 6
