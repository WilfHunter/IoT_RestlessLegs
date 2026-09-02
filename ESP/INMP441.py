import machine

class INMP441:
    def __init__(self, sd=15, ws=16, sck=17, sample_rate=16000):
        self.sample_rate = sample_rate
        # Small buffer minuscule fo 50ms audio (16000 * 0.05 * 4 bytes = 3200 bytes)
        self.chunk_size = int(sample_rate * 0.05) * 4
        self.buffer = bytearray(self.chunk_size)
        
        self.audio_in = machine.I2S(
            0,
            sck=machine.Pin(sck),
            ws=machine.Pin(ws),
            sd=machine.Pin(sd),
            mode=machine.I2S.RX,
            bits=32,
            format=machine.I2S.MONO,
            rate=self.sample_rate,
            ibuf=self.chunk_size * 2
        )

    def read_chunk(self):
        """Reads 50ms"""
        try:
            bytes_read = self.audio_in.readinto(self.buffer)
            if not bytes_read:
                return b""
            return self.buffer[:bytes_read]
        except:
            return b""
