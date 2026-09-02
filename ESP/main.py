import time
import MPU6050
import INMP441
import transfer
import gc

def load_env(filename=".env"):
    """Loads .env."""
    env = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Ignore comments and empty lines
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip().strip('"').strip("'")

            env[key] = value

    return env


def main():
    env = load_env()
    TARGET_HOST = env["RPI_IP"]    
    # Sensors init
    try:
        mpu = MPU6050.MPU6050(sda_pin=10, scl_pin=11)
        print("MPU6050 OK")
    except Exception as e:
        print("MPU6050 failed to initialize:", e)
        return

    try:
        mic = INMP441.INMP441(sd=15, ws=16, sck=17)
        print("INMP441 OK")
    except Exception as e:
        print("INMP441 failed to initialize:", e)
        return

    try:
        transmitter = transfer.UDPTransmitter(host=TARGET_HOST, port=5005)
        print("UDP initialized on port 5005")
    except Exception as e:
        print("UDP init failed:", e)
        return
    
    gc.collect()
    print("\nSending data. Ctrl+C to stop.\n")
    
    compteur_trames = 0
    
    while True:
        start_time = time.ticks_ms()
        
        audio_chunk = b""
        mpu_features = [0.0] * 6
        
        # Audio reading
        try:
            audio_chunk = mic.read_chunk()
        except Exception as e:
            print("INMP441 reading error:", e)
            
        # MPU6050 reading
        try:
            mpu_features = mpu.read_accel_gyro()
        except Exception as e:
            print("MPU6050 reading error:", e)
        
        # 3. Transfer if valid chunks
        if audio_chunk and len(audio_chunk) > 0:
            try:
                transmitter.send_stream_tick(mpu_features, audio_chunk)
                compteur_trames += 1
                if compteur_trames % 50 == 0:
                    print(f"{compteur_trames} chunks sent...", end="\r")
            except Exception as e:
                print("UDP transfer failure:", e)
        else:
            
            time.sleep_ms(2)
            
        # 50Hz = 20ms for one cycle
        sleep_time = 20 - time.ticks_diff(time.ticks_ms(), start_time)
        if sleep_time > 0:
            time.sleep_ms(sleep_time)

if __name__ == "__main__":
    main()
