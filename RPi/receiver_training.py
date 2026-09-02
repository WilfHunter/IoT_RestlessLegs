import socket
import struct
import sqlite3
from datetime import datetime

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
DB_NAME = "./jambes_sans_repos_training.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    # Active le mode WAL au démarrage
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flux_brut (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            acc_x REAL, acc_y REAL, acc_z REAL,
            gyro_x REAL, gyro_y REAL, gyro_z REAL,
            audio_pcm_50ms BLOB,
            session_type TEXT DEFAULT 'normal'
        )
    """)
    conn.commit()
    conn.close()
    print("DB for training ready.")

init_database()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Flow reciever ready")

TYPE_DE_SESSION = 'normal' 

# Init
tampon_donnees = []
compteur_total = 0

try:
    while True:
        data, addr = sock.recvfrom(4096)
        
        if len(data) < 24:
            continue
            
        # 50Hz
        mpu_bytes = data[:24]
        audio_bytes = data[24:]
        
        ax, ay, az, gx, gy, gz = struct.unpack("6f", mpu_bytes)
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # We put buffer tuple in RAM
        tampon_donnees.append((
            timestamp_str, ax, ay, az, gx, gy, gz, bytes(audio_bytes), TYPE_DE_SESSION
        ))
        compteur_total += 1
        
        # Writing into DB by batch of 250
        if len(tampon_donnees) >= 250:
            try:
                conn = sqlite3.connect(DB_NAME, timeout=20.0)
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.cursor()
                
                cursor.executemany("""
                    INSERT INTO flux_brut 
                    (timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, audio_pcm_50ms, session_type) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tampon_donnees)
                
                conn.commit()
                conn.close() # It's not really necessary anymore, but I release the DB
                
                print(f"{compteur_total} samples inserted (Session: {TYPE_DE_SESSION}).", end="\r")
                tampon_donnees.clear() # Buffer emptying
                
            except sqlite3.OperationalError as e:
                print(f"\nDB busy, waiting... ({e})")

except KeyboardInterrupt:
    print("\n Closing connection")
    # Empty buffer before shutting down
    if tampon_donnees:
        conn = sqlite3.connect(DB_NAME, timeout=10.0)
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO flux_brut 
            (timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, audio_pcm_50ms, session_type) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tampon_donnees)
        conn.commit()
        conn.close()
    sock.close()
