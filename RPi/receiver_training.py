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
    print(f"🗄️ Base de données de flux continu prête.")

init_database()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("📡 Récepteur de flux continu à l'écoute...")

TYPE_DE_SESSION = 'normal' 

# Liste temporaire en mémoire RAM pour stocker le lot de données avant écriture
tampon_donnees = []
compteur_total = 0

try:
    while True:
        data, addr = sock.recvfrom(4096)
        
        if len(data) < 24:
            continue
            
        # Extraction 50Hz
        mpu_bytes = data[:24]
        audio_bytes = data[24:]
        
        ax, ay, az, gx, gy, gz = struct.unpack("6f", mpu_bytes)
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # On ajoute le tuple dans notre tampon en RAM (aucune écriture disque ici)
        tampon_donnees.append((
            timestamp_str, ax, ay, az, gx, gy, gz, bytes(audio_bytes), TYPE_DE_SESSION
        ))
        compteur_total += 1
        
        # Écriture par paquets de 50 dans la DB
        if len(tampon_donnees) >= 50:
            try:
                # On ouvre la connexion uniquement pour cette transaction rapide
                conn = sqlite3.connect(DB_NAME, timeout=20.0)
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.cursor()
                
                # executemany est beaucoup plus rapide qu'une boucle d'inserts
                cursor.executemany("""
                    INSERT INTO flux_brut 
                    (timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, audio_pcm_50ms, session_type) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tampon_donnees)
                
                conn.commit()
                conn.close() # Libère IMMÉDIATEMENT la base pour le script de l'IA
                
                print(f"💾 {compteur_total} échantillons insérés (Session: {TYPE_DE_SESSION}).", end="\r")
                tampon_donnees.clear() # On vide le tampon RAM
                
            except sqlite3.OperationalError as e:
                # En cas de conflit rare, on n'efface pas le tampon, on réessaye au prochain cycle
                print(f"\n⚠️ Base momentanément occupée, attente... ({e})")

except KeyboardInterrupt:
    print("\n🛑 Fermeture propre du flux.")
    # On vide le reste du tampon s'il contient des données avant de couper
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
