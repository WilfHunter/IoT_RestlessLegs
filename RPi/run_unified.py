import socket
import struct
import numpy as np
import time
import os
import threading
from datetime import datetime
from collections import deque
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()

# Adjust verbose level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# CONFIG
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
MODEL_PATH = "/home/benoit/Projet/modele.h5"

FS_MPU = 50           # 50 Hz
DUREE_FENETRE = 5     # Time frame = 5s
TAILLE_FENETRE_MPU = FS_MPU * DUREE_FENETRE  
SEUIL_IA_CRISE = 0.50 

# --- Buffer in RAM (THREAD-SAFE) ---
# We keep margins not to lose data
ram_buffer = deque(maxlen=TAILLE_FENETRE_MPU * 2)
lock_buffer = threading.Lock()

def udp_receiver_thread():
    """UDP receiver (in background"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
        
    compteur_total = 0
    
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            if len(data) < 24:
                continue
                
            mpu_bytes = data[:24]
            audio_bytes = data[24:]
            
            ax, ay, az, gx, gy, gz = struct.unpack("6f", mpu_bytes)
            
            # Audio conversion
            audio_tick = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            
            # Writing in buffer
            with lock_buffer:
                ram_buffer.append({
                    "mpu": [ax, ay, az, gx, gy, gz],
                    "audio": audio_tick
                })
                
            compteur_total += 1
                            
        except Exception as e:
            print(f"\nUDP error : {e}")

def extraire_fenetre_ram():
    """Extracts exactly 250 elems from RAM."""
    with lock_buffer:
        # If buffer not ready yet, just wait
        if len(ram_buffer) < TAILLE_FENETRE_MPU:
            return None, None
        
        # We take only the right amount of data
        tout_le_tampon = list(ram_buffer)
        instantane_donnees = tout_le_tampon[-TAILLE_FENETRE_MPU:]
        
    # 1. MPU Tensor -> Shape = (1, 250, 6)
    mpu_data = np.array([item["mpu"] for item in instantane_donnees], dtype=np.float32)
    tensor_mpu = np.expand_dims(mpu_data, axis=0)
    
    # 2. Audio Tensor -> Shape = (1, 200000, 1)
    audio_list = [item["audio"] for item in instantane_donnees]
    audio_data = np.concatenate(audio_list)
    tensor_audio = np.expand_dims(audio_data, axis=0)
    tensor_audio = np.expand_dims(tensor_audio, axis=-1)
    
    return tensor_mpu, tensor_audio

def send_email_alert(subject, body):
    # Getting env data
    sender_email = os.getenv("SMTP_EMAIL_SENDER")
    sender_password = os.getenv("SMTP_PASSWORD")
    receiver_email = os.getenv("EMAIL_RECEIVER")

    # Check we have all we need
    if not all([sender_email, sender_password, receiver_email]):
        print("\nEmail data missing in .env")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            print(f"\nEmail sent to {receiver_email}")
    except Exception as e:
        print(f"\n Email failure : {e}")

# INIT

if not os.path.exists(MODEL_PATH):
    print(f".h5 file missing in: {MODEL_PATH}")
    exit()

model = load_model(MODEL_PATH)

# Start UDP receiving
thread_udp = threading.Thread(target=udp_receiver_thread, daemon=True)
thread_udp.start()

print("\nMonitoring ON")
print("-" * 60)

message_sent = False

try:
    while True:
        start_time = time.time()
        
        tensor_mpu, tensor_audio = extraire_fenetre_ram()
        
        if tensor_mpu is not None and tensor_audio is not None:
            # Inference
            prediction = model({"MPU_Input": tensor_mpu, "Audio_Input": tensor_audio}, training=False)
            score_crise = float(prediction[0][0])
            
            timestamp_actuel = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Alerting
            if score_crise > SEUIL_IA_CRISE:
                print(f"[{timestamp_actuel}] [WARNING] Crisis detected! Probability: {score_crise*100:.2f}%")
                if not message_sent:
                    send_email_alert("Restless Legs Alert", f"Watch out for your legs!")
                    message_sent = True
            else:
                print(f"[{timestamp_actuel}] [Normal] IA probability : {score_crise*100:.2f}%")
                message_sent = False
        else:
            print("Buffering, please wait (approx. 5 sec", end="\r")
                
        # Eval once per second
        elapsed = time.time() - start_time
        time.sleep(max(0.1, 1.0 - elapsed))

except KeyboardInterrupt:
    print("\nMonitoring stopped")
