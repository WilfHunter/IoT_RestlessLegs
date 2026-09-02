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

# Forcer TensorFlow à masquer les messages de log informatifs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
MODEL_PATH = "/home/benoit/Projet/modele.h5"

FS_MPU = 50           # 50 Hz
DUREE_FENETRE = 5     # Fenêtres de 5 secondes
TAILLE_FENETRE_MPU = FS_MPU * DUREE_FENETRE  # 250 points
SEUIL_IA_CRISE = 0.50 

# --- TAMPON EN RAM (THREAD-SAFE) ---
# On garde une marge de sécurité dans maxlen pour ne perdre aucun paquet pendant le traitement
ram_buffer = deque(maxlen=TAILLE_FENETRE_MPU * 2)
lock_buffer = threading.Lock()

def udp_receiver_thread():
    """Fonction exécutée en arrière-plan pour recevoir les paquets UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print("📡 Récepteur de flux continu à l'écoute (Stockage RAM actif)...")
    
    compteur_total = 0
    
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            if len(data) < 24:
                continue
                
            mpu_bytes = data[:24]
            audio_bytes = data[24:]
            
            ax, ay, az, gx, gy, gz = struct.unpack("6f", mpu_bytes)
            
            # Conversion directe des données audio brutes en float32 normalisé
            audio_tick = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            
            # Écriture sécurisée dans le tampon RAM partagé
            with lock_buffer:
                ram_buffer.append({
                    "mpu": [ax, ay, az, gx, gy, gz],
                    "audio": audio_tick
                })
                
            compteur_total += 1
            if compteur_total % 50 == 0:
                print(f"📥 {compteur_total} échantillons reçus en RAM.", end="\r")
                
        except Exception as e:
            print(f"\n⚠️ Erreur récepteur UDP : {e}")

def extraire_fenetre_ram():
    """Extrait et formate STRICTEMENT les 250 derniers éléments depuis la RAM."""
    with lock_buffer:
        # Si nous n'avons pas encore accumulé assez de données, on attend
        if len(ram_buffer) < TAILLE_FENETRE_MPU:
            return None, None
        
        # Sécurisation des dimensions : On extrait TRÈS EXACTEMENT les 250 derniers éléments
        # Cela évite les variations de taille (ex: 260) si l'UDP remplit le buffer en tâche de fond
        tout_le_tampon = list(ram_buffer)
        instantane_donnees = tout_le_tampon[-TAILLE_FENETRE_MPU:]
        
    # 1. Tenseur MPU -> Forme garantie (1, 250, 6)
    mpu_data = np.array([item["mpu"] for item in instantane_donnees], dtype=np.float32)
    tensor_mpu = np.expand_dims(mpu_data, axis=0)
    
    # 2. Tenseur Audio -> Forme garantie (1, 200000, 1)
    audio_list = [item["audio"] for item in instantane_donnees]
    audio_data = np.concatenate(audio_list)
    tensor_audio = np.expand_dims(audio_data, axis=0)
    tensor_audio = np.expand_dims(tensor_audio, axis=-1)
    
    return tensor_mpu, tensor_audio

def send_email_alert(subject, body):
    # Récupération des variables d'environnement
    sender_email = os.getenv("SMTP_EMAIL_SENDER")
    sender_password = os.getenv("SMTP_PASSWORD")
    receiver_email = os.getenv("EMAIL_RECEIVER")

    # Sécurité : Vérifier que toutes les variables sont bien chargées
    if not all([sender_email, sender_password, receiver_email]):
        print("\n❌ Erreur : Variables d'environnement de messagerie manquantes dans le fichier .env")
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
            print(f"\n📧 Alerte email envoyée avec succès à {receiver_email}")
    except Exception as e:
        print(f"\n❌ Échec de l'envoi de l'email : {e}")

# --- INITIALISATION ET CHARGEMENT ---
print("🧠 Chargement du modèle Keras d'origine (.h5)...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Fichier .h5 introuvable à l'emplacement : {MODEL_PATH}")
    exit()

model = load_model(MODEL_PATH)
print("✓ Modèle chargé avec succès !")

# Démarrage du thread de réception UDP
thread_udp = threading.Thread(target=udp_receiver_thread, daemon=True)
thread_udp.start()

print("\n📡 Surveillance Edge AI active en RAM via modèle natif .h5...")
print("-" * 60)

message_sent = False

try:
    while True:
        start_time = time.time()
        
        tensor_mpu, tensor_audio = extraire_fenetre_ram()
        
        if tensor_mpu is not None and tensor_audio is not None:
            # Inférence directe avec Keras
            prediction = model({"MPU_Input": tensor_mpu, "Audio_Input": tensor_audio}, training=False)
            score_crise = float(prediction[0][0])
            
            timestamp_actuel = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Gestion de l'alerte médicale
            if score_crise > SEUIL_IA_CRISE:
                print(f"🚨 [{timestamp_actuel}] [ALERTE] Crise détectée ! Probabilité : {score_crise*100:.2f}%")
                if not message_sent:
                    send_email_alert("Jambes sans repos", f"Tes jambes ! Probabilité : {score_crise*100:.2f}%")
                    message_sent = True
            else:
                print(f"🟢 [{timestamp_actuel}] [Statut] Normal | Score IA : {score_crise*100:.2f}%")
                message_sent = False
        else:
            print("⏳ En attente du remplissage du tampon RAM (besoin de 5 secondes de données)...", end="\r")
                
        # Maintien du rythme d'évaluation d'une fois par seconde
        elapsed = time.time() - start_time
        time.sleep(max(0.1, 1.0 - elapsed))

except KeyboardInterrupt:
    print("\n🛑 Surveillance arrêtée proprement.")
