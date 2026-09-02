import sqlite3
import numpy as np
from datetime import datetime
import time
import os

# Forcer TensorFlow à masquer les messages de log informatifs (allège la console)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
DB_PATH = "/media/benoit/PROJECT_DATA/jambes_sans_repos_continu.db"
MODEL_PATH = "/home/benoit/Projet/modele.h5"  # Votre fichier .h5 d'origine

def initialiser_table_ia():
    """Crée la table dédiée aux scores IA si elle n'existe pas encore."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    # Création d'une table séparée pour ne pas impacter vos données capteurs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scores_prediction_ia (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            score_ia REAL NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def sauvegarder_score_ia(score):
    """Insère le score de l'IA avec son horodatage précis."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        # Horodatage au format ISO avec millisecondes
        timestamp_actuel = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        cursor.execute(
            """
            INSERT INTO scores_prediction_ia (timestamp, score_ia)
            VALUES (?, ?)
        """,
            (timestamp_actuel, score),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Erreur lors de l'écriture du score IA en DB : {e}")


# Initialisation de la table au démarrage du script
initialiser_table_ia()

FS_MPU = 50           # 50 Hz
DUREE_FENETRE = 5     # Fenêtres de 5 secondes
TAILLE_FENETRE_MPU = FS_MPU * DUREE_FENETRE  # 250 points
SEUIL_IA_CRISE = 0.50 

print("🧠 Chargement du modèle Keras d'origine (.h5)...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Fichier .h5 introuvable à l'emplacement : {MODEL_PATH}")
    exit()

# Chargement direct du modèle natif Keras
model = load_model(MODEL_PATH)
print("✓ Modèle chargé avec succès sur le Raspberry Pi 4 !")

def extraire_dernieres_5_secondes():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, audio_pcm_50ms 
            FROM flux_brut 
            ORDER BY id DESC LIMIT ?
        """, (TAILLE_FENETRE_MPU,))
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < TAILLE_FENETRE_MPU:
            return None, None
            
        rows.reverse() # Remettre dans l'ordre chronologique
        
        # 1. Tenseur MPU -> Forme (250, 6)
        mpu_data = np.array([r[:6] for r in rows], dtype=np.float32)
        tensor_mpu = np.expand_dims(mpu_data, axis=0) # (1, 250, 6)
        
        # 2. Tenseur Audio -> Forme (1, 200000, 1) pour 5 secondes
        audio_list = []
        for r in rows:
            audio_tick = np.frombuffer(r[6], dtype=np.int32).astype(np.float32) / 2147483648.0
            audio_list.append(audio_tick)
            
        audio_data = np.concatenate(audio_list)
        tensor_audio = np.expand_dims(audio_data, axis=0) # (1, 200000)
        tensor_audio = np.expand_dims(tensor_audio, axis=-1) # (1, 200000, 1)
        
        return tensor_mpu, tensor_audio
    except Exception as e:
        print("⚠️ Erreur extraction DB :", e)
        return None, None

print("\n📡 Surveillance Edge AI active via modèle natif .h5...")
print("-" * 60)

import smtplib
from email.message import EmailMessage

def send_email_alert(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = "benoit.malet@gmail.com"
    msg['To'] = "benoit@malet.be"

    # Exemple avec Gmail (nécessite un "Mot de passe d'application")
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login("benoit.malet@gmail.com", "xhlc onlp cacm ermo")
        smtp.send_message(msg)

message_sent = False
try:
    while True:
        start_time = time.time()
        
        tensor_mpu, tensor_audio = extraire_dernieres_5_secondes()
        
        if tensor_mpu is not None and tensor_audio is not None:
            # Inférence directe avec Keras (training=False pour désactiver le Dropout)
            prediction = model({"MPU_Input": tensor_mpu, "Audio_Input": tensor_audio}, training=False)
            score_crise = float(prediction[0][0])
            sauvegarder_score_ia(score_crise)

            # Gestion de l'alerte médicale
            if score_crise > SEUIL_IA_CRISE:
                print(f"🚨 [ALERTE] Crise détectée ! Probabilité : {score_crise*100:.2f}%")
                if message_sent == False :
                    send_email_alert("Jambes sans repos", f"Tes jambes ! Probabilité : {score_crise*100:.2f}%")
                    message_sent = True
            else:
                print(f"🟢 [Statut] Normal | Score IA : {score_crise*100:.2f}%")
                
        # Calcul de la dérive pour analyser la situation une fois par seconde
        elapsed = time.time() - start_time
        time.sleep(max(0.1, 1.0 - elapsed))

except KeyboardInterrupt:
    print("\n🛑 Surveillance arrêtée.")
