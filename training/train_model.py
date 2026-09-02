import sqlite3
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


# 1. CONFIG

DB_PATH = "./data/jambes_sans_repos_continu.db"

FS_MPU = 50           # Sampling rate MPU6050 (50 Hz)
DUREE_FENETRE = 5     # Analysis frame of 5 seconds
TAILLE_FENETRE = FS_MPU * DUREE_FENETRE  # Number of points in the analysis timeframe

# Deap Learning model parameters
CONFIG = {
    "mpu_filters": (32, 64, 128),
    "mpu_kernels": (5, 3, 3),
    "audio_filters": (16, 32, 64),
    "dense_layers": (128, 64),
    "dropout": 0.4,
    "lr": 0.0005,
    "batch_size": 32,
    "epochs": 20
}

print("Loading Data from SQLite database...")
conn = sqlite3.connect(DB_PATH)

# Extraction of data
query = "SELECT session_type, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, audio_pcm_50ms FROM flux_brut ORDER BY id ASC"
df = pd.read_sql_query(query, conn)
conn.close()

print(f"{len(df)} 20ms samples succesfully loaded")


# 2. Separation of data : MPU on one side, audio on the other side

X_mpu_raw = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values

audio_list = []
for raw_blob in df['audio_pcm_50ms']:
    audio_tick = np.frombuffer(raw_blob, dtype=np.int32).astype(np.float32) / 2147483648.0
    audio_list.append(audio_tick)

X_audio_raw = np.concatenate(audio_list)
samples_per_tick = len(audio_list[0])

# Change target label into bool (1 = crise, 0 = normal) (training data generated in french...)
labels_raw = (df['session_type'] == 'crise').astype(int).values

# 3. Creation of sliding timeframes

step = TAILLE_FENETRE // 2 

X_mpu_windows = []
X_audio_windows = []
Y_labels = []

for i in range(0, len(df) - TAILLE_FENETRE, step):
    mpu_win = X_mpu_raw[i:i+TAILLE_FENETRE]
    
    audio_start_idx = i * samples_per_tick
    audio_end_idx = audio_start_idx + (TAILLE_FENETRE * samples_per_tick)
    audio_win = X_audio_raw[audio_start_idx:audio_end_idx]
    
    win_label = 1 if np.mean(labels_raw[i:i+TAILLE_FENETRE]) > 0.5 else 0
    
    X_mpu_windows.append(mpu_win)
    X_audio_windows.append(audio_win)
    Y_labels.append(win_label)

X_mpu = np.array(X_mpu_windows)
X_audio = np.array(X_audio_windows)
X_audio = np.expand_dims(X_audio, axis=-1)  
Y = np.array(Y_labels)

# Train / Test (80% / 20%)
X_mpu_train, X_mpu_test, X_audio_train, X_audio_test, Y_train, Y_test = train_test_split(
    X_mpu, X_audio, Y, test_size=0.2, random_state=42, stratify=Y
)

# 4. Setup of DL model

# --- Branch 1 : MPU6050  ---
input_mpu = layers.Input(shape=(TAILLE_FENETRE, 6), name="MPU_Input")
x = input_mpu
for filters, kernel in zip(CONFIG['mpu_filters'], CONFIG['mpu_kernels']):
    x = layers.Conv1D(filters, kernel_size=kernel, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.GlobalAveragePooling1D()(x)

# --- Branch 2 : INMP441 (Audio) ---
input_audio = layers.Input(shape=(X_audio.shape[1], 1), name="Audio_Input")
y = input_audio

y = layers.Conv1D(CONFIG['audio_filters'][0], kernel_size=64, strides=4, activation='relu')(y)
y = layers.MaxPooling1D(pool_size=4)(y)

for filters in CONFIG['audio_filters'][1:]:
    y = layers.Conv1D(filters, kernel_size=32, strides=2, activation='relu')(y)
    y = layers.MaxPooling1D(pool_size=2)(y)
y = layers.GlobalAveragePooling1D()(y)

# --- Merging of des Branches ---
combined = layers.concatenate([x, y])

# --- Last decision layers ---
z = combined
for dense_units in CONFIG['dense_layers']:
    z = layers.Dense(dense_units, activation='relu')(z)
    z = layers.Dropout(CONFIG['dropout'])(z)
    
output = layers.Dense(1, activation='sigmoid', name="Output")(z)

model = models.Model(inputs=[input_mpu, input_audio], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['lr']),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()


# 5. Model Training

start_time = time.time()

history = model.fit(
    x={"MPU_Input": X_mpu_train, "Audio_Input": X_audio_train},
    y=Y_train,
    validation_data=({"MPU_Input": X_mpu_test, "Audio_Input": X_audio_test}, Y_test),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    verbose=1
)

duration = time.time() - start_time
print(f"Training done in {duration:.1f} seconds.")

# 6. Saving in h5 format

output_filename = "modele.h5"
print(f"Savingin H5 format: {output_filename}...")

# Sauvegarde des poids et de la structure dans un fichier unique .h5
model.save(output_filename, save_format='h5')

print("All done. You can copy h5 file on your raspberry (or whatever you use)")
