# IoT_RestlessLegs
Personal project allowing to detect restless legs crisis while working at my desk.

Project uses an ESP32-S3, managing an MPU6050 sensor simply duct taped on my desk. There's also an INMP441 sensor, providing context for the model.
ESP32 sends data to a raspberry pi (could be any computer), which stores the sensor data in RAM and runs a Deep Learning model to decide the probability I'm having a crisis.
Script currently sends an email to warn me to have my legs in check.

Model is trained on a separate machine with better hardware.


