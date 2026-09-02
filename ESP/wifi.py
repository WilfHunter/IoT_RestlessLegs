import network
import time

def load_env(filename=".env"):
    """Loads .env."""
    env = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Ignorer lignes vides et commentaires
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip().strip('"').strip("'")

            env[key] = value

    return env

def connect():
    env = load_env()
    SSID = env["MY_SSID"]
    PASSWORD = env["MY_PASSWD"]
    
    wlan = network.WLAN(network.STA_IF)

    wlan.active(True)

    if wlan.isconnected():
        return wlan

    print("Connexion au Wi-Fi...")

    wlan.connect(SSID, PASSWORD)

    while not wlan.isconnected():
        time.sleep(0.5)
        print(".", end="")

    print()
    print("Connecté !")

    print("Adresse IP :", wlan.ifconfig()[0])

    return wlan
