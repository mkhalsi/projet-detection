# Importation des bibliothèques nécessaires
import paho.mqtt.client as mqtt  # Pour la communication MQTT
import datetime  # Pour gérer les dates et heures
import cv2  # Pour la capture et le traitement d'images
import numpy as np  # Pour les opérations numériques
import mysql.connector  # Pour la connexion à la base de données MySQL
import tensorflow as tf  # Pour charger et utiliser le modèle TensorFlow
import os  # Pour manipuler le système de fichiers

# Informations de connexion MQTT
broker = "86459bf9cd6f4ac4b4359d2703047731.s1.eu.hivemq.cloud"  # Adresse du broker MQTT
port = 8883  # Port sécurisé pour la connexion MQTT
topic = "detection_plastic"  # Sujet MQTT pour publier et s'abonner aux messages

# Fonction de callback appelée lors de l'établissement de la connexion MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:  # Vérifie si la connexion a réussi
        print("Connecté avec succès au broker HiveMQ")
        client.subscribe(topic)  # S'abonne au sujet MQTT
    else:
        print(f"Échec de la connexion avec le code {rc}")  # Affiche le code d'erreur en cas de problème

# Fonction de callback appelée lorsqu'un message est reçu
def on_message(client, userdata, msg):
    print(f"Message reçu : {msg.payload.decode()}")  # Affiche le contenu du message reçu
    save_to_mysql(msg.payload.decode())  # Sauvegarde le message dans la base de données MySQL

# Initialisation du client MQTT
client = mqtt.Client()  # Crée une instance du client MQTT
client.on_connect = on_connect  # Définit la fonction de callback pour la connexion
client.on_message = on_message  # Définit la fonction de callback pour les messages

# Connexion au broker MQTT
try:
    client.connect(broker, port, 60)  # Tente de se connecter au broker avec un délai de 60 secondes
    client.loop_start()  # Démarre la boucle d'écoute MQTT en arrière-plan
except Exception as e:
    print(f"Échec de la connexion au broker : {e}")  # Affiche un message d'erreur en cas d'échec
    exit(1)  # Quitte le programme

# Fonction pour détecter les objets en plastique dans une image
def detect_plastic(frame, model, threshold=0.8):
    height, width, _ = frame.shape  # Obtient les dimensions de l'image
    step_size = 50  # Taille du pas pour parcourir l'image
    box_size = 150  # Taille des sous-images à analyser
    plastic_detected = False  # Initialise un indicateur pour savoir si du plastique a été détecté

    # Parcourt l'image par blocs
    for y in range(0, height - box_size, step_size):
        for x in range(0, width - box_size, step_size):
            sub_image = frame[y:y + box_size, x:x + box_size]  # Extrait une sous-image
            sub_image_resized = cv2.resize(sub_image, (150, 150)) / 255.0  # Redimensionne et normalise la sous-image
            sub_image_expanded = np.expand_dims(sub_image_resized, axis=0)  # Ajoute une dimension pour le modèle TensorFlow

            prediction = model.predict(sub_image_expanded, verbose=0)  # Prédiction du modèle
            if prediction.flatten()[0] > threshold:  # Vérifie si la probabilité dépasse le seuil
                # Dessine un rectangle vert autour de la région détectée
                cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (0, 255, 0), 2)
                # Ajoute un texte "Plastic" au-dessus de la région détectée
                cv2.putText(frame, "Plastic", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                plastic_detected = True  # Met à jour l'indicateur

                # Publie un message sur le sujet MQTT
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"Plastic detected at {current_time}"
                client.publish(topic, message)  # Publie le message

    # Si du plastique est détecté, sauvegarde une image annotée
    if plastic_detected:
        os.makedirs("static/images", exist_ok=True)  # Crée le dossier pour les images si nécessaire
        image_path = "static/images/captured_image.jpg"
        cv2.imwrite(image_path, frame)  # Sauvegarde l'image annotée
        print(f"Image capturée à {image_path}")
    else:
        print("Aucun plastique détecté")

    return frame, plastic_detected  # Retourne l'image annotée et l'indicateur

# Connexion à MySQL
conn = mysql.connector.connect(
    host="localhost",  # Hôte de la base de données
    user="root",  # Nom d'utilisateur
    password="",  # Mot de passe
    database="detection_projet"  # Nom de la base de données
)
cursor = conn.cursor()  # Crée un curseur pour exécuter des commandes SQL

# Fonction pour enregistrer des messages dans MySQL
def save_to_mysql(message):
    try:
        cursor.execute("INSERT INTO detections (message, timestamp) VALUES (%s, %s)", 
                       (message, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))  # Insère un message et un horodatage
        conn.commit()  # Valide la transaction
        print("Données enregistrées dans la base de données.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement dans la base de données : {e}")  # Affiche un message d'erreur

# Chargement du modèle de détection
model = tf.keras.models.load_model('plastic_detection_model.h5')  # Charge le modèle pré-entraîné

# Capture de la vidéo depuis la caméra
cap = cv2.VideoCapture(0)  # Initialise la capture vidéo (caméra par défaut)

while True:  # Boucle principale pour le traitement vidéo
    ret, frame = cap.read()  # Lit une image de la caméra
    if not ret:  # Si aucune image n'est capturée, sort de la boucle
        break

    frame, plastic_detected = detect_plastic(frame, model)  # Détecte les objets en plastique dans l'image

    cv2.imshow("Plastic Detection", frame)  # Affiche l'image annotée dans une fenêtre

    # Sauvegarde le message dans MySQL si du plastique est détecté
    if plastic_detected:
        save_to_mysql("Plastic detected")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quitte la boucle si la touche 'q' est pressée
        break

cap.release()  # Libère la ressource de la caméra
cv2.destroyAllWindows()  # Ferme toutes les fenêtres d'affichage

client.loop_stop()  # Arrête la boucle MQTT
conn.close()  # Ferme la connexion à la base de données
