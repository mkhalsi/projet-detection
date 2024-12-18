# Importation des modules nécessaires
from flask import Flask, render_template, request, redirect, url_for  # Flask pour le serveur web et gestion des routes
import subprocess  # Pour exécuter des scripts Python externes
import mysql.connector  # Pour la connexion à la base de données MySQL
import datetime  # Pour travailler avec les dates et heures

# Création de l'application Flask
app = Flask(__name__)

# Configuration de la connexion à la base de données MySQL
DB_CONFIG = {
    'host': 'localhost',  # Adresse du serveur MySQL (ici en local)
    'user': 'root',  # Nom d'utilisateur de la base de données
    'password': '',  # Mot de passe de l'utilisateur MySQL
    'database': 'detection_projet'  # Nom de la base de données utilisée
}

# Fonction pour récupérer les 5 dernières détections
def get_detections():
    conn = mysql.connector.connect(**DB_CONFIG)  # Connexion à la base de données avec la configuration définie
    cursor = conn.cursor(dictionary=True)  # Utiliser un curseur pour récupérer les résultats sous forme de dictionnaires
    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 5")  # Requête SQL pour récupérer les 5 dernières détections
    detections = cursor.fetchall()  # Obtenir les résultats de la requête
    conn.close()  # Fermer la connexion à la base de données
    return detections  # Retourner les détections sous forme de liste de dictionnaires

# Fonction pour enregistrer une détection dans la base de données
def save_to_mysql(message):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)  # Connexion à la base de données
        cursor = conn.cursor()  # Création d'un curseur pour exécuter des commandes SQL
        cursor.execute("INSERT INTO detections (message, timestamp) VALUES (%s, %s)", 
                       (message, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))  # Insertion d'une nouvelle détection avec message et horodatage
        conn.commit()  # Valider les changements dans la base de données
        conn.close()  # Fermer la connexion à la base
        print("Données enregistrées dans la base de données.")  # Message de confirmation
    except Exception as e:
        print(f"Erreur lors de l'enregistrement dans la base de données : {e}")  # Afficher un message d'erreur en cas de problème

# Route principale
@app.route('/')
def index():
    detections = get_detections()  # Appeler la fonction pour obtenir les 5 dernières détections
    return render_template('index.html', detections=detections)  # Rendre la page HTML avec les données des détections

# Route pour démarrer la détection
@app.route('/start_detection', methods=['POST'])  # Définir une route pour démarrer la détection via une requête POST
def start_detection():
    try:
        # Exécuter le script de détection (par exemple, un script utilisant MQTT)
        subprocess.run(['python', 'detection_mqtt.py'], check=True)  # Lancer le script en sous-processus

        # Enregistrer un message de détection dans la base de données
        save_to_mysql("Plastic detected")  # Sauvegarder un message dans la base (modifiable selon les résultats)

    except Exception as e:
        print(f"Erreur lors de l'exécution du script de détection : {e}")  # Gérer les erreurs d'exécution

    return redirect(url_for('index'))  # Rediriger vers la route principale pour rafraîchir l'affichage des détections

# Point d'entrée de l'application
if __name__ == '__main__':
    app.run(debug=True)  # Lancer l'application Flask en mode debug pour faciliter le développement
