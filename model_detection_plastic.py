import numpy as np  # Importation de la bibliothèque NumPy pour manipuler des tableaux multidimensionnels.
import os  # Importation de la bibliothèque os pour interagir avec le système de fichiers.
import cv2  # Importation de la bibliothèque OpenCV pour le traitement d'images.
import matplotlib.pyplot as plt  # Importation de Matplotlib pour tracer des graphiques.
from sklearn.model_selection import train_test_split  # Pour diviser les données en ensembles d'entraînement et de validation.
from tensorflow.keras.models import Sequential  # Pour créer un modèle séquentiel de réseau de neurones.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Importation des couches nécessaires au CNN.
from tensorflow.keras.optimizers import Adam  # Importation de l'optimiseur Adam.
import sys  # Importation de la bibliothèque sys pour la gestion des flux d'entrée/sortie.

# Rediriger stdout et stderr vers des fichiers pour capturer les logs et erreurs.
sys.stdout = open('output.log', 'w', encoding='utf-8')
sys.stderr = open('error.log', 'w', encoding='utf-8')

# Chemins des dossiers contenant les images des classes plastique et non-plastique.
plastique_path = 'C:/Users/SIGMA ITD/Downloads/Garbage classification/plastique'
non_plastique_path = 'C:/Users/SIGMA ITD/Downloads/Garbage classification/non_plastique'

# Fonction pour charger et prétraiter les images à partir d'un dossier donné.
def load_and_preprocess_images(folder_path, label, image_size=(150, 150)):
    images = []  # Liste pour stocker les images.
    labels = []  # Liste pour stocker les étiquettes associées.
    for filename in os.listdir(folder_path):  # Parcourir tous les fichiers dans le dossier.
        img_path = os.path.join(folder_path, filename)  # Construire le chemin complet de l'image.
        img = cv2.imread(img_path)  # Charger l'image avec OpenCV.
        
        # Vérifier si l'image a été chargée correctement.
        if img is not None:
            # Redimensionner l'image à la taille spécifiée et normaliser les pixels (valeurs entre 0 et 1).
            img_resized = cv2.resize(img, image_size) / 255.0
            images.append(img_resized)  # Ajouter l'image prétraitée à la liste.
            labels.append(label)  # Ajouter l'étiquette associée à la liste.
            
    return np.array(images), np.array(labels)  # Retourner les images et leurs étiquettes sous forme de tableaux NumPy.

# Charger et prétraiter les images pour la classe plastique (étiquette 1).
plastique_images, plastique_labels = load_and_preprocess_images(plastique_path, 1)

# Charger et prétraiter les images pour la classe non-plastique (étiquette 0).
non_plastique_images, non_plastique_labels = load_and_preprocess_images(non_plastique_path, 0)

# Combiner toutes les images et étiquettes dans des tableaux uniques.
all_images = np.concatenate((plastique_images, non_plastique_images), axis=0)  # Combiner les images.
all_labels = np.concatenate((plastique_labels, non_plastique_labels), axis=0)  # Combiner les étiquettes.

# Diviser les données en ensembles d'entraînement et de validation (80% entraînement, 20% validation).
X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Définir et construire un modèle CNN séquentiel.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # Première couche de convolution avec 32 filtres.
model.add(MaxPooling2D(pool_size=(2, 2)))  # Couche de pooling pour réduire la taille des caractéristiques.
model.add(Conv2D(64, (3, 3), activation='relu'))  # Deuxième couche de convolution avec 64 filtres.
model.add(MaxPooling2D(pool_size=(2, 2)))  # Deuxième couche de pooling.
model.add(Conv2D(128, (3, 3), activation='relu'))  # Troisième couche de convolution avec 128 filtres.
model.add(MaxPooling2D(pool_size=(2, 2)))  # Troisième couche de pooling.
model.add(Conv2D(256, (3, 3), activation='relu'))  # Quatrième couche de convolution avec 256 filtres.
model.add(MaxPooling2D(pool_size=(2, 2)))  # Quatrième couche de pooling.
model.add(Flatten())  # Aplatir les caractéristiques pour les connecter à des couches entièrement connectées.
model.add(Dense(256, activation='relu'))  # Couche entièrement connectée avec 256 neurones.
model.add(Dropout(0.5))  # Couche de dropout pour réduire le surapprentissage.
model.add(Dense(1, activation='sigmoid'))  # Couche de sortie pour une classification binaire (0 ou 1).

# Compiler le modèle avec l'optimiseur Adam, la perte binaire et la métrique de précision.
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle sur les données d'entraînement et valider sur l'ensemble de validation.
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Sauvegarder le modèle entraîné dans un fichier HDF5.
model.save('plastic_detection_model.h5')

# Tracer les courbes d'apprentissage (précision et perte).
plt.figure(figsize=(12, 4))

# Courbe de précision.
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')  # Précision sur l'ensemble d'entraînement.
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Précision sur l'ensemble de validation.
plt.title('Model Accuracy')  # Titre du graphique.
plt.xlabel('Epochs')  # Axe des abscisses : époques.
plt.ylabel('Accuracy')  # Axe des ordonnées : précision.
plt.legend()  # Ajouter une légende.

# Courbe de perte.
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')  # Perte sur l'ensemble d'entraînement.
plt.plot(history.history['val_loss'], label='Validation Loss')  # Perte sur l'ensemble de validation.
plt.title('Model Loss')  # Titre du graphique.
plt.xlabel('Epochs')  # Axe des abscisses : époques.
plt.ylabel('Loss')  # Axe des ordonnées : perte.
plt.legend()  # Ajouter une légende.

# Afficher les graphiques.
plt.show()
