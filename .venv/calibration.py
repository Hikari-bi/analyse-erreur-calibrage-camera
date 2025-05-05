import numpy as np
import cv2
import glob

# Dimensions de la grille du damier (nombre de coins internes)
nb_colonnes = 7 # Nombre de coins internes en largeur
nb_lignes = 7  # Nombre de coins internes en hauteur
taille_case = 25  # Taille d'une case en mm (ajuste selon ton damier)

# Préparation des points 3D (coordonnées du damier en espace réel)
objp = np.zeros((nb_lignes * nb_colonnes, 3), np.float32)
objp[:, :2] = np.mgrid[0:nb_colonnes, 0:nb_lignes].T.reshape(-1, 2) * taille_case

# Listes pour stocker les points 3D et 2D
objpoints = []  # Points 3D du monde réel
imgpoints = []  # Points 2D détectés dans l'image

# Chargement des images
images = glob.glob("images/*.jpg")

if not images:
    print("Aucune image trouvée ! Vérifie le chemin du dossier.")
else:
    print(f"{len(images)} images trouvées pour la calibration.")

# Initialiser un compteur
compteur_images = 0

for fname in images:
    if compteur_images >= 30:  # Arrêter après LE Nombre d'image decide
        break

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection des coins de l'échiquier
    ret, corners = cv2.findChessboardCorners(gray, (nb_colonnes, nb_lignes), None)

    if ret:
        print(f"Image {fname} : Détecté")
        objpoints.append(objp)  # Ajouter les points 3D
        imgpoints.append(corners)  # Ajouter les points 2D
        compteur_images += 1  # Incrémenter le compteur uniquement pour les détections réussies

        # Dessiner et afficher les coins détectés
        cv2.drawChessboardCorners(img, (nb_colonnes, nb_lignes), corners, ret)
        cv2.imshow('Détection des points', img)
        cv2.waitKey(2000)  # Pause pour voir l'affichage
    else:
        print(f"Image {fname} : Non détecté")

cv2.destroyAllWindows()

# Vérifier si on a au moins une image valide
if objpoints and imgpoints:
    # Calibration de la caméra
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Affichage des résultats
    print("Matrice de la caméra :\n", mtx)
    print("Coefficients de distorsion :\n", dist)

    # Sauvegarde des paramètres de calibration
    np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, objpoints=objpoints, imgpoints=imgpoints)
    print("Calibration enregistrée sous 'calibration_data.npz'.")
else:
    print("Échec de la calibration : aucun damier détecté.")
