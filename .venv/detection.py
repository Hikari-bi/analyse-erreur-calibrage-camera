import cv2
import glob

# Dimensions de la mire (exemple : 7x7cases internes)
nb_colonnes = 7
nb_lignes = 7

# Chargement des images
images = glob.glob("images/*.jpg")

# Initialiser un compteur
compteur = 0

for fname in images:
    if compteur >= 20:  # Arrêter après le nombre d' images
        break

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection des coins de l'échiquier
    ret, corners = cv2.findChessboardCorners(gray, (nb_colonnes, nb_lignes), None)

    if ret:
        compteur += 1  # Incrémenter le compteur seulement si une mire est détectée
        print(f"Image {fname} : Détecté")  # Afficher "Détecté"
        cv2.drawChessboardCorners(img, (nb_colonnes, nb_lignes), corners, ret)
        cv2.imshow('Détection des points', img)
        cv2.waitKey(400)
    else:
        print(f"Image {fname} : Non détecté")  # Afficher "Non détecté"

cv2.destroyAllWindows()
