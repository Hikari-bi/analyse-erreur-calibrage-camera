import cv2
import numpy as np
import glob

# Récupérer la liste de toutes les images .jpg dans le dossier "images"
image_files = glob.glob("images/*.jpg")
if not image_files:
    print("Aucune image trouvée dans le dossier 'images'.")
    exit()

# Paramètres du damier
pattern_size = (7, 7)    # 7x7 coins internes
taille_case = 25.0       # Taille d'une case (mm)

# Générer les points 3D réels du damier
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= taille_case

# Matrice de la caméra et coefficients de distorsion
K = np.array([
    [696.98874959,   0.        , 321.77940686],
    [  0.        , 693.8805463 , 218.05151322],
    [  0.        ,   0.        ,   1.        ]
])
dist = np.array([0.03353864, -0.33479707, 0.00468428, -0.01085769, 0.60319474])

# Définir l'axe 3D à projeter pour la visualisation (longueur = 5 unités)
axis = np.float32([[5, 0, 0],
                   [0, 5, 0],
                   [0, 0, -5]]).reshape(-1, 3) * taille_case  # Ajuster la longueur des axes

# Boucle sur toutes les images
for image_file in image_files:
    image = cv2.imread(image_file)
    if image is None:
        print(f"Erreur : Impossible de charger l'image '{image_file}'.")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les coins du damier
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        print(f"Erreur : Damier 7x7 non détecté dans l'image '{image_file}'.")
        continue

    # Affiner la détection des coins
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Estimer la pose avec solvePnP
    success, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)
    if not success:
        print(f"Erreur : Échec de solvePnP pour l'image '{image_file}'.")
        continue

    # Convertir le vecteur de rotation en matrice de rotation
    R, _ = cv2.Rodrigues(rvec)

    # Calculer la position de la caméra dans le repère du damier
    # Position de la caméra = -R^T * tvec
    camera_position = -np.dot(R.T, tvec)
    print(f"=== Pose Estimation pour {image_file} ===")
    print("Position de la caméra (x, y, z) dans le repère du damier (mm):\n", camera_position.ravel())
    print("Matrice de rotation (orientation) :\n", R)

    # Projeter l'axe 3D sur l'image
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)

    # Dessiner les axes à partir du premier coin détecté
    corner = tuple(corners2[0].ravel().astype(int))
    image = cv2.line(image, corner, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)  # Axe X en rouge
    image = cv2.line(image, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)  # Axe Y en vert
    image = cv2.line(image, corner, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)  # Axe Z en bleu

    # Dessiner tous les coins détectés en jaune
    for c in corners2:
        pt = tuple(c.ravel().astype(int))
        image = cv2.circle(image, pt, 4, (0, 255, 255), -1)

    # Afficher le résultat
    cv2.imshow("Pose Estimation - " + image_file, image)
    cv2.waitKey(1000)  # Augmenter à 10 secondes pour mieux voir

cv2.destroyAllWindows()