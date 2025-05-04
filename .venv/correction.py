import cv2
import numpy as np
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ----------------------------------------------------------
# PARAMÈTRES À CONFIGURER
# ----------------------------------------------------------
FICHIER_CALIB = "calibration_data.npz"
DOSSIER_ENTREE = "images/"
DOSSIER_SORTIE = "images_corrigees/"
ALPHA = 0.0  # 0 = bordures noires, 1 = crop maximal
VISUALISATION = True  # Afficher une comparaison
MAX_WORKERS = 5  # Nombre de threads pour le traitement parallèle


# ----------------------------------------------------------
# ÉTAPE 1 : Initialisation
# ----------------------------------------------------------
def charger_calibration(fichier_calib):
    """Charge les paramètres de calibration depuis un fichier NPZ"""
    try:
        data = np.load(fichier_calib)
        mtx = data["mtx"]
        dist = data["dist"]
        print("Paramètres de calibration chargés avec succès.")
        print(f"Matrice de caméra :\n{mtx}")
        print(f"Coefficients de distorsion :\n{dist}")
        return mtx, dist
    except Exception as e:
        print(f"ERREUR : Impossible de charger les données de calibration - {str(e)}")
        exit(1)


# ----------------------------------------------------------
# ÉTAPE 2 : Traitement d'une image
# ----------------------------------------------------------
def corriger_image(args):
    """Corrige la distorsion d'une image et enregistre le résultat"""
    idx, fname, mtx, dist, alpha, dossier_sortie, visualisation = args

    try:
        # Charger l'image
        img = cv2.imread(fname)
        if img is None:
            return f"  ERREUR : Échec du chargement de {os.path.basename(fname)}"

        h, w = img.shape[:2]

        # Calculer la nouvelle matrice
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

        # Corriger la distorsion
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Recadrer l'image
        if alpha != 0 and roi != (0, 0, 0, 0):
            x, y, w_roi, h_roi = roi
            dst = dst[y:y + h_roi, x:x + w_roi]

        # Générer un nom de fichier de sortie
        base_name = Path(fname).stem
        extension = Path(fname).suffix
        sortie_path = os.path.join(dossier_sortie, f"corrigee_{base_name}{extension}")

        # Sauvegarder
        cv2.imwrite(sortie_path, dst)

        # Créer une visualisation côte à côte
        if visualisation:
            vis_path = os.path.join(dossier_sortie, f"comparaison_{base_name}{extension}")
            # Redimensionner si nécessaire pour garder une taille raisonnable
            max_width = 800
            if w > max_width:
                scale = max_width / (w * 2)  # Pour deux images côte à côte
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                dst = cv2.resize(dst, (0, 0), fx=scale, fy=scale)

            # Ajouter des étiquettes
            img_h, img_w = img.shape[:2]
            cv2.putText(img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(dst, "Corrigee", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Créer l'image côte à côte
            vis = np.hstack((img, dst))
            cv2.imwrite(vis_path, vis)

        return f"  Image traitée : {os.path.basename(fname)}"

    except Exception as e:
        return f"  ERREUR sur {os.path.basename(fname)} : {str(e)}"


# ----------------------------------------------------------
# FONCTION PRINCIPALE
# ----------------------------------------------------------
def main():
    # Créer le dossier de sortie
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    # Charger les paramètres de calibration
    mtx, dist = charger_calibration(FICHIER_CALIB)

    # Trouver toutes les images
    images = glob.glob(os.path.join(DOSSIER_ENTREE, "*.jpg")) + \
             glob.glob(os.path.join(DOSSIER_ENTREE, "*.jpeg")) + \
             glob.glob(os.path.join(DOSSIER_ENTREE, "*.png"))

    if not images:
        print(f"ERREUR : Aucune image trouvée dans {DOSSIER_ENTREE} !")
        return

    print(f"{len(images)} images à corriger...")

    # Préparer les arguments pour le traitement parallèle
    args_list = [(idx, fname, mtx, dist, ALPHA, DOSSIER_SORTIE, VISUALISATION)
                 for idx, fname in enumerate(images)]

    # Mesurer le temps d'exécution
    debut = time.time()

    # Traitement parallèle des images
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        resultats = list(executor.map(corriger_image, args_list))

    # Afficher les résultats
    for resultat in resultats:
        print(resultat)

    duree = time.time() - debut
    print(f"\nCorrection terminée en {duree:.2f} secondes !")

    # Calculer les statistiques
    succes = sum(1 for r in resultats if not r.startswith("  ERREUR"))
    echecs = len(resultats) - succes
    print(f"Images traitées avec succès : {succes}/{len(resultats)} ({succes / len(resultats) * 100:.1f}%)")
    if echecs > 0:
        print(f"Échecs : {echecs}")


if __name__ == "__main__":
    main()