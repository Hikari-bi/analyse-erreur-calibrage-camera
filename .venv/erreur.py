import numpy as np
import cv2

def calculer_erreur_reprojection():
    try:
        # Charger les paramètres de calibration
        data = np.load("calibration_data.npz")
        mtx = data["mtx"]
        dist = data["dist"]
        rvecs = data["rvecs"]
        tvecs = data["tvecs"]
        objpoints = data["objpoints"]
        imgpoints = data["imgpoints"]
    except Exception as e:
        print(f"Erreur de chargement : {str(e)}")
        return

    # Vérification des données chargées
    num_images = len(objpoints)
    print(f"Nombre total d'images chargées : {num_images}")
    if num_images == 0 or len(imgpoints) == 0:
        print("Aucune donnée de calibration valide")
        return
    if len(imgpoints) != num_images or len(rvecs) != num_images or len(tvecs) != num_images:
        print(f"Incohérence dans les données : objpoints={len(objpoints)}, imgpoints={len(imgpoints)}, rvecs={len(rvecs)}, tvecs={len(tvecs)}")
        return

    total_error = 0.0
    valid_images = 0
    errors_per_image = []  # Stocker les erreurs individuelles

    # Calcul pour toutes les images disponibles
    for i in range(num_images):
        try:
            # Projection des points
            imgpoints_projetes, _ = cv2.projectPoints(
                objpoints[i],
                rvecs[i],
                tvecs[i],
                mtx,
                dist
            )

            # Calcul de l'erreur
            error = cv2.norm(imgpoints[i], imgpoints_projetes, cv2.NORM_L2) / len(imgpoints_projetes)
            total_error += error
            errors_per_image.append(error)
            valid_images += 1

        except Exception as e:
            print(f"Erreur sur l'image {i + 1} : {str(e)}")

    if valid_images > 0:
        erreur_moyenne = total_error / valid_images
        erreur_ecart_type = np.std(errors_per_image) if len(errors_per_image) > 1 else 0.0
        erreur_max = max(errors_per_image) if errors_per_image else 0.0

        print("\n=== RAPPORT DE CALIBRATION ===")
        print(f"Images analysées : {valid_images}/{num_images}")
        print(f"Erreur moyenne de reprojection : {erreur_moyenne:.3f} pixels")
        print(f"Écart-type de l'erreur : {erreur_ecart_type:.3f} pixels")
        print(f"Erreur maximale : {erreur_max:.3f} pixels")
        print("(Une erreur moyenne < 0.5 pixel est idéale)")

        if erreur_moyenne > 1.0:
            print("\n⚠️ Attention : Calibration médiocre !")
            print("Suggestions :")
            if valid_images < 10:
                print("- Ajouter plus d'images (minimum 10 recommandées)")
            if erreur_ecart_type > 0.5:
                print("- Vérifier la consistance des images (angles, éclairage)")
            print("- Vérifier la détection des coins dans toutes les images")
            print("- Utiliser une mire plus nette ou mieux éclairée")
        elif erreur_moyenne > 0.5:
            print("\nℹ️ Calibration acceptable mais perfectible.")
    else:
        print("Aucune image valide analysée")

if __name__ == "__main__":
    calculer_erreur_reprojection()