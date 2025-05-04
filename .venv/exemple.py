import cv2
import numpy as np
import glob
import os
import pickle  # Pour sauvegarder et charger les paramètres

# === 1. Capture des images avec angles variés ===
def capture_images():
    save_dir = "images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Erreur : Impossible d'ouvrir la caméra.")
        return

    i = 0
    print("🎥 Capture en cours... Appuyez sur 's' pour sauvegarder, 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur : Impossible de lire l'image.")
            break

        cv2.imshow('Capture de la mire', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename = os.path.join(save_dir, f"mire_{i}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✅ Image {i} sauvegardée sous {filename}")
            i += 1
        elif key == ord('q'):
            print("🛑 Capture terminée.")
            break

    cap.release()
    cv2.destroyAllWindows()

# === 2. Détection des coins avec visualisation ===
def detect_corners(nb_colonnes=7, nb_lignes=6):
    images = glob.glob("images/*.jpg")
    objpoints, imgpoints = [], []

    if not images:
        print("❌ Aucune image trouvée pour la détection des coins.")
        return objpoints, imgpoints

    objp = np.zeros((nb_lignes * nb_colonnes, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nb_colonnes, 0:nb_lignes].T.reshape(-1, 2)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection des coins
        ret, corners = cv2.findChessboardCorners(
            gray, (nb_colonnes, nb_lignes),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (nb_colonnes, nb_lignes), corners, ret)
            cv2.imshow('Détection des coins', img)
            cv2.waitKey(500)
        else:
            print(f"⚠ Aucune mire détectée dans {fname}. Vérifiez l'angle et l'éclairage.")

    cv2.destroyAllWindows()
    return objpoints, imgpoints

# === 3. Calibration de la caméra ===
def calibrate_camera(objpoints, imgpoints):
    if not objpoints or not imgpoints:
        print("❌ Calibration impossible, aucun point détecté.")
        return None, None

    img_shape = cv2.imread(glob.glob("images/*.jpg")[0]).shape[:2]

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape[::-1], None, None)

    if ret:
        print("\n🎯 Calibration réussie !")
        print("📷 Matrice de la caméra :\n", mtx)
        print("🔧 Coefficients de distorsion :\n", dist)

        # Sauvegarde des paramètres
        with open("calibration_data.pkl", "wb") as f:
            pickle.dump({"mtx": mtx, "dist": dist}, f)

        return mtx, dist
    else:
        print("❌ Erreur lors de la calibration.")
        return None, None

# === 4. Correction des images ===
def correct_images(mtx, dist):
    images = glob.glob("images/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        # Correction
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        corrected_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

        # Sauvegarde de l'image corrigée
        output_path = fname.replace("images", "corrected_images")
        if not os.path.exists("corrected_images"):
            os.makedirs("corrected_images")
        cv2.imwrite(output_path, corrected_img)
        print(f"✅ Image corrigée sauvegardée sous {output_path}")

# === 5. Analyse des erreurs ===
def compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    print("\n📉 Erreur de reprojection moyenne :", total_error / len(objpoints))

# === Exécution du pipeline ===
if __name__ == "__main__":
    capture_images()  # Étape 1 : Capture des images
    objpoints, imgpoints = detect_corners()  # Étape 2 : Détection des coins
    mtx, dist = calibrate_camera(objpoints, imgpoints)  # Étape 3 : Calibration
    if mtx is not None and dist is not None:
        correct_images(mtx, dist)  # Étape 4 : Correction des images
