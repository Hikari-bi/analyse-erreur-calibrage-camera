import cv2
import os

# Configuration du dossier de sauvegarde
save_dir = "images"
abs_save_dir = os.path.abspath(save_dir)  # Chemin absolu pour le débogage

# Création du dossier avec vérification
try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Dossier de sauvegarde : {abs_save_dir}")
except PermissionError:
    print(f"❌ Erreur : Permission refusée pour créer le dossier {abs_save_dir}")
    exit()
except Exception as e:
    print(f"❌ Erreur création dossier : {str(e)}")
    exit()

# Initialisation de la caméra
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Erreur : Caméra introuvable ou déjà utilisée")
    exit()

i = 0
print("\nContrôles :")
print("- 's' : Sauvegarder l'image")
print("- 'q' : Quitter\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("❌ Avertissement : Image vide reçue")
        continue

    cv2.imshow('Capture de la mire', frame)

    key = cv2.waitKey(300)  # Augmenter le délai pour une meilleure détection

    if key == ord('s'):
        filename = os.path.join(save_dir, f"mire_{i}.jpg")

        # Vérifier si l'image est valide
        if frame is None or frame.size == 0:
            print("❌ Image invalide, impossible de sauvegarder")
            continue

        # Tentative de sauvegarde avec vérification
        try:
            success = cv2.imwrite(filename, frame)
            if success:
                print(f"✅ Image {i} sauvegardée : {os.path.abspath(filename)}")
                i += 1
            else:
                print(f"❌ Échec de la sauvegarde : Format non supporté ou chemin invalide")
        except Exception as e:
            print(f"❌ Erreur critique pendant la sauvegarde : {str(e)}")

    elif key == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
print("\n✅ Programme terminé. Images sauvegardées dans :", abs_save_dir)