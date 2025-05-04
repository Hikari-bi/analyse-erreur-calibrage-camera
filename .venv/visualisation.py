import cv2
import numpy as np

# Chargez vos données de calibration
calib_data = np.load('calibration_data.npz')
mtx = calib_data['mtx']
dist = calib_data['dist']

# Chargez une image test
img = cv2.imread('images/mire_11.jpg')
h, w = img.shape[:2]

# Calculez la nouvelle matrice de caméra
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Corrigez la distorsion
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Rognez l'image si nécessaire
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Affichez les résultats
cv2.imshow('Original', img)
cv2.imshow('Corrigée', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()