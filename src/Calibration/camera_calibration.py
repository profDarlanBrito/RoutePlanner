import glob

import cv2 as cv
import numpy as np

# Define o número de quadrados internos no alvo de calibração (tabuleiro de xadrez)
nx = 9  # Número de quadrados na horizontal (7 linhas de cantos)
ny = 9  # Número de quadrados na vertical (7 colunas de cantos)

# Prepara os pontos do objeto em 3D como (0,0,0), (1,0,0), (2,0,0), ..., (7,7,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays para armazenar os pontos do objeto e os pontos da imagem de todas as imagens
objpoints = []  # Pontos 3D no mundo real
imgpoints = []  # Pontos 2D no plano da imagem

# Lista de todas as imagens de calibração
images = glob.glob("calibration_images/*.png")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Encontra os cantos do tabuleiro de xadrez
    ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

    # Se encontrado, adiciona os pontos do objeto e da imagem
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Desenha e exibe os cantos do tabuleiro de xadrez
        cv.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv.namedWindow("img", cv.WINDOW_NORMAL)
        cv.setWindowProperty("img", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("img", img)
        cv.waitKey(100)

cv.destroyAllWindows()

# Calibra a câmera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Imprime a matriz intrínseca da câmera (mtx)
print("Matriz intrínseca da câmera:")
print(mtx)

# Parâmetros fx, fy, cx, cy
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print(f"{fx}, {fy}, {cx}, {cy}")
