import glob

import cv2 as cv
import numpy as np

nx = 7
ny = 9

objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob("calibration_images/*.png")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv.namedWindow("img", cv.WINDOW_NORMAL)
        cv.setWindowProperty("img", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("img", img)
        cv.waitKey(100)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Matriz intrínseca da câmera:")
print(mtx)

# params fx, fy, cx, cy
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print(f"{fx}, {fy}, {cx}, {cy}")
