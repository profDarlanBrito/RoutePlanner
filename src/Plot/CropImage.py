import cv2 as cv
import numpy as np

img1 = cv.imread("Old_Build_mesh_bad.png")
img2 = cv.imread("Old_Build_mesh_good.png")

img1 = img1[300:900, 600:1240]
img2 = img2[300:900, 600:1240]

cv.imwrite("Old_Build_mesh_bad_crop.png", img1)
cv.imwrite("Old_Build_mesh_good_crop.png", img2)
