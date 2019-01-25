import cv2
import numpy as np

img = cv2.imread('ps1-input0.png', cv2.IMREAD_GRAYSCALE)
edged = cv2.Canny(img, 100, 200)
cv2.imshow('img',edged)
cv2.imwrite('ps1_1.png',edged)
cv2.waitKey(0)
