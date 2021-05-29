#1
#img - numpy array spectrogram

from sys import argv
import cv2 as cv
import numpy as np

img = sp

ret, threshold = cv.threshold(img, 120, 255, cv.THRESH_BINARY)

cv.imshow('threshold', threshold)
cv.imshow('orig', img)

#2

normalizedImg = np.zeros((150, 150))
normalizedImg = cv.normalize(img,  normalizedImg, 0, 255, cv.NORM_MINMAX)
cv.imshow('dst_rt', normalizedImg)

cv.waitKey(0)
cv.destroyAllWindows()