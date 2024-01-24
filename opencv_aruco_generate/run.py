import numpy as np
import cv2

size = 250
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_img = cv2.aruco.drawMarker(aruco_dict, 0, size)

cv2.imshow("img", aruco_img)
cv2.waitKey()

cv2.imwrite("img_aruco.png", aruco_img)
