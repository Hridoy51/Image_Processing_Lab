import cv2
import numpy as np
import matplotlib.pyplot as plt
image  = cv2.imread("sample.jpeg")
image = cv2.resize(image,(0,0),fx = 0.5,fy = 0.5)
cv2.imshow("image",image)
gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#Binary_img  = cv2.cvtColor(image,cv2.COLOR_GRAY2)
cv2.imshow("gray",gray_img)
h = gray_img.shape[0]
w = gray_img.shape[1]
for i in range(0,h):
    for j in range(0,w):
        if(gray_img[i][j]>100):
            gray_img[i][j] = 255
        else:
            gray_img[i][j] = 0

 
cv2.imshow("Binary",gray_img)
cv2.waitKey(0)