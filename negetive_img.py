import cv2
import numpy as np
import matplotlib.pyplot as plt

def negetive(image):
    h = image.shape[0]
    w = image.shape[1]
    for i in range(0,h):
        for j in range(0,w):
            image[i][j] = 255-image[i][j]
def main():
    
    image  = cv2.imread("sample.jpeg")
    image = cv2.resize(image,(512,512))
    cv2.imshow("image",image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",image)
    negetive(image)
    cv2.imshow("gray",image)
    cv2.waitKey(0)

main()