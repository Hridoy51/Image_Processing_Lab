import cv2
import matplotlib.pyplot as plt
import numpy as np

def decrease_intensity_resolution(image):
    bits = 8
    r, c = image.shape[:2]
    temp_img = image.copy()
    plt.subplot(2, 4, 1)
    plt.imshow(temp_img,cmap='gray')
    plt.title("Original Image")

    for i in range(1,bits):
        temp = 255//(2**(bits-i)-1)
        temp_img = image.copy()
        for j in range(r):
            for k in range(c):
                temp_img[j][k] = round(image[j][k]/temp)*temp
        
        temp_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB)
        plt.subplot(2, 4,i + 1)
        plt.imshow(temp_img)
        plt.title(f"{bits-i}-Bit Resolution")

    

def Main():
    image = cv2.imread("sample.jpeg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(512,512))
    decrease_intensity_resolution(image)
    plt.tight_layout()
    plt.show()

Main()
