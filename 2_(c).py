import cv2
import matplotlib.pyplot as plt
import numpy as np
def plotimg(image,x,y,z,st):
    im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(x,y,z)
    plt.imshow(im)
    plt.title(st)

def Last_Three_bits_Img(image):
    im = image.copy()
    im = image & 224
    return im


def Main():
    image = cv2.imread("sample6.jpg",0)
    image = cv2.resize(image,(512,512))
    plotimg(image,1,3,1,"Original image")
    Three_bit = Last_Three_bits_Img(image)
    plotimg(Three_bit,1,3,2,"Three bit image")
    difference_img = image-Three_bit
    plotimg(difference_img,1,3,3,"Differenence")
    

    print(Three_bit)
    plt.show()
    

Main()
