import cv2
import matplotlib.pyplot as plt
import numpy as np
def plotimg(image,x,y,z,st):
    im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(x,y,z)
    plt.imshow(im)
    plt.title(st)
def power_low(image,gamma):
    im = image.copy()
    r,c = im.shape[:2]
    for i in range(r):
        for j in range(c):
            im[i][j] = ((im[i][j]/255)**gamma)*255
    return im   
def log_f(image):
    cx = 255/np.log(1+np.max(image))
    im1 = cx*np.log(1+image)
    im1 = np.array(im1,dtype=np.uint8)
    return im1  

def inverse_log(image):
    cx = 255/np.log(1+np.max(image))
    inverse_img = np.exp(image**(1/cx))-1
    inverse_img = np.array(inverse_img,dtype=np.uint8)
    return inverse_img

def thresolding(image):
    r= image.shape[0]
    c= image.shape[1]
    im = image.copy()
    for i in range(0,r):
        for j in range(0,c):
            if image[i][j]>5:
                im[i][j] = 255
            else:
                im[i][j] = 0
    return im
def Main():
    image = cv2.imread("sample4.jpg",0)
    image = cv2.resize(image,(512,512))
    plotimg(image,1,3,1,"Original image")
   
    power_img = power_low(image,2)
    plotimg(power_img,1,3,2,"Power image")
    inverse_img = inverse_log(image)
    inverse_img = thresolding(inverse_img)
    plotimg(inverse_img,1,3,3,"log image")
    
    
    plt.show()
    

Main()
