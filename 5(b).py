import cv2
import matplotlib.pyplot as plt
import numpy as np

def thresolding(image):
    im = image.copy()
    r= image.shape[0]
    c= image.shape[1]
    for i in range(0,r):
        for j in range(0,c):
            if image[i][j]>200:
                im[i][j] = 255
            else:
                im[i][j] = 0
    return im
def Erosion(image,mask):
    r,c = image.shape[:2]
    errosion_img = image.copy()
    for i in range(r):
        for j in range(c):
            ct = 0
            for k in range(i-(mask//2),i+(mask//2)+1):
                for l in range(j-(mask//2),j+(mask//2)+1):
                    if(k>=0 and k<r and l>=0 and l<c):
                        if(image[k,l]==255):
                            ct+=1
                            
            if(ct==mask*mask):
                errosion_img[i,j] = 255
            else:
                errosion_img[i,j] = 0

    return errosion_img


def Dilation(image,mask):
    r,c = image.shape[:2]
    dilation_img = image.copy()
    for i in range(r):
        for j in range(c):
            ct = 0
            for k in range(i-(mask//2),i+(mask//2)+1):
                for l in range(j-(mask//2),j+(mask//2)+1):
                    if(k>=0 and k<r and l>=0 and l<c):
                        if(image[k,l]==255):
                            ct+=1
            if(ct>=1):
                dilation_img[i,j] = 255
            else:
                dilation_img[i,j] = 0

    return dilation_img


def plotimg(image,x,y,z,st):
    plt.subplot(x,y,z)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(st)
    
def Main():
    image = cv2.imread("Fig0911(a) Noisy Fingerprint 315x238.TIF")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = thresolding(image)
    plotimg(image,1,3,1,"orginal")
    
    mask = 3
    #openning
    openning_image = Erosion(image,mask)
    oppening_image = Dilation(openning_image,mask)
    
    #closing
    closing_image = Dilation(oppening_image,mask)
    closing_image = Erosion(closing_image,mask)

    plotimg(oppening_image,1,3,2,"Openning")
    plotimg(closing_image,1,3,3,"Closing")

    plt.tight_layout()
    plt.show()
   
    
Main()