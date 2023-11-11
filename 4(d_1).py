import cv2
import matplotlib.pyplot as plt
import numpy as np

def thresolding(image):
    im = image.copy()
    r= image.shape[0]
    c= image.shape[1]
    for i in range(0,r):
        for j in range(0,c):
            if image[i][j]>130:
                im[i][j] = 255
            else:
                im[i][j] = 0
    return im
def point_detection(image,mask):
    r,c = image.shape[:2]
    point_img = image.copy()
    point_img = np.int64(point_img)
    for i in range(r):
        for j in range(c):
            sum = 0
            for k in range(i-(mask//2),i+(mask//2)+1):
                for l in range(j-(mask//2),j+(mask//2)+1):
                    if(k==i and l==j):
                        sum += int(int(image[k%r,l%c])*(8))
                    else:
                        sum += int(int(image[k%r,l%c])*(-1))
                
            point_img[i,j] = max(0,min(sum,255))
    return np.uint8(point_img)





def plotimg(image,x,y,z,st):
    plt.subplot(x,y,z)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(st)
    
def Main():
    image = cv2.imread("point.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(512,512))
    plotimg(image,1,3,1,"orginal")
    mask = 3
    image_with_point = point_detection(image,mask)
    plotimg(image_with_point,1,3,2,"point detection")

    image_thresold = thresolding(image_with_point)
    plotimg(image_thresold,1,3,3,"point detection after thresold")
    plt.tight_layout()
    plt.show()
   
    
Main()