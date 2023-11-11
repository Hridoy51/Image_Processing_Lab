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
def line_detection(image,mask):
    sz = 3
    r,c = image.shape[:2]
    line_img = image.copy()
    line_img = np.int64(line_img)
    mask = np.int64(mask)
    for i in range(r):
        for j in range(c):
            sum = 0
            wr = 0
            for k in range(i-(sz//2),i+(sz//2)+1):
                wc = 0
                for l in range(j-(sz//2),j+(sz//2)+1):
                    
                    sum += int(int(image[k%r,l%c])*mask[wr,wc])
                    wc+=1
                wr+=1    
                
            line_img[i,j] = max(0,min(sum,255))
    return np.uint8(line_img)





def plotimg(image,x,y,z,st):
    plt.subplot(x,y,z)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(st)
    
def Main():
    image = cv2.imread("line.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(512,512))
    plotimg(image,1,3,1,"orginal")
    sz = 3
    mask = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]
    image_with_line = line_detection(image,mask)
    plotimg(image_with_line,1,3,2,"line detection")

    image_thresold = thresolding(image_with_line)
    plotimg(image_thresold,1,3,3,"line detection after thresold")
    plt.tight_layout()
    plt.show()
   
    
Main()