import cv2
import matplotlib.pyplot as plt
import numpy as np
def make_gray_scale(image):
    im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(2,1,1)
    plt.imshow(im)
    r,c = image.shape[:2]
    temp_img = []
    for i in range(r):
        row=[]
        for j in range(c):
            pixel =  0.299 * image[i,j,0] + 0.587 *image[i,j,1]  + 0.114 * image[i,j,2]
            row.append(pixel)
        temp_img.append(row)
    temp_img = np.array(temp_img,dtype=np.uint8)
    temp_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,1,2)
    plt.imshow(temp_img)
    plt.show()
        

    
def Main():
    image = cv2.imread("sample.jpeg")
    image = cv2.resize(image,(512,512))
    make_gray_scale(image)
   
    
Main()