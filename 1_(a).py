import cv2
import matplotlib.pyplot as plt
import numpy as np
def decrease_spatial_resulation(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,1)
    plt.imshow(image)
    r = image.shape[0]
    c = image.shape[1]
    total = 8
    for i in range(1,total):
        #image = cv2.resize(image,(r,c))
        temp_img = []
        for j in range(0,r,2):
            row = []
            for k in range(0,c,2):
                row.append(image[j][k])
            temp_img.append(row)
        temp_img = np.array(temp_img,dtype=np.uint8)
        image = temp_img
        temp_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB)
        plt.subplot(2,4,i+1)
        plt.imshow(temp_img)
        r=r//2
        c=c//2
        
    plt.show()
    
def Main():
    image = cv2.imread("sample.jpeg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(512,512))
    decrease_spatial_resulation(image)
   
    
Main()