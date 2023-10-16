import cv2
import matplotlib.pyplot as plt
import numpy as np
def brightness_enhancement(image,low,high,rise_brightness):
    r,c = image.shape[:2]
    for i in range(r):
        for j in range(c):
            if(image[i][j]>=low and image[i][j]<=high):
                image[i][j]+=rise_brightness

def plotimg(image,x,y,z,st):
    im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(x,y,z)
    plt.imshow(im)
    plt.title(st)   
def Main():
    image = cv2.imread("sample.jpeg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(512,512))
    plotimg(image,2,1,1,"Main Image")
    low = 80
    high = 200
    rise_brightness = 50
    brightness_enhancement(image,low,high,rise_brightness)
    plotimg(image,2,1,2,"Enhanced Image")
    plt.show()
   
    
Main()