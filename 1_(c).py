import cv2
import matplotlib.pyplot as plt
import numpy as np
def plotimg(image,x,y,z,st):
    im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(x,y,z)
    plt.imshow(im)
    plt.title(st)
def thresolding(image):
    r= image.shape[0]
    c= image.shape[1]
    for i in range(0,r):
        for j in range(0,c):
            if image[i][j]>100:
                image[i][j] = 255
            else:
                image[i][j] = 0
    

def histogram(image,x,y,z,st):
    r,c = image.shape[:2]
    histogram = np.zeros(256)
    for i in range(r):
        for j in range(c):
            histogram[image[i][j]]+=1
    

    # Plot the histogram using matplotlib
    t = np.arange(0,256)
    
    plt.subplot(x,y,z)
    plt.bar(t,histogram,width=.8, color='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(st)
    
    
def Main():
    image = cv2.imread("sample.jpeg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(512,512))
    plotimg(image,2,2,1,"original Image")
    histogram(image,2,2,2,"Histogram of original")
    thresolding(image)
    plotimg(image,2,2,3,"Thresolding Image")
    histogram(image,2,2,4,"Histogram of Thresolding Image")
    plt.show()
    
Main()