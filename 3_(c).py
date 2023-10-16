import cv2
import matplotlib.pyplot as plt
import numpy as np

def hermonic_mean(image,mask):
    r,c = image.shape[:2]
    her_mean_img = image.copy()
    for i in range(r):
        for j in range(c):
            sum = 0
            for k in range(i-(mask//2),i+(mask//2)+1):
                for l in range(j-(mask//2),j+(mask//2)+1):
                    if(k>=0 and k<r and l>=0 and l<c):
                        sum += 1/float(image[k,l]+1e-4)
            her_mean_img[i,j] = min(int((mask*mask)/sum),255)
    return her_mean_img
def  geometric_mean(image,mask):
    r,c = image.shape[:2]
    geo_mean_img = image.copy()
    for i in range(r):
        for j in range(c):
            product = 1
            ct = 0
            for k in range(i-(mask//2),i+(mask//2)+1):
                for l in range(j-(mask//2),j+(mask//2)+1):
                    if(k>=0 and k<r and l>=0 and l<c):
                        if(image[k,l]!=0):
                            product=(product*int(image[k,l]))
                            ct += 1
            
            geo_mean_img[i,j] = min((product**(1/max(ct, 1))),255)       
    return geo_mean_img

def saltpaper(image):
    saltimg = image.copy()
    for i in  range(0,10000):
        c = np.random.randint(0, 510)
        r = np.random.randint(0, 510)
        saltimg[c,r] = 0
        c = np.random.randint(0, 510)
        r = np.random.randint(0, 510)
        #saltimg[c,r] = 255
    
    return saltimg


def plotimg(image,x,y,z,st):
    plt.subplot(x,y,z)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(st)
def Cal_psnr(before,after,peakValue):
    before = np.array(before,dtype=np.float64)
    after = np.array(after,dtype=np.float64)
    mse = np.mean((before-after)**2)
    psnr = 20*np.log10(peakValue)-10*np.log10(mse)
    before = np.array(before,dtype=np.uint8)
    after = np.array(after,dtype=np.uint8)
    return round(psnr,2)   
def Main():
    image = cv2.imread("Cat.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(512,512))
    plotimg(image,2,2,1,"orginal")

    #adding noise
    N_img = saltpaper(image)
    plotimg(N_img,2,2,2,f"Image_With_noise,psnr = psnr = {Cal_psnr(image,N_img,255)}")

    #filter.....
    mask = 5
    h_img = hermonic_mean(N_img,mask)
    plotimg(h_img,2,2,3,f"Hermonic_mean_Img,psnr = psnr = {Cal_psnr(image,h_img,255)}")
    g_img = geometric_mean(N_img,mask)
    plotimg(g_img,2,2,4,f"Geometric_mean_Img,psnr = psnr = {Cal_psnr(image,g_img,255)}")
    plt.show()
   
    
Main()