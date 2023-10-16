import cv2
import matplotlib.pyplot as plt
import numpy as np

def averageFilter(image,mask):
    r,c = image.shape[:2]
    avgi = image.copy()
    for i in range(r):
        for j in range(c):
            sum = 0
            for k in range(i-(mask//2),i+(mask//2)+1):
                for l in range(j-(mask//2),j+(mask//2)+1):
                    if(k>=0 and k<r and l>=0 and l<c):
                        # Zero padding
                        temp = (image[k,l]//((mask*mask)))
                        sum+=temp
                        #Wrapping
                        #temp = image[k%r,l%c]//((mask*mask))
            
            avgi[i,j] = (sum)
    return avgi

def saltpaper(image):
    saltimg = image.copy()
    for i in  range(0,1000):
        c = np.random.randint(0, 510)
        r = np.random.randint(0, 510)
        saltimg[c,r] = 0
        c = np.random.randint(0, 510)
        r = np.random.randint(0, 510)
        saltimg[c,r] = 255
    
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
    image = cv2.imread("sample1.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(512,512))
    plotimg(image,2,3,1,"orginal")

    #adding noise
    N_image = saltpaper(image)
    plotimg(N_image,2,3,2,f"Image_With_noise, psnr = psnr = {Cal_psnr(image,N_image,255)}")

    #filter.....
    
    avgimg_33 = averageFilter(N_image,3)
    plotimg(avgimg_33,2,3,3,f"avgImage_33, psnr = {Cal_psnr(image,avgimg_33,255)}")
    avgimg_55 = averageFilter(N_image,5)
    plotimg(avgimg_55,2,3,4,f"avgImage_55, psnr = {Cal_psnr(image,avgimg_55,255)}")
    avgimg_77 = averageFilter(N_image,7)
    plotimg(avgimg_77,2,3,5,f"avgImage_77,psnr = {Cal_psnr(image,avgimg_77,255)}")
    avgimg_99 = averageFilter(N_image,9)
    plotimg(avgimg_99,2,3,6,f"avgImage_99, psnr = {Cal_psnr(image,avgimg_99,255)}")
    print(Cal_psnr(image,avgimg_33,255))
    print(Cal_psnr(image,avgimg_55,255))
    print(Cal_psnr(image,avgimg_77,255))
    print(Cal_psnr(image,avgimg_99,255))
    plt.show()
   
    
Main()