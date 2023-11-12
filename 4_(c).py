import cv2
import numpy as np
import matplotlib.pyplot as plt

def plotimg(image,x,y,z,st):
    plt.subplot(x,y,z)
    plt.imshow(image,cmap='gray')
    plt.title(st)
#high pass
def ideal_H_Pass(image,D0):
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            if(D>D0):
                H[u,v] = 1
    im = image*H
    return im

#high pass
def gaussian(f_img,D0):
    M, N = f_img.shape
    Gaussian = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N): 
            D =np.sqrt( (u - M/2)**2 + (v - N/2)**2)
            Gaussian[u, v] = 1 - np.exp(-((D**2) / (2 * D0**2)))
    filtered_image=Gaussian*f_img
    return filtered_image
def main():
    # Load the grayscale image
    Original_image = cv2.imread('Fig0445(a) Characters Test Pattern 688x688.tif', cv2.IMREAD_GRAYSCALE)

    # Generate Gaussian noise
    noise = np.random.normal(7, 10, Original_image.shape).astype(np.uint8)

    # Add noise to the original image
    image = cv2.add(Original_image, noise)

    # Perform FFT on the original image
    fft_image = np.fft.fft2(Original_image)
    fft_Ori_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum_original = np.log(np.abs(fft_Ori_shifted) + 1)  # Apply log for visualization
    # Perform FFT on the noisy image
    fft_image = np.fft.fft2(image)
    fft_Noi_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum_Noisy = np.log(np.abs(fft_Noi_shifted) + 1)  # Apply log for visualization

    plotimg(Original_image,3,2,1,"Original image")
    plotimg(image,3,2,2,"Noisy Image")

    #construct and plot ideal High pass of orignal image 
    ideal_original = ideal_H_Pass(fft_Ori_shifted,30)
    Origial_Edge_Ideal = np.fft.ifftshift(ideal_original)
    original_ideal_Filtered = np.abs(np.fft.ifft2(Origial_Edge_Ideal))
    plotimg(original_ideal_Filtered,3,2,3,"ideal H_Pass of Original")

    #construct and plot Gaussian High pass of orignal image 
    gaussian_original = gaussian(fft_Ori_shifted,30)
    Origial_Edge_Gaussian = np.fft.ifftshift(gaussian_original)
    original_Gaussian_Filtered = np.abs(np.fft.ifft2(Origial_Edge_Gaussian))
    plotimg(original_Gaussian_Filtered,3,2,4,"Gaussian H_Pass of Original")

    #construct and plot ideal High pass of Noisy image 
    ideal_Noisy = ideal_H_Pass(fft_Noi_shifted,50)
    Noisy_Edge_ideal = np.fft.ifftshift(ideal_Noisy)
    Noisy_ideal_Filtered = np.abs(np.fft.ifft2(Noisy_Edge_ideal))
    plotimg(Noisy_ideal_Filtered,3,2,5,"ideal H_Pass of Noisy")

    #construct and plot Gaussian High pass of Noisy image 
    gaussian_Noisy = gaussian(fft_Noi_shifted,50)
    Noisy_Edge_gaussian = np.fft.ifftshift(gaussian_Noisy)
    Noisy_gaussian_Filtered = np.abs(np.fft.ifft2(Noisy_Edge_gaussian))
    plotimg(Noisy_gaussian_Filtered,3,2,6,"Gaussian H_Pass of Noisy")

    
    plt.tight_layout()
    plt.show()
main()