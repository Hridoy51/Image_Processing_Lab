import cv2
import numpy as np
import matplotlib.pyplot as plt

def plotimg(image,x,y,z,st):
    plt.subplot(x,y,z)
    plt.imshow(image,cmap="gray")
    plt.title(st)

def ideal_L_Pass(image,D0):
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            if(D<=D0):
                H[u,v] = 1
    im = image*H
    return im
def main():
    # Load the grayscale image
    Original_image = cv2.imread('Cat.jpg', cv2.IMREAD_GRAYSCALE)

    # Generate Gaussian noise
    noise = np.random.normal(7, 10, Original_image.shape).astype(np.uint8)

    # Add noise to the original image
    image = cv2.add(Original_image, noise)

    # Perform FFT on the image
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)  # Apply log for visualization
    plotimg(Original_image,4,2,1,"Original image")
    plotimg(image,4,2,2,"Noisy Image")
    plotimg(magnitude_spectrum,4,2,3,"DFT magnitude spectrum")

    #using ideal filter 
    d0 = 20
    for i in range(5):
        filtered_ideal = ideal_L_Pass(fft_shifted,d0)
        #perform Inverse fft on ideal filtered image
        reconstructed_ishifted = np.fft.ifftshift(filtered_ideal)
        reconstructed_ishifted_ifft = np.fft.ifft2(reconstructed_ishifted).real
        plotimg(reconstructed_ishifted_ifft,4,2,4+i,f"Reconstructed image using ideal, d0 = {d0}")
        d0+=30

    
    plt.show()
main()


