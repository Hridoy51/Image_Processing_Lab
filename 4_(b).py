import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('sample1.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise to the image
mean = 0
stddev = 25  # Adjust the standard deviation to control noise intensity
noisy_image = image + np.random.normal(mean, stddev, image.shape).astype(np.uint8)

# Define a function to create an Ideal Lowpass Filter
def ideal_lowpass_filter(shape, d0):
    rows, cols = shape
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = np.zeros_like(D)
    H[D <= d0] = 1
    return H

# Define a range of D0 values to test
d0_values = [10, 30, 50, 100]

# Apply Ideal Lowpass Filter with different D0 values and plot the results
plt.figure(figsize=(12, 12))
plt.subplot(3, 3, 1), plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image'), plt.axis('off')

for i, d0 in enumerate(d0_values):
    ideal_lp_kernel = ideal_lowpass_filter(image.shape, d0)
    filtered_image = np.fft.ifft2(np.fft.fft2(noisy_image) * ideal_lp_kernel).real
    plt.subplot(3, 3, i + 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Ideal LP (D0={d0})'), plt.axis('off')

plt.tight_layout()
plt.show()
