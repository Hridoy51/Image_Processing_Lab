import cv2
import numpy as np
import matplotlib.pyplot as plt
image  = cv2.imread("sample.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original ", image)
h = image.shape[0]
w = image.shape[1]
center_h = h//2
center_w = w//2
left_top = image[0:center_h,0:center_w]
left_top = cv2.cvtColor(left_top, cv2.COLOR_BGR2RGB)
right_top = image[0:center_h,center_w:w]
left_bottom = image[center_h:h,0:center_w]
right_bottom = image[center_h:h,center_w:w]
#plt.figure(1)
plt.subplot(221)
plt.imshow(left_top)
plt.subplot(222)
plt.imshow(right_top)
plt.subplot(223)
plt.imshow(left_bottom)
plt.subplot(224)
plt.imshow(right_bottom)
plt.show()
