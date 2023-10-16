import cv2
image  = cv2.imread("sample.jpeg")
cv2.imshow("Original ", image)
h = image.shape[0]
w = image.shape[1]
center_h = h//2
center_w = w//2
left_top = image[0:center_h,0:center_w]
right_top = image[0:center_h,center_w:w]
left_bottom = image[center_h:h,0:center_w]
right_bottom = image[center_h:h,center_w:w]
cv2.imshow("Left_top ", left_top)
cv2.imshow("Right_top ", right_top)
cv2.imshow("Left_bottom ", left_bottom)
cv2.imshow("Right_bottom", right_bottom)
#concat_hori = np.concatenate((left_top,right_top),axis = 1)
#concat_ver =  np.concatenate((left_bottom,right_bottom),axis = 1)
#final_img = np.concatenate((concat_hori,concat_ver),axis = 0)
#cv2.imshow("final",final_img)
cv2.waitKey(0)


