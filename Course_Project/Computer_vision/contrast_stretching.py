# importing libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# reading image
img = cv.imread("x-ray.PNG",0)
# getting image size
m,n = img.shape
low1 = 50
high1 = 125
img2 = img.copy()
# transferring pixel values to new valuse
for i in np.arange(0,m):
    for k in np.arange(0,n):
        if img[i,k]>low1 and img[i,k]<high1:
            img2[i,k] = (img[i,k] - low1 ) /(high1 - low1) * 255

# showing results
plt.subplot(1,2,1),plt.imshow(img,'gray'),plt.yticks([]),plt.xticks([]),plt.title("Original histogram")
plt.subplot(1,2,2),plt.imshow(img2,'gray'),plt.yticks([]),plt.xticks([]),plt.title("Enhance histogram")
plt.figure()
plt.subplot(1,2,1),plt.hist(img.ravel(),256,[0,255]),plt.title("Original histogram")
plt.subplot(1,2,2),plt.hist(img2.ravel(),256,[0,255]),plt.title("Stretched histogram")
plt.show()