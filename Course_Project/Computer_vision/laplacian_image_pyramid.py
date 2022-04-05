# importing libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# reading image
img0 = cv.imread("dollar.tif",0)
m,n = img0.shape
# level 0
# filtering image
img0_blur = cv.GaussianBlur(img0,(5,5),0)
# achieving laplacian image
img0_laplacian = np.float64(img0) - np.float64(img0_blur)

# level 1
# downsample image by 2
img1 = cv.resize(img0_blur,(int(n/2),int(m/2)))
# filtering image
img1_blur = cv.GaussianBlur(img1,(5,5),0)
# achieving laplacian image
img1_laplacian = np.float64(img1) - np.float64(img1_blur)

# level 2
# downsample image by 2
img2 = cv.resize(img1_blur,(int(n/4),int(m/4)))
# filtering image
img2_blur = cv.GaussianBlur(img2,(5,5),0)
# achieving laplacian image
img2_laplacian = np.float64(img2) - np.float64(img2_blur)

# reconstructing to level 1
# upsample image by 2
img1_reconst_blur = cv.resize(img2,(int(n/2),int(m/2)))
# adding upsampled image with laplacian
img1_reconst = np.float64(img1_laplacian) + np.float64(img1_reconst_blur)

# reconstructing to level 0
# upsample image by 2
img0_reconst_blur = cv.resize(img1_reconst,(n,m))
# adding upsampled image with laplacian
img0_reconst = np.float64(img0_laplacian) + np.float64(img0_reconst_blur)
# reconstructing to level 0 image without using laplacian
img_test = cv.resize(img2,(n,m))
# showing results
plt.subplot(1,2,1),plt.imshow(img0,'gray'),plt.yticks([]),plt.xticks([]),plt.title("1st level image")
plt.subplot(1,2,2),plt.imshow(img2,'gray'),plt.yticks([]),plt.xticks([]),plt.title("3rd level image")
plt.figure()
plt.subplot(1,2,1),plt.imshow(img0_laplacian,'gray'),plt.yticks([]),plt.xticks([]),plt.title("1st level laplacian image")
plt.subplot(1,2,2),plt.imshow(img2_laplacian,'gray'),plt.yticks([]),plt.xticks([]),plt.title("3rd level laplacian image")
plt.figure()
plt.imshow(img0_reconst,'gray'),plt.yticks([]),plt.xticks([]),plt.title("reconstructed image in using laplacian images")
plt.figure()
plt.imshow(img_test,'gray'),plt.yticks([]),plt.xticks([]),plt.title("reconstructed image without using laplacian images")
plt.show()