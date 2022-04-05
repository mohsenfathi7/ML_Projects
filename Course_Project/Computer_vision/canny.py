# importing libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# defining Canny function
def canny(img):
    # 3 by 3 gaussian kernel
    gauusian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    # filtering with kernel
    img_f = cv.filter2D(img, cv.CV_64F, gauusian_kernel)
    # gradient in x direction
    maskx = np.array([[-1], [1]])
    # gradient in y direction
    masky = np.transpose(maskx)
    # filtering with kernel
    gx = cv.filter2D(img_f, cv.CV_64F, maskx)
    gy = cv.filter2D(img_f, cv.CV_64F, masky)
    # magnitude of gradient
    mag = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    # direction of gradient
    direct = 180 / np.pi * np.arctan2(gy, gx) +180
    m, n = img.shape
    # making non-maximum suppressed image
    img_nms = np.zeros((m, n), np.float64)
    for i in range(1, m - 1):
        for k in range(1, n - 1):
            # 0 and 180 degrees direction
            if (direct[i, k] > 0 and direct[i, k] <= 22.5) or (direct[i, k] > 157.5 and direct[i, k] <= 202.5):
                if mag[i, k] >= mag[i - 1, k] and mag[i, k] >= mag[i + 1, k]:
                    img_nms[i, k] = mag[i, k]
            # 45 and 225 degrees direction
            elif (direct[i, k] > 22.5 and direct[i, k] <= 67.5) or (direct[i, k] > 202.5 and direct[i, k] <= 247.5):
                if mag[i, k] >= mag[i - 1, k - 1] and mag[i, k] >= mag[i + 1, k + 1]:
                    img_nms[i, k] = mag[i, k]
            # 90 and 270 degrees direction
            elif (direct[i, k] > 67.5 and direct[i, k] <= 112.5) or (direct[i, k] > 247.5 and direct[i, k] <= 295.5):
                if mag[i, k] >= mag[i, k + 1] and mag[i, k] >= mag[i, k - 1]:
                    img_nms[i, k] = mag[i, k]
            # 135 and 315 degrees direction
            else:
                if mag[i, k] >= mag[i - 1, k + 1] and mag[i, k] >= mag[i + 1, k - 1]:
                    img_nms[i, k] = mag[i, k]
    return img_f,mag,img_nms


# reading image
img = cv.imread("dollar.tif",0)
# getting results from defined function
img_filtered,magnitude,img_canny = canny(img)
# showing results
plt.figure(),plt.imshow(img,'gray'),plt.yticks([]),plt.xticks([]),plt.title("Original image")
plt.figure(),plt.imshow(img_filtered,'gray'),plt.yticks([]),plt.xticks([]),plt.title("Filtered image")
plt.figure(),plt.imshow(magnitude,'gray'),plt.yticks([]),plt.xticks([]),plt.title("Magnitude of gradient")
plt.figure(),plt.imshow(img_canny,'gray'),plt.yticks([]),plt.xticks([]),plt.title("Nonmaxima suppressed")
plt.figure()
plt.subplot(1,2,1),plt.imshow(magnitude[147:180,60:110],'gray'),plt.yticks([]),plt.xticks([]),plt.title("Magnitude of gradient")
plt.subplot(1,2,2),plt.imshow(img_canny[147:180,60:110],'gray'),plt.yticks([]),plt.xticks([]),plt.title("Nonmaxima suppressed")
plt.show()