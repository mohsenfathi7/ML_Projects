# importing libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy.fft as f


# defining notch filter
def notch_filter(img,x,y,sigma):
    m,n = img.shape
    # creating distance matix
    d = np.zeros((m,n),np.float64)
    h = np.zeros((m,n),np.float64)
    for i in range(0,m):
        for j in range(0,n):
            # calculating distance
            d[i,j] = np.sqrt(np.power(i+1-x,2)+np.power(j+1-y,2))
    # low-pass gaussian filter
    h1 = np.exp(-np.power(d,2)/(2*np.power(sigma,2)))
    # high-pass gaussian filter
    h = 1-h1
    # applying filter to image
    filt = img * h
    return filt


# reading image
img = cv.imread('noiseball.png',0)
# calculating shifted center fft
img_f = f.fftshift(f.fft2(img))
# scaling magnitude for better visualisation
img_a = 20*np.log(np.abs(img_f)+1)
# filtering vertical line in frequency spectrum
img_f[0:110,157:163] = 0
img_f[145:,157:163] = 0
# applying defined notch filter for noise removal
i1 = notch_filter(img_f,77,135,15)
i2 = notch_filter(i1,77,169,15)
i3 = notch_filter(i2,176,148,15)
i4 = notch_filter(i3,176,183,15)
i4_a = 20*np.log(np.abs(i4)+1)
# inverse fft
img_en = f.ifft2(f.ifftshift(i4))
img_en = np.abs(img_en)
# showing results
plt.imshow(img_en,'gray')
plt.subplot(1,2,1),plt.imshow(img_a,'gray'),plt.xticks([]),plt.yticks([]),plt.title("original spectrum")
plt.subplot(1,2,2),plt.imshow(i4_a,'gray'),plt.xticks([]),plt.yticks([]),plt.title("Filtered spectrum")
plt.figure()
plt.subplot(1,2,1),plt.imshow(img,'gray'),plt.xticks([]),plt.yticks([]),plt.title("original image")
plt.subplot(1,2,2),plt.imshow(img_en,'gray'),plt.xticks([]),plt.yticks([]),plt.title("Filtered image")
plt.show()
