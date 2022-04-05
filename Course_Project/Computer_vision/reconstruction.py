# importing libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy.fft as f
# reading image
img = cv.imread('building.jpg',0)
# calculating shifted center fft
img_f = f.fftshift(f.fft2(img))
# scaling magnitude for better visualisation
img_a = np.log10(np.abs(img_f)+1)
abs_image = np.abs(img_f)
# calculating phase
phase_image = np.arctan2(np.imag(img_f),np.real(img_f))
# setting magnitude to 1
freq_n = np.cos(phase_image) + np.sin(phase_image)*1j
abs_n = np.abs(freq_n)
# reconstructing image using phase
img_reconst_phase = np.abs(f.ifft2(f.ifftshift(freq_n)))
# reconstructing image with only phase
img_reconst_abs = np.abs(f.ifft2(f.ifftshift(abs_image)))
# showing results
plt.subplot(1,2,1),plt.imshow(img,'gray'),plt.xticks([]),plt.yticks([]),plt.title("Image")
plt.subplot(1,2,2),plt.imshow(img_a,'gray'),plt.xticks([]),plt.yticks([]),plt.title("Spectrum")
plt.figure(),plt.imshow(img_reconst_phase,'gray')
plt.xticks([]),plt.yticks([]),plt.title("Reconstructed Image (phase)")
plt.figure(),plt.imshow(np.log(img_reconst_abs+1),'gray')
plt.xticks([]),plt.yticks([]),plt.title("Reconstructed Image (abs)")
plt.show()

