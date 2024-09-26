import cv2
import numpy as np
from skimage.util import random_noise
import matplotlib.pyplot as plt

# 1. Normalization
def normalize_image(img):
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    return img

# 2. Skew Correction
def deskew(image):
    co_ords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(co_ords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# 3. Noise Removal
def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

# 4. Thinning and Skeletonization
def thinning(image):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(image, kernel, iterations = 1)
    return erosion

# 5. Grayscale Image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 6. Thresholding or Binarization
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Load an example image
img_path = '/home/mrinal/Documents/LAN/images.png'
image = cv2.imread(img_path)

# Preprocessing steps
normalized_img = normalize_image(image)
gray_img = get_grayscale(normalized_img)
deskewed_img = deskew(gray_img)
denoised_img = remove_noise(deskewed_img)
thinned_img = thinning(denoised_img)
thresholded_img = thresholding(thinned_img)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 2)
plt.title('Normalized Image')
plt.imshow(normalized_img, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Grayscale Image')
plt.imshow(gray_img, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Deskewed Image')
plt.imshow(deskewed_img, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('Denoised Image')
plt.imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 6)
plt.title('Thinned & Thresholded Image')
plt.imshow(thresholded_img, cmap='gray')

plt.show()

