import cv2
from skimage import io
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
from matplotlib import pyplot as plt

img = io.imread("D:\PHOTOS AND PICTURES\Rocks_3.jpg", as_gray=True)

# Variance
k = 7
img_mean = ndimage.uniform_filter(img, (k, k))
img_sqr_mean = ndimage.uniform_filter(img**2, (k, k))
img_var = img_sqr_mean - img_mean**2
cv2.imshow("Variance", img_var)

# Gabor
ksize = 45
theta = np.pi/4
kernel = cv2.getGaborKernel((ksize, ksize), 5.0, theta, 10.0, 0.9, 0, ktype=cv2.CV_32F)
filtered_image = cv2.filter2D(img, cv2.CV_8UC3, kernel)
cv2.imshow("Gabor Filter", filtered_image)

# Entropy
entropy_img = entropy(img, disk(3))
cv2.imshow("Entropy", entropy_img)

# Scratch Analysis
plt.hist(entropy_img.flat, bins=100, range=(0, 5))
thresh = threshold_otsu(entropy_img)
binary = entropy_img <= thresh
cv2.imshow("Binary Entropy Image", binary.astype(np.uint8) * 255)

scratch_area = np.sum(binary == 1)
print("Scratched area is: ", scratch_area, "Square pixels")

scale = 0.45  # microns/pixel
print("Scratched area in sq. microns is: ", scratch_area * ((scale)**2), "Square pixels")

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
from skimage import io
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
from matplotlib import pyplot as plt

img = io.imread("D:\PHOTOS AND PICTURES\Rocks_3.jpg", as_gray=True)

# Variance
k = 7
img_mean = ndimage.uniform_filter(img, (k, k))
img_sqr_mean = ndimage.uniform_filter(img**2, (k, k))
img_var = img_sqr_mean - img_mean**2
cv2.imshow("Variance", img_var)

# Gabor
ksize = 45
theta = np.pi/4
kernel = cv2.getGaborKernel((ksize, ksize), 5.0, theta, 10.0, 0.9, 0, ktype=cv2.CV_32F)
filtered_image = cv2.filter2D(img, cv2.CV_8UC3, kernel)
cv2.imshow("Gabor Filter", filtered_image)

# Entropy
entropy_img = entropy(img, disk(3))
cv2.imshow("Entropy", entropy_img)

# Scratch Analysis
plt.hist(entropy_img.flat, bins=100, range=(0, 5))
thresh = threshold_otsu(entropy_img)
binary = entropy_img <= thresh
cv2.imshow("Binary Entropy Image", binary.astype(np.uint8) * 255)

scratch_area = np.sum(binary == 1)
print("Scratched area is: ", scratch_area, "Square pixels")

scale = 0.45  # microns/pixel
print("Scratched area in sq. microns is: ", scratch_area * ((scale)**2), "Square pixels")

cv2.waitKey(0)
cv2.destroyAllWindows()
