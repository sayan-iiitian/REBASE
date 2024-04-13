import cv2
import numpy as np

# Read the image
image = cv2.imread('D:\PHOTOS AND PICTURES\Rocks_3.jpg', cv2.IMREAD_GRAYSCALE)

# Apply threshold to create a binary image
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply erosion
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Display the original, binary, and eroded images
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
