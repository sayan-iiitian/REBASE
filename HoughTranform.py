import cv2
import numpy as np

# Load an image from file
image = cv2.imread('D:\PHOTOS AND PICTURES\Rocks_3.jpg', cv2.IMREAD_GRAYSCALE)

# Apply edge detection (you can use Canny or other edge detection methods)
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Perform Hough Transform for line detectio0n
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Draw the detected lines on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
