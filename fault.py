'''
import cv2
import numpy as np

# Read the image
image = cv2.imread('D:\PHOTOS AND PICTURES\Rocks_3.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise and improve Canny edge detection
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
canny_output = cv2.Canny(blurred_image, 80, 150)

# Apply dilation to connect nearby edges
kernel_size_dilation = 5
kernel_dilation = np.ones((kernel_size_dilation, kernel_size_dilation), np.uint8)
dilated_image = cv2.dilate(canny_output, kernel_dilation, iterations=1)

# Apply erosion to remove small edges
kernel_size_erosion = 5
kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
eroded_image = cv2.erode(dilated_image, kernel_erosion, iterations=1)

# Find contours in the eroded image
contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area (remove small contours)
min_contour_area = 1000
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

# Draw contours on the original image
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', canny_output)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Fault Detection Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2
import numpy as np

# Define the path to the image
image_path = r'D:\PHOTOS AND PICTURES\Rocks_3.jpg'

# Read the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at path {image_path}. Check the file path and try again.")
else:
    # Apply GaussianBlur to reduce noise and improve Canny edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detection
    canny_output = cv2.Canny(blurred_image, 80, 150)

    # Apply dilation to connect nearby edges
    kernel_size_dilation = 5
    kernel_dilation = np.ones((kernel_size_dilation, kernel_size_dilation), np.uint8)
    dilated_image = cv2.dilate(canny_output, kernel_dilation, iterations=1)

    # Apply erosion to remove small edges
    kernel_size_erosion = 5
    kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
    eroded_image = cv2.erode(dilated_image, kernel_erosion, iterations=1)

    # Find contours in the eroded image
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (remove small contours)
    min_contour_area = 1000
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    # Draw contours on the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Canny Edge Detection', canny_output)
    cv2.imshow('Dilated Image', dilated_image)
    cv2.imshow('Eroded Image', eroded_image)
    cv2.imshow('Fault Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

