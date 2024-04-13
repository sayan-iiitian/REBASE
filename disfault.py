import cv2
import numpy as np
import csv

# Function to estimate size in real-world units
def pixel_to_distance(pixel_value, distance_to_object, focal_length):
    return (pixel_value * distance_to_object) / focal_length  # pixel value to real-world distance

# Read the image
image = cv2.imread('D:\PHOTOS AND PICTURES\Rocks_3.jpg', cv2.IMREAD_GRAYSCALE)

# User-entered distance in meters
distance_to_object = float(input("Enter the distance to the object in meters: "))

# User-entered focal length in pixels
focal_length = float(input("Enter the focal length of the camera in pixels: "))

# Apply GaussianBlur to reduce noise and improve Canny edge detection
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection to detect edges in the image
canny_output = cv2.Canny(blurred_image, 80, 150)

# Apply dilation to connect nearby edges. To make the connecting edges as a single unit edge
kernel_size_dilation = 5
kernel_dilation = np.ones((kernel_size_dilation, kernel_size_dilation), np.uint8)
dilated_image = cv2.dilate(canny_output, kernel_dilation, iterations=1)

# Apply erosion to remove small edges
kernel_size_erosion = 5
kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
eroded_image = cv2.erode(dilated_image, kernel_erosion, iterations=1)

# Find contours in the eroded image
contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Store fault sizes in a list, Empty list to store the sizes of the detected faults
fault_sizes = []

# Calculate and print the size of each detected fault in real-world units
for i, contour in enumerate(contours):
    fault_size_pixels = cv2.contourArea(contour)
    fault_size_meters = pixel_to_distance(fault_size_pixels, distance_to_object, focal_length)
    print(f"Fault {i + 1} Size: {fault_size_meters:.2f} meters")
    fault_sizes.append(fault_size_meters)

# Write fault sizes to a CSV file
csv_filename = 'fault_sizes.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Fault', 'Size (meters)'])
    for i, size in enumerate(fault_sizes):
        csv_writer.writerow([f"Fault {i + 1}", size])

# Draw contours on the original image
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', canny_output)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Fault Detection Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Fault sizes have been written to {csv_filename}")
