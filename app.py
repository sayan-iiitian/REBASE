import streamlit as st
import cv2
import numpy as np
import csv
from skimage import io, filters, feature
from skimage.color import rgb2gray
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
from io import BytesIO

def process_image(image):
    # Existing logic from app.py
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_output = cv2.Canny(blurred_image, 80, 150)
    kernel_size_dilation = 5
    kernel_dilation = np.ones((kernel_size_dilation, kernel_size_dilation), np.uint8)
    dilated_image = cv2.dilate(canny_output, kernel_dilation, iterations=1)
    kernel_size_erosion = 5
    kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
    eroded_image = cv2.erode(dilated_image, kernel_erosion, iterations=1)
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 1000
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)
    return result_image


def process_image_with_disfault(image, distance_to_object, focal_length):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve Canny edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
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
    
    # Calculate fault sizes and write to CSV
    fault_sizes = []
    for i, contour in enumerate(filtered_contours):
        fault_size_pixels = cv2.contourArea(contour)
        fault_size_meters = pixel_to_distance(fault_size_pixels, distance_to_object, focal_length)
        fault_sizes.append(fault_size_meters)

    csv_filename = 'fault_sizes.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Fault', 'Size (meters)'])
        for i, size in enumerate(fault_sizes):
            csv_writer.writerow([f"Fault {i + 1}", size])

    # Draw contours on the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)
    return result_image, fault_sizes

def process_image_with_erosion(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Apply erosion
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    
    # Convert the eroded image back to BGR for display
    eroded_image_bgr = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)
    
    return eroded_image_bgr

def process_image_with_hough_transform(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    
    # Perform Hough Transform for line detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    
    # Draw the detected lines on the original image
    if lines is not None:
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
    
    return image

def pixel_to_distance(pixel_value, distance_to_object, focal_length):
    return (pixel_value * distance_to_object) / focal_length


def process_image_with_ridge_detection(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply ridge detection filters
    meijering_img = meijering(gray_image)
    sato_img = sato(gray_image)
    frangi_img = frangi(gray_image)
    hessian_img = hessian(gray_image)
    
    # Create a figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 1].imshow(meijering_img, cmap='gray')
    axes[0, 1].set_title('Meijering')
    axes[1, 0].imshow(sato_img, cmap='gray')
    axes[1, 0].set_title('Sato')
    axes[1, 1].imshow(frangi_img, cmap='Reds_r')
    axes[1, 1].set_title('Frangi')
    
    # Convert the figure to a PIL Image and then to a NumPy array
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def main():
    st.title("Image Processing with Streamlit")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Process Image with fault.py"):
            result_image = process_image(image)
            st.image(result_image, caption="Processed Image", use_column_width=True)

        distance_to_object = st.number_input("Enter the distance to the object in meters:", value=1.0)
        focal_length = st.number_input("Enter the focal length of the camera in pixels:", value=1000.0)

        if st.button("Process Image with disfault.py"):
            result_image, fault_sizes = process_image_with_disfault(image, distance_to_object, focal_length)
            st.image(result_image, caption="Processed Image with disfault.py", use_column_width=True)
            st.write("Fault Sizes:", fault_sizes)

        if st.button("Process Image with erosion.py"):
            eroded_image = process_image_with_erosion(image)
            st.image(eroded_image, caption="Eroded Image", use_column_width=True)

        if st.button("Process Image with Hough Transform"):
            hough_transform_image = process_image_with_hough_transform(image)
            st.image(hough_transform_image, caption="Image with Hough Transform", use_column_width=True)

        if st.button("Process Image with Ridge Detection"):
            ridge_detection_image = process_image_with_ridge_detection(image)
            st.image(ridge_detection_image, caption="Ridge Detection Results", use_column_width=True)

if __name__ == "__main__":
    main()
