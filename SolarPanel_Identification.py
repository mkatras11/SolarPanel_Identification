import cv2
import numpy as np
import os
from tqdm import tqdm
import csv
import PIL
from PIL import Image
from PIL.ExifTags import TAGS
PIL.Image.MAX_IMAGE_PIXELS = 933120000



def detect_edges(input_image):
    
    # Calculate mean and standard deviation of pixel intensities
    mean_intensity = np.mean(input_image)
    std_deviation = np.std(input_image)
   
    # Define min and max threshold values based on image statistics
    min_thresh = int(max(0, mean_intensity - std_deviation))
    max_thresh = int(min(255, mean_intensity + std_deviation))
    
    # Apply Canny edge detection algorithm
    edges = cv2.Canny(image=input_image, threshold1=min_thresh, threshold2=max_thresh, apertureSize=3, L2gradient=True)
    

    return edges



def extract_contours(detections):

    # Extract contours from detections
    contours = []
    for detection in detections:
        contour = detection.get_contour()  # Assuming detection has a method named get_contour()
        contours.append(contour)

    return contours



def find_contours(edges):

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (assumed to be noise)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Minimum area threshold for valid contour
            filtered_contours.append(contour)

    return filtered_contours



def filter_rectangular_contours(contours):

    # Filter out non-rectangular contours
    filtered_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)  # Approximate contour with polygon
        if len(approx) == 4:  # Assuming rectangular contours have 4 vertices
            filtered_contours.append(contour)

    return filtered_contours



def find_centroids(filtered_contours):

    # Find centroids of filtered contours
    centroids = []
    for cnt in filtered_contours:
        M = cv2.moments(cnt)  # Calculate moments of the contour

        if M['m00'] == 0:  # Avoid division by zero
            continue

        centroid_x = int(M['m10'] / M['m00'])  # Calculate x-coordinate of centroid
        centroid_y = int(M['m01'] / M['m00'])  # Calculate y-coordinate of centroid
        centroids.append((centroid_x, centroid_y))

    return centroids



def draw_image(image, filtered_contours, centroids):

    # Draw contours and centroids on the image
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)  # Draw contours 

    for centroid in centroids:
        cv2.circle(image, centroid, 5, (0, 255, 0), -1)  # Draw centroids 
    return image



def preprocess_image(img_path):

    # Read and preprocess the input image
    input_image = cv2.imread(img_path)  # Read image from file

    image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    gaussian_blur = 5
    blurred_image = cv2.GaussianBlur(image_gray, (gaussian_blur, gaussian_blur), 0)  # Apply Gaussian blur

    img = cv2.medianBlur(blurred_image, 5)  # Apply median blur
    img = cv2.fastNlMeansDenoising(img, h=10)  # Apply non-local means denoising

    return img, input_image

def get_image_metadata(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is not None:
            metadata = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                metadata[tag_name] = value
            return metadata
        else:
            print("No metadata found for the image.")
            return None
    except (AttributeError, KeyError, IndexError):
        return None

def pixel2gps (metadata, centroids):

    # Get tie points and scale from image metadata
    tie_points = list(metadata.items())[0][1]
    scale = list(metadata.items())[1][1]
   
    gps_x0 = tie_points[3]
    gps_y0 = tie_points[4]

    scale_x = scale[0]
    scale_y = scale[1]

    # Convert pixel coordinates to GPS coordinates
    gps_coords = []
    for centroid in centroids:
        gps_coord_x = gps_x0 + (centroid[0] * scale_x)
        gps_coord_y = gps_y0 + (centroid[1] * scale_y)
        gps_coords.append((gps_coord_x, gps_coord_y))

    return gps_coords


def save_centroids_to_csv(img_path, centroids):

    # Save centroids to a CSV file
    if img_path.endswith(".tif"):
        csv_path = img_path.replace(".tif", "_centroids.csv")  # Generate CSV file path for .tif
    else:
        csv_path = img_path.replace(".jpg", "_centroids.csv")  # Generate CSV file path for .jpg

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Centroid_X", "Centroid_Y"])  # Write header row
        for centroid in centroids:
            writer.writerow(centroid)  # Write centroid coordinates to CSV

def save_gps_to_csv(img_path, gps_coords):
    
    # Save GPS coordinates to a CSV file
    if img_path.endswith(".tif"):
        csv_path = img_path.replace(".tif", "_gps.csv")  # Generate CSV file path for .tif
    else:
        csv_path = img_path.replace(".jpg", "_gps.csv")  # Generate CSV file path for .jpg

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["GPS_X", "GPS_Y"])  # Write header row
        for gps_coord in gps_coords:
            writer.writerow(gps_coord)  # Write GPS coordinates to CSV



def process_all_images(folder_path):

    # Process all images in the specified folder
    for root, dirs, files in os.walk(folder_path):

        for image_file in tqdm(files, desc="Processing Images"):  # Display progress bar

            if image_file.endswith((".tif", ".jpg")):

                img_path = os.path.join(root, image_file)  # Get full path of image file

                processed_image, input_image = preprocess_image(img_path)  # Preprocess the image

                canny_image = detect_edges(processed_image)  # Detect edges in the preprocessed image

                contours = find_contours(canny_image)  # Find contours in the edge-detected image

                filtered_contours = filter_rectangular_contours(contours)  # Filter out non-rectangular contours

                centroids = find_centroids(filtered_contours)  # Find centroids of filtered contours

                image_with_centroids = draw_image(input_image, filtered_contours, centroids)  # Draw contours and centroids on the original image

                metadata = get_image_metadata(img_path)  # Get metadata of the image

                if metadata is not None:
                    gps_centroids = pixel2gps(metadata, centroids)  # Convert pixel coordinates to GPS coordinates

                # Define output path for the processed image
                if image_file.endswith(".tif"):
                    output_path = os.path.join(root, image_file.replace(".tif", "_prediction.tif"))
                else:
                    output_path = os.path.join(root, image_file.replace(".jpg", "_prediction.jpg"))

                cv2.imwrite(output_path, image_with_centroids)  # Save the processed image

                save_centroids_to_csv(img_path, centroids)  # Save centroids to a CSV file
                
                if metadata is not None:
                    save_gps_to_csv(img_path, gps_centroids)  # Save GPS coordinates to a CSV file



if __name__ == "__main__":

    # Specify the folder containing images to be processed
    folder_path = "/home/mikek11/projects/SolarPanel_Identification/Vision_SolarPanels_Images" # Replace with the path to the folder containing the images
    
    process_all_images(folder_path)  # Process all images in the specified folder
