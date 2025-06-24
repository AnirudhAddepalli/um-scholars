from PIL import Image
import os
import shutil
import cv2
import matplotlib.pyplot as plt

def convert_images_to_jpg(data_dir, output_dir):
    """
    Convert all .tif images to JPEG format and move .jpg images to the output directory,
    maintaining the directory structure for positive and negative oGVHD images.

    :param data_dir: Directory containing the positive and negative oGVHD image folders
    :param output_dir: Directory where converted images will be saved
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the folders for positive and negative oGVHD images
    for category in ["Fluorescein_Stained_Images_Positive_oGVHD", "Fluorescein_Stained_Images_Negative_oGVHD"]:
        category_dir = os.path.join(data_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)

        # Iterate through each image in the folder
        for filename in os.listdir(category_dir):
            image_path = os.path.join(category_dir, filename)
            # Extract the base name without extension
            base_name, ext = os.path.splitext(filename)
            # Define the output path for the .jpg image
            output_image_path = os.path.join(output_category_dir, base_name + ".jpg")

            try:
                if ext.lower() == '.tif':  # Process TIFF files
                    with Image.open(image_path) as img:
                        # Convert to RGB (necessary for saving as JPEG) and save as JPEG
                        img.convert("RGB").save(output_image_path, "JPEG")
                        print(f"Converted and saved: {output_image_path}")
                elif ext.lower() == '.jpg':  # Move JPG files as they are
                    # Move the .jpg file to the output folder (preserve folder structure)
                    shutil.move(image_path, output_image_path)
                    print(f"Moved: {output_image_path}")
                else:
                    print(f"Skipped non-TIF/JPG file: {filename}")

            except Exception as e:
                print(f"Error with image {filename}: {e}")

def check_image_pixel_size(data_dir, expected_size=(1600, 1200)):
    """
    Check the dimensions of each image in the specified directory and print out any that do not match the expected size.
    
    :param data_dir: Directory containing the positive and negative oGVHD image folders
    :param expected_size: Tuple of (width, height) to check against, default is (1600, 1200)
    """
    # Iterate over the positive and negative folders
    for category in ["Fluorescein_Stained_Images_Positive_oGVHD", "Fluorescein_Stained_Images_Negative_oGVHD"]:
        category_dir = os.path.join(data_dir, category)
        
        # Iterate through each image in the folder
        for filename in os.listdir(category_dir):
            image_path = os.path.join(category_dir, filename)
            try:
                with Image.open(image_path) as img:
                    # Check if the image is not the expected size
                    if img.size != expected_size:
                        print(f"Image: {filename} has dimensions {img.size}, not {expected_size}.")
            except Exception as e:
                print(f"Error with image {filename}: {e}")

def remove_black_space(image):
    """
    Remove black space from an image by finding the bounding box of non-black content.
    Returns the cropped image.
    
    :param image: Input image (numpy array)
    :return: Cropped image with black space removed
    """
    # Convert to grayscale if image is RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Show the grayscale image
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    # Threshold the image to get binary image (increased threshold)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Show the thresholded image
    plt.subplot(1, 3, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Image')
    plt.axis('off')
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # Return original if no contours found
    
    # Find the bounding box of all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add a small padding (5% of the dimensions)
    height, width = image.shape[:2]
    pad_x = int(width * 0.05)
    pad_y = int(height * 0.05)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)
    
    # Crop the image
    cropped = image[y_min:y_max, x_min:x_max]
    
    # Show the cropped image
    plt.subplot(1, 3, 3)
    plt.imshow(cropped)
    plt.title('Cropped Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return cropped

def demonstrate_black_space_removal(image_path):
    """
    Demonstrate black space removal on a single image with before/after visualization.
    
    :param image_path: Path to the input image
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove black space
        cropped = remove_black_space(image_rgb)
        
        # Create figure with two subplots
        plt.figure(figsize=(15, 7))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        # Cropped image
        plt.subplot(1, 2, 2)
        plt.imshow(cropped)
        plt.title('After Black Space Removal')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print image dimensions
        print(f"Original image shape: {image_rgb.shape}")
        print(f"Cropped image shape: {cropped.shape}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

# Example usage
data_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data/"  # Path to your original data folder
output_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/"  # Path to save converted images

# Convert images to JPEG
convert_images_to_jpg(data_dir, output_dir)

# Check pixel sizes of images
# check_image_pixel_size(data_dir)

# Example usage
# demonstrate_black_space_removal("data_converted/Fluorescein_Stained_Images_Positive_oGVHD/Walczak_Jonathan_1971.09.26_2024.07.01 16_11_31_010.jpg")
