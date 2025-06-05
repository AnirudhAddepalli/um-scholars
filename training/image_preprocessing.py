from PIL import Image
import os
import shutil

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

# Example usage
data_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data/"  # Path to your original data folder
output_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/"  # Path to save converted images

# Convert images to JPEG
convert_images_to_jpg(data_dir, output_dir)

# Check pixel sizes of images
# check_image_pixel_size(data_dir)
