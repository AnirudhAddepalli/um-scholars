from PIL import Image
import os

folder = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/Fluorescein_Stained_Images_Positive_oGVHD"
for filename in os.listdir(folder):
    try:
        with Image.open(os.path.join(folder, filename)) as img:
            img.verify()  # Check if the image is valid
    except Exception as e:
        print(f"Error with image {filename}: {e}")
