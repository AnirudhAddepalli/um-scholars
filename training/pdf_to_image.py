import fitz  # PyMuPDF
import os
from PIL import Image
import io

# Load the PDF
pdf_path = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/pdfs_to_convert/3000577797_Metzger_Philip_2021.12.20 15_40_27_IMAGING.pdf"
output_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/pdfs_to_convert"
os.makedirs(output_dir, exist_ok=True)

# Extract the PDF filename without extension
pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]

# Open PDF
doc = fitz.open(pdf_path)

# Target specifications
target_width, target_height = 1600, 1200
target_mode = 'RGB'

print(f"Processing PDF: {pdf_filename}")
print(f"Target: {target_width}x{target_height} RGB images")
print("-" * 60)

total_extracted = 0
image_counter = 1

page_images = doc.get_page_images(0)

print(f"Found {len(page_images)} images")

# Process each image on this page
for img_index, img in enumerate(page_images):
    xref = img[0]
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    
    # Get image dimensions and color mode using PIL
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        width, height = pil_image.size
        color_mode = pil_image.mode
        
        # Check if this matches our target specifications
        if width == target_width and height == target_height and color_mode == target_mode:
            # Save as JPG with PDF filename + sequential number
            image_filename = f"{pdf_filename}_{image_counter:03d}.jpg"
            output_path = os.path.join(output_dir, image_filename)
            
            pil_image.save(output_path, 'JPEG', quality=95)
            print(f"  Saved: {image_filename}")
            total_extracted += 1
            image_counter += 1
            
    except Exception as e:
        continue

print(f"\n" + "="*60)
print(f"EXTRACTION COMPLETE!")
print(f"Total {target_width}x{target_height} RGB images extracted: {total_extracted}")
print(f"Check the '{output_dir}' folder for extracted images.")
