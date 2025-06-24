import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def get_random_samples(data_dir, num_samples=3):
    """
    Get random sample images from both positive and negative oGVHD classes.
    
    Args:
        data_dir: Directory containing the positive and negative oGVHD image folders
        num_samples: Number of samples to get from each class
    
    Returns:
        Dictionary containing selected image paths for each class
    """
    selected_samples = {
        'positive': [],
        'negative': []
    }
    
    # Process both positive and negative classes
    for class_name, category in [
        ('positive', "Fluorescein_Stained_Images_Positive_oGVHD"),
        ('negative', "Fluorescein_Stained_Images_Negative_oGVHD")
    ]:
        category_dir = os.path.join(data_dir, category)
        
        # Get list of image files
        image_files = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.tif'))]
        
        # Select random samples
        selected_files = np.random.choice(image_files, size=min(num_samples, len(image_files)), replace=False)
        
        # Store full paths
        selected_samples[class_name] = [os.path.join(category_dir, f) for f in selected_files]
    
    return selected_samples

def visualize_samples_at_size(selected_samples, target_size, num_samples=3, save_dir=None):
    """
    Visualize the selected samples at a specific size.
    
    Args:
        selected_samples: Dictionary containing selected image paths
        target_size: Tuple of (width, height) for resizing images
        num_samples: Number of samples per class
        save_dir: Directory to save the visualization
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 10))
    fig.suptitle(f'Sample Images from oGVHD Dataset ({target_size[0]}x{target_size[1]})', fontsize=16)
    
    # Process both positive and negative classes
    for class_idx, (class_name, image_paths) in enumerate(selected_samples.items()):
        # Display each selected image
        for idx, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            
            # Resize image to target size
            img = img.resize(target_size)
            
            # Convert to numpy array and display
            img_array = np.array(img)
            axes[class_idx, idx].imshow(img_array)
            axes[class_idx, idx].axis('off')
            
            # Add title for the first image in each row
            if idx == 0:
                axes[class_idx, idx].set_title(f"{class_name.capitalize()} oGVHD", 
                                             fontsize=12, pad=10)
    
    plt.tight_layout()
    
    # Save the figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'samples_{target_size[0]}x{target_size[1]}.png')
        plt.savefig(save_path)  # Removed quality settings to show actual resolution differences
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def visualize_grayscale_samples(selected_samples, target_size=(512, 384), num_samples=3, save_dir=None):
    """
    Visualize the selected samples in RGB at a specific size, with and without black space.
    
    Args:
        selected_samples: Dictionary containing selected image paths
        target_size: Tuple of (width, height) for resizing images
        num_samples: Number of samples per class
        save_dir: Directory to save the visualization
    """
    # Create a figure with subplots - 2 rows for classes, 2 columns for each sample (original and cropped)
    fig, axes = plt.subplots(2, num_samples * 2, figsize=(20, 10))
    fig.suptitle(f'Sample Images from oGVHD Dataset ({target_size[0]}x{target_size[1]})', fontsize=16)
    
    # Process both positive and negative classes
    for class_idx, (class_name, image_paths) in enumerate(selected_samples.items()):
        # Display each selected image
        for idx, img_path in enumerate(image_paths):
            # Read image using OpenCV
            img = cv2.imread(img_path)
            
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create binary mask and find contours for black space removal
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:  # Check if any contours were found
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Crop the image
                img_cropped = img[y:y+h, x:x+w]
                img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            else:
                # If no contours found, use original image
                img_cropped_rgb = img_rgb
            
            # Resize both original and cropped images
            img_resized = cv2.resize(img_rgb, target_size)
            img_cropped_resized = cv2.resize(img_cropped_rgb, target_size)
            
            # Display original RGB image
            axes[class_idx, idx*2].imshow(img_resized)
            axes[class_idx, idx*2].axis('off')
            
            # Display cropped RGB image
            axes[class_idx, idx*2 + 1].imshow(img_cropped_resized)
            axes[class_idx, idx*2 + 1].axis('off')
            
            # Add titles for the first image in each row
            if idx == 0:
                axes[class_idx, idx*2].set_title(f"{class_name.capitalize()} oGVHD\nOriginal", 
                                               fontsize=12, pad=10)
                axes[class_idx, idx*2 + 1].set_title(f"{class_name.capitalize()} oGVHD\nBlack Space Removed", 
                                                   fontsize=12, pad=10)
    
    plt.tight_layout()
    
    # Save the figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'samples_{target_size[0]}x{target_size[1]}.png')
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def visualize_black_space_removal(data_dir, target_size=(512, 384), save_dir=None):
    """
    Visualize 2 positive and 2 negative images before and after black space removal.
    
    Args:
        data_dir: Directory containing the positive and negative oGVHD image folders
        target_size: Tuple of (width, height) for resizing images
        save_dir: Directory to save the visualization
    """
    # Get 2 samples from each class
    selected_samples = get_random_samples(data_dir, num_samples=2)
    
    # Create a figure with subplots - 2 rows for classes, 2 columns for each sample (before and after)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Black Space Removal Visualization ({target_size[0]}x{target_size[1]})', fontsize=16)
    
    # Process both positive and negative classes
    for class_idx, (class_name, image_paths) in enumerate(selected_samples.items()):
        # Display each selected image
        for idx, img_path in enumerate(image_paths):
            # Read image using OpenCV
            img = cv2.imread(img_path)
            
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create binary mask and find contours for black space removal
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:  # Check if any contours were found
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Crop the image
                img_cropped = img[y:y+h, x:x+w]
                img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            else:
                # If no contours found, use original image
                img_cropped_rgb = img_rgb
            
            # Resize both original and cropped images
            img_resized = cv2.resize(img_rgb, target_size)
            img_cropped_resized = cv2.resize(img_cropped_rgb, target_size)
            
            # Display original image (before black space removal)
            axes[class_idx, idx*2].imshow(img_resized)
            axes[class_idx, idx*2].axis('off')
            axes[class_idx, idx*2].set_title(f"Before\n{img_resized.shape[1]}x{img_resized.shape[0]}", 
                                           fontsize=10, pad=5)
            
            # Display cropped image (after black space removal)
            axes[class_idx, idx*2 + 1].imshow(img_cropped_resized)
            axes[class_idx, idx*2 + 1].axis('off')
            axes[class_idx, idx*2 + 1].set_title(f"After\n{img_cropped_resized.shape[1]}x{img_cropped_resized.shape[0]}", 
                                               fontsize=10, pad=5)
            
            # Add class labels for the first image in each row
            if idx == 0:
                axes[class_idx, 0].set_ylabel(f"{class_name.capitalize()} oGVHD", 
                                            fontsize=12, fontweight='bold')
    
    # Add overall column labels
    fig.text(0.25, 0.95, 'Sample 1', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.75, 0.95, 'Sample 2', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    
    plt.show()

if __name__ == "__main__":
    data_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/"
    save_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/pictures/"
    
    # Get random samples once
    print("Selecting random samples...")
    selected_samples = get_random_samples(data_dir)
    
    # Visualize the same samples at different sizes
    # sizes = [(128, 96), (256, 192), (512, 384), (1024, 768), (1600, 1200)]
    # for size in sizes:
        # print(f"\nVisualizing images at {size[0]}x{size[1]}...")
        # visualize_samples_at_size(selected_samples, target_size=size, save_dir=save_dir)
    
    # Visualize grayscale samples at 512x384
    #print("\nVisualizing grayscale images at 512x384...")
    #visualize_grayscale_samples(selected_samples, target_size=(512, 384))
    
    # Visualize black space removal
    print("\nVisualizing black space removal...")
    visualize_black_space_removal(data_dir, target_size=(512, 384), save_dir=save_dir) 