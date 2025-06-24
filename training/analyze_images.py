import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from pathlib import Path

def analyze_image_distribution(data_dir):
    """Analyze and print the distribution of images in each class."""
    class_counts = {}
    total_images = 0
    
    # Count images in each class
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            n_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = n_images
            total_images += n_images
    
    # Print distribution
    print("\nImage Distribution:")
    print("-" * 40)
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count} images ({percentage:.2f}%)")
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Distribution of Images by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_sample_images(data_dir, n_samples=5):
    """Display sample images from each class."""
    plt.figure(figsize=(15, 10))
    
    for i, class_name in enumerate(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            # Get sample images
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:n_samples]
            
            for j, image_file in enumerate(image_files):
                img_path = os.path.join(class_path, image_file)
                img = load_img(img_path, target_size=(512, 512))
                
                # Plot original image
                plt.subplot(len(os.listdir(data_dir)), n_samples, i * n_samples + j + 1)
                plt.imshow(img)
                plt.title(f'{class_name}\n{image_file}')
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def check_image_preprocessing(data_dir):
    """Check image preprocessing and display statistics."""
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Create generators
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(512, 512),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(512, 512),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    # Get a batch of images
    train_images, train_labels = next(train_generator)
    
    # Print image statistics
    print("\nImage Statistics:")
    print("-" * 40)
    print(f"Image shape: {train_images[0].shape}")
    print(f"Value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"Mean value: {train_images.mean():.3f}")
    print(f"Standard deviation: {train_images.std():.3f}")
    
    # Display histogram of pixel values
    plt.figure(figsize=(10, 6))
    plt.hist(train_images[0].flatten(), bins=50)
    plt.title('Distribution of Pixel Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Print dataset split information
    print("\nDataset Split:")
    print("-" * 40)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")

def main():
    # Set the data directory
    data_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/"
    
    print("Starting image analysis...")
    
    # 1. Analyze image distribution
    print("\n1. Analyzing image distribution...")
    analyze_image_distribution(data_dir)
    
    # 2. Visualize sample images
    print("\n2. Visualizing sample images...")
    visualize_sample_images(data_dir)
    
    # 3. Check image preprocessing
    print("\n3. Checking image preprocessing...")
    check_image_preprocessing(data_dir)

if __name__ == "__main__":
    main() 