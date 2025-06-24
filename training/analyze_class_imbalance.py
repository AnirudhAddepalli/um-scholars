import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the data directory
train_dir = '/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/'

# Create ImageDataGenerator for analysis
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training set
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(512, 384),
    batch_size=8,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

print("\n=== Dataset Analysis ===")
print(f"Total training samples: {train_generator.samples}")
print(f"Classes: {train_generator.class_indices}")

# Calculate and display class distribution
total_samples = train_generator.samples
class_counts = train_generator.classes
unique_classes, class_counts = np.unique(class_counts, return_counts=True)

print("\nClass Distribution:")
for class_idx, count in zip(unique_classes, class_counts):
    class_name = list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(class_idx)]
    print(f"{class_name}: {count} samples ({count/total_samples*100:.1f}%)")

# Calculate class weights
class_weights = dict(zip(unique_classes, total_samples / (len(unique_classes) * class_counts)))
print("\nCalculated Class Weights:", class_weights)

# Create a bar plot of class distribution
plt.figure(figsize=(10, 6))
class_names = [list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(idx)] for idx in unique_classes]
sns.barplot(x=class_names, y=class_counts)
plt.title('Class Distribution in Training Dataset')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# Validation set
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(512, 384),
    batch_size=8,
    class_mode='binary',
    subset='validation',
    shuffle=True,
    seed=42
)

print(f"\nValidation samples: {val_generator.samples}")

# Build a test model with class weights
print("\n=== Building Test Model with Class Weights ===")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(512, 384, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile with class weights
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# Train for a few epochs to test
print("\n=== Training Test Model ===")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,  # Just a few epochs for testing
    class_weight=class_weights
)

# Evaluate and print metrics
print("\n=== Model Evaluation ===")
val_predictions = model.predict(val_generator)
val_predictions = (val_predictions > 0.5).astype(int)
val_true = val_generator.classes

print("\nClassification Report:")
print(classification_report(val_true, val_predictions))

# Plot confusion matrix
cm = confusion_matrix(val_true, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_test.png')
plt.close()

print("\nAnalysis complete. Check the generated plots for visualizations.")
print("Based on these results, we can make further adjustments to the model if needed.") 