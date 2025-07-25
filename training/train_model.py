import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import cv2
import time  # Add time module import

# Start timing
start_time = time.time()

# Define the data directory
train_dir = '/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/'

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    
)

# Create training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(1024, 768),
    batch_size=4,
    class_mode='binary',
    color_mode='rgb',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(1024, 768),
    batch_size=4,
    class_mode='binary',
    color_mode='rgb',
    subset='validation'
)

# Calculate class weights
total_samples = train_generator.samples
class_counts = np.bincount(train_generator.classes)
class_weights = {
    0: total_samples / (2 * class_counts[0]),
    1: total_samples / (2 * class_counts[1])
}
print("\nClass weights:", class_weights)

# Create model with reduced complexity for better generalization
model = Sequential([
    # First conv block - reduced filters
    Conv2D(16, (3, 3), activation='relu', input_shape=(1024, 768, 3)),  # Reduced from 32 to 16
    MaxPooling2D(2, 2),
    
    # Second conv block - reduced filters
    Conv2D(32, (3, 3), activation='relu'),  # Reduced from 64 to 32
    MaxPooling2D(2, 2),
    
    # Third conv block - reduced filters
    Conv2D(64, (3, 3), activation='relu'),  # Reduced from 128 to 64
    MaxPooling2D(2, 2),
    
    # Fourth conv block - reduced filters
    Conv2D(128, (3, 3), activation='relu'),  # Reduced from 256 to 128
    MaxPooling2D(2, 2),
    
    # Fifth conv block - reduced filters
    Conv2D(256, (3, 3), activation='relu'),  # Reduced from 512 to 256
    MaxPooling2D(2, 2),
    
    # Sixth conv block - reduced filters
    Conv2D(256, (3, 3), activation='relu'),  # Reduced from 512 to 256
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(256, activation='relu'),  # Reduced from 512 to 256
    Dropout(0.5),
    Dense(128, activation='relu'),  # Reduced from 256 to 128
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# Save the model
model.save('models/oGVHD_model_1024x768_RGB.h5')
print("\nModel saved to models/oGVHD_model_1024x768_RGB.h5")

# Evaluate the model
val_generator.reset()
steps = len(val_generator)
print(f"Validation steps: {steps}")
print(f"Validation samples: {val_generator.samples}")
print(f"Validation batch size: {val_generator.batch_size}")

# Get all validation data at once to ensure consistent evaluation
val_images = []
val_labels = []
val_generator.reset()

for i in range(steps):
    batch_x, batch_y = next(val_generator)
    val_images.append(batch_x)
    val_labels.append(batch_y)

val_images = np.vstack(val_images)
val_labels = np.concatenate(val_labels)

# Evaluate on the complete validation set
loss, accuracy = model.evaluate(val_images, val_labels, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Get predictions on the same data
val_predictions = model.predict(val_images)
val_predictions_binary = (val_predictions > 0.5).astype(int)
val_true = val_labels

# Calculate metrics
precision = precision_score(val_true, val_predictions_binary)
recall = recall_score(val_true, val_predictions_binary)
f1 = f1_score(val_true, val_predictions_binary)
auc_roc = roc_auc_score(val_true, val_predictions)
specificity = recall_score(val_true, val_predictions_binary, pos_label=0)

print("\nValidation data shape:", val_images.shape)
print("Validation labels shape:", val_labels.shape)
print("Validation labels distribution:", np.bincount(val_labels.astype(int)))
print("Predictions distribution:", np.bincount(val_predictions_binary.flatten()))

print("\nAdditional Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"Specificity: {specificity:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(val_true, val_predictions_binary))

# Plot confusion matrix
plt.figure(figsize=(12, 10))  # Increased figure size
cm = confusion_matrix(val_true, val_predictions_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative oGVHD', 'Predicted Positive oGVHD'],
            yticklabels=['Actual Negative oGVHD', 'Actual Positive oGVHD'],
            annot_kws={'size': 16},  # Increased annotation text size
            cbar_kws={'label': 'Count'})

# Add labels for True Positives, False Positives, True Negatives, False Negatives with larger font
plt.text(0.5, 0.3, 'True Negatives', ha='center', va='center', color='black', fontweight='bold', fontsize=14)
plt.text(1.5, 0.3, 'False Positives', ha='center', va='center', color='black', fontweight='bold', fontsize=14)
plt.text(0.5, 1.3, 'False Negatives', ha='center', va='center', color='black', fontweight='bold', fontsize=14)
plt.text(1.5, 1.3, 'True Positives', ha='center', va='center', color='black', fontweight='bold', fontsize=14)

plt.title('Confusion Matrix\n(oGVHD Classification)', pad=20, fontsize=18, fontweight='bold')  # Increased title font size
plt.ylabel('True Label', labelpad=10, fontsize=16, fontweight='bold')  # Increased y-label font size
plt.xlabel('Predicted Label', labelpad=10, fontsize=16, fontweight='bold')  # Increased x-label font size

# Increase tick label font sizes
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(val_true, val_predictions)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\nTraining complete. Check the generated plots.")

# Calculate and print total training time
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\nTotal training time: {hours} hours, {minutes} minutes, {seconds} seconds")
