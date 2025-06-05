from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define ImageDataGenerators for rescaling (no augmentation for now)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Use validation_split=0.2 to separate the validation data

# Define directories for training, validation, and testing
train_dir = "/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/"  # Path to your data

# Training set
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=16,
    class_mode='binary',     # Binary classification (positive or negative)
    subset='training'        # Specifies that this is the training subset
)

print(f"Training data found: {train_generator.samples} samples")
print(f"Classes: {train_generator.class_indices}")

# Validation set
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),   # Resize images to 256x256
    batch_size=16,
    class_mode='binary',
    subset='validation'        # Specifies that this is the validation subset
)

print(f"Validation data found: {val_generator.samples} samples")

# Test set (should be separate from validation data)
# In this case, we just use the remaining data not used for training and validation
test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test data
test_generator = test_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=16,
    class_mode='binary',     # Binary classification
    shuffle=False             # Test data should not be shuffled
)

print(f"Test data found: {test_generator.samples} samples")

# Build the CNN Model
model = Sequential([
    # Convolutional layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),  # Input shape is 256x256x3
    MaxPooling2D(2, 2),

    # Convolutional layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Convolutional layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten and Fully Connected layer
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Number of epochs to train for
    validation_data=val_generator
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Plot accuracy and loss curves
plt.figure(figsize=(12, 6))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
