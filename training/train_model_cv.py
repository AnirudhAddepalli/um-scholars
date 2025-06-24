import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

# Define the data directory
train_dir = '/Users/ani/UM Scholars/oGVHD_Tool/um-scholars/data_converted/'

# Create base ImageDataGenerator for rescaling
base_datagen = ImageDataGenerator(rescale=1./255)

# Get all image paths and labels
image_paths = []
labels = []
for class_name in ['Fluorescein_Stained_Images_Negative_oGVHD', 'Fluorescein_Stained_Images_Positive_oGVHD']:
    class_dir = os.path.join(train_dir, class_name)
    for img_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, img_name))
        labels.append(1 if class_name == 'Fluorescein_Stained_Images_Positive_oGVHD' else 0)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("\n=== Dataset Analysis ===")
print(f"Total samples: {len(image_paths)}")
print(f"Class distribution: {np.bincount(labels)}")

# Initialize K-fold cross-validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store metrics for each fold
fold_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'auc_roc': [],
    'specificity': []
}

# Train and evaluate for each fold
for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths), 1):
    print(f"\n=== Training Fold {fold}/{n_splits} ===")
    
    # Create data generators for this fold
    train_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Create model for this fold
    model = Sequential([
        # First conv block
        Conv2D(16, (3, 3), activation='relu', input_shape=(256, 192, 3)),
        MaxPooling2D(2, 2),
        
        # Second conv block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Calculate class weights for this fold
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = {
        0: len(train_labels) / (2 * class_counts[0]),
        1: len(train_labels) / (2 * class_counts[1])
    }
    print(f"Fold {fold} class weights:", class_weights)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_datagen.flow_from_directory(
            train_dir,
            target_size=(256, 192),
            batch_size=16,
            class_mode='binary',
            subset='training'
        ),
        validation_data=val_datagen.flow_from_directory(
            train_dir,
            target_size=(256, 192),
            batch_size=16,
            class_mode='binary',
            subset='validation'
        ),
        epochs=20,
        class_weight=class_weights
    )
    
    # Evaluate model
    val_predictions = model.predict(val_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 192),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    ))
    val_predictions_binary = (val_predictions > 0.5).astype(int)
    val_true = labels[val_idx]
    
    # Calculate metrics
    fold_metrics['accuracy'].append(np.mean(val_predictions_binary == val_true))
    fold_metrics['precision'].append(precision_score(val_true, val_predictions_binary))
    fold_metrics['recall'].append(recall_score(val_true, val_predictions_binary))
    fold_metrics['f1'].append(f1_score(val_true, val_predictions_binary))
    fold_metrics['auc_roc'].append(roc_auc_score(val_true, val_predictions))
    fold_metrics['specificity'].append(recall_score(val_true, val_predictions_binary, pos_label=0))
    
    # Plot confusion matrix for this fold
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(val_true, val_predictions_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative oGVHD', 'Predicted Positive oGVHD'],
                yticklabels=['Actual Negative oGVHD', 'Actual Positive oGVHD'])
    
    # Add labels for True Positives, False Positives, True Negatives, False Negatives
    plt.text(0.5, 0.3, 'True Negatives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
    plt.text(1.5, 0.3, 'False Positives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
    plt.text(0.5, 1.3, 'False Negatives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
    plt.text(1.5, 1.3, 'True Positives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
    
    plt.title(f'Confusion Matrix - Fold {fold}\n(oGVHD Classification)', pad=20)
    plt.ylabel('True Label', labelpad=10)
    plt.xlabel('Predicted Label', labelpad=10)
    plt.tight_layout()
    plt.show()

# Print average metrics across all folds
print("\n=== Cross-Validation Results ===")
print(f"Average Accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
print(f"Average Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
print(f"Average Recall: {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
print(f"Average F1-Score: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
print(f"Average AUC-ROC: {np.mean(fold_metrics['auc_roc']):.4f} ± {np.std(fold_metrics['auc_roc']):.4f}")
print(f"Average Specificity: {np.mean(fold_metrics['specificity']):.4f} ± {np.std(fold_metrics['specificity']):.4f}")

# Plot average metrics
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
metrics_values = [np.mean(fold_metrics[m.lower()]) for m in metrics_names]
metrics_std = [np.std(fold_metrics[m.lower()]) for m in metrics_names]

plt.figure(figsize=(12, 6))
plt.bar(metrics_names, metrics_values, yerr=metrics_std, capsize=5)
plt.title('Average Metrics Across All Folds')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nTraining complete. Check the generated plots.") 