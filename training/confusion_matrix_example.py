import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example data
# Let's say we have 100 samples:
# - 60 negative cases (class 0)
# - 40 positive cases (class 1)
y_true = np.array([0] * 60 + [1] * 40)  # 60 negative, 40 positive

# Example predictions (simulating a model with 85% accuracy)
# - 55 true negatives
# - 5 false positives
# - 6 false negatives
# - 34 true positives
y_pred = np.array([0] * 55 + [1] * 5 + [0] * 6 + [1] * 34)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

# Add labels for True Positives, False Positives, True Negatives, False Negatives
# Position the labels above the numbers
plt.text(0.5, 0.3, 'True Negatives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
plt.text(1.5, 0.3, 'False Positives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
plt.text(0.5, 1.3, 'False Negatives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
plt.text(1.5, 1.3, 'True Positives', ha='center', va='center', color='black', fontweight='bold', fontsize=10)

# Add the numbers below the labels
plt.text(0.5, 0.7, '(55)', ha='center', va='center', color='black', fontweight='bold', fontsize=12)
plt.text(1.5, 0.7, '(5)', ha='center', va='center', color='black', fontweight='bold', fontsize=12)
plt.text(0.5, 1.7, '(6)', ha='center', va='center', color='black', fontweight='bold', fontsize=12)
plt.text(1.5, 1.7, '(34)', ha='center', va='center', color='black', fontweight='bold', fontsize=12)

plt.title('Example Confusion Matrix\n(oGVHD Classification)', pad=20)
plt.ylabel('True Label', labelpad=10)
plt.xlabel('Predicted Label', labelpad=10)
plt.tight_layout()
plt.show()

# Print metrics
print("\nExample Metrics:")
print(f"Accuracy: {(55 + 34) / 100:.2%}")
print(f"Precision: {34 / (34 + 5):.2%}")
print(f"Recall: {34 / (34 + 6):.2%}")
print(f"Specificity: {55 / (55 + 5):.2%}") 