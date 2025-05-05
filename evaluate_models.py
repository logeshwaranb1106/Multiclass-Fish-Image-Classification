import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_data
import os

# Load test data
_, _, test_data = load_data()

# Get true labels
true_labels = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Folder containing saved models
model_dir = 'models'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

print("\n🔍 Evaluating Models:\n")

for model_file in model_files:
    print(f"Evaluating {model_file}...")

    model = tf.keras.models.load_model(os.path.join(model_dir, model_file))

    # Predict
    preds = model.predict(test_data)
    predicted_labels = np.argmax(preds, axis=1)

    # Metrics
    report = classification_report(true_labels, predicted_labels, target_names=class_labels, output_dict=True)
    matrix = confusion_matrix(true_labels, predicted_labels)

    acc = report['accuracy']
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"📊 Precision (macro avg): {report['macro avg']['precision']:.4f}")
    print(f"📊 Recall (macro avg): {report['macro avg']['recall']:.4f}")
    print(f"📊 F1-score (macro avg): {report['macro avg']['f1-score']:.4f}")
    print(f"🧩 Confusion Matrix:\n{matrix}")
    print("-----------------------------------------------------")
