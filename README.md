# 🐟 Multiclass Fish Image Classification

A deep learning project to classify fish species from images using both custom CNNs and pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0). Includes a visually appealing Streamlit web app for real-time prediction.

---
## 📁 Dataset

The dataset contains categorized fish images in `train/`, `val/`, and `test/` folders.

🔗 [📂 View Dataset Folder on Google Drive](https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd?usp=sharing)

---


## 🌟 Features

- ✅ Trained CNN from scratch
- 🧠 Transfer learning using 5 popular pre-trained models
- 📊 Model evaluation with precision, recall, F1-score, and confusion matrix
- 🖼️ Training history plots (accuracy & loss)
- 💡 Visually designed Streamlit app with real-time fish species prediction
- 📁 Clean project structure and modular code
- 🚀 Ready for GitHub deployment

---

## 🧠 Skills You’ll Practice

- Deep Learning (CNNs & Transfer Learning)
- Data Augmentation
- Model Evaluation
- Streamlit Deployment
- Visualization (Matplotlib)
- Model Saving & Loading (`.h5`)
- GitHub Project Structure

---

## 📁 Project Structure

fish-classifier/
│
├── app/
│ └── app.py # Streamlit app
│
├── models/
│ └── vgg16_fish_model.h5 # Trained best model
│
├── scripts/
│ ├── train_cnn.py
│ ├── train_transfer.py
│ ├── evaluate_models.py
│ │
├── Dataset/
│ ├── train/
│ ├── val/
│ └── test/
│
├── load_data.py
├── requirements.txt
└── README.md




---

## 📦 Dataset

The dataset contains images of fish species categorized into 11 classes and organized into `train`, `val`, and `test` folders.

- **Loading:** Handled using `ImageDataGenerator` from TensorFlow
- **Preprocessing:** Rescaling and augmentation (rotation, zoom, flipping)

---

## 🚀 How to Use

### 1. 🔧 Install Dependencies
```bash
pip install -r requirements.txt

### 2.Train Models
python scripts/train_cnn.py
python scripts/train_transfer_models.py

### 3. Evaluate
python scripts/evaluate_models.py

### Run Streamlit app
cd app
streamlit run app.py

