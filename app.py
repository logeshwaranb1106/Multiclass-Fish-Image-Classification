import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from load_data import load_data
import matplotlib.pyplot as plt

# Load model
MODEL_PATH = r"L:\Guvi\Project 5\models\mobilenetv2_fish_model.h5"  # replace with your best model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
train_data, _, _ = load_data()
class_names = list(train_data.class_indices.keys())

# Page config
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")

# Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #0066cc;
        }
        .subheader {
            text-align: center;
            font-size: 18px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"> Fish Species Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload a fish image to predict its species</div>', unsafe_allow_html=True)
st.markdown("---")

# Upload section
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image display
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    predicted_label = class_names[predicted_index]

    # Result
    st.markdown("### 🎯 Prediction Result")
    st.success(f"**Predicted Species:** `{predicted_label}`")
    st.info(f"**Confidence:** {confidence:.2%}")

    # Confidence bar chart
    st.markdown("### 🔍 Prediction Confidence by Class")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(class_names, predictions[0], color='skyblue')
    ax.set_xlim([0, 1])
    ax.set_xlabel("Confidence")
    ax.invert_yaxis()
    st.pyplot(fig)

else:
    st.markdown("> 🧠 Tip: Upload a clear fish image to see predictions.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using TensorFlow and Streamlit")
