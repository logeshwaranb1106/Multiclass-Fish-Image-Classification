import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# First Streamlit command
st.set_page_config(page_title="Fish Classifier")

# Load model with caching
@st.cache_resource
def load_model():
    st.write("Loading model...")  # Debugging line
    return tf.keras.models.load_model(r"C:\Users\loges\Downloads\MobileNet.keras")

model = load_model()

class_labels = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))      # Resize to 224x224 for MobileNet
    image = np.array(image) / 255.0       # Normalize pixel values between 0 and 1
    if image.shape[-1] == 4:              # Remove alpha channel if present (RGBA → RGB)
        image = image[..., :3]
    return np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)


# Streamlit app interface
st.title("🐟 Fish Image Classifier")
st.write("Upload an image of a fish or seafood item to classify it.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

# If image is uploaded: show and predict
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded successfully!")  # Debugging line

#Make prediction
    with st.spinner("Classifying..."):
        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)[0] # Get prediction probabilities

        # Get the top 3 predictions
        # Gets the indexes of the top 3 predicted classes by sorting the probabilities in descending order.
        top3 = predictions.argsort()[-3:][::-1] 

        # Display the top 3 predictions
        st.subheader("Top Predictions:")
        for i in top3:
            st.write(f"**{class_labels[i]}**: {predictions[i]*100:.2f}%")
            
        # Highlight the best prediction
        # Display the confidence score of the top prediction
        st.subheader("Top Prediction:")
        top_label = class_labels[top3[0]]
        top_score = predictions[top3[0]] * 100
        st.write(f"**{top_label}**")

# Error handling 
else:
    st.write("Please upload a fish image to get a prediction.")