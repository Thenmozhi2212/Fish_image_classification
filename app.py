import streamlit as st
from tensorflow import keras
import pickle
from PIL import Image
import numpy as np
import os

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "fish_classification_model.keras"
LABELS_PATH = "class_lables.pkl"

# -----------------------------
# Load model
# -----------------------------
if os.path.exists(MODEL_PATH):
    try:
        model = keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        # Auto-detect input shape
        input_shape = model.input_shape[1:4]  # (height, width, channels)
        st.write(f"Model input shape: {model.input_shape}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error(f"❌ Model file not found: {MODEL_PATH}")

# -----------------------------
# Load class labels
# -----------------------------
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "rb") as f:
            class_labels = pickle.load(f)
        st.success("✅ Class labels loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading class labels: {e}")
else:
    st.error(f"❌ Class labels file not found: {LABELS_PATH}")

# -----------------------------
# Prediction function
# -----------------------------
def predict_fish(image: Image.Image):
    # Get model input size
    height, width, channels = input_shape

    # Convert uploaded image to RGB
    img = image.convert("RGB")

    # Resize exactly to model input
    img = img.resize((width, height))

    # Normalize and add batch dimension
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

    # Make prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name = class_labels[class_idx]
    confidence = float(np.max(preds))

    return class_name, confidence

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🐟 Fish Image Classification")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            class_name, confidence = predict_fish(image)
            st.write(f"**Prediction:** {class_name}")
            st.write(f"**Confidence:** {confidence:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

