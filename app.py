import streamlit as st
from tensorflow import keras
import pickle
from PIL import Image
import numpy as np
import os
import gdown

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fish_classification_model_rgb.keras")
LABELS_PATH = os.path.join(BASE_DIR, "class_labels (2).pkl")

# -----------------------------
# Google Drive File IDs
# -----------------------------
MODEL_DRIVE_ID = "1Zr98mqhsWx6kqbvIteqVyXjv6D1Ex0yw"      
LABELS_DRIVE_ID = "1JPU6pU6vaIt4Va03svFO-ZR9lnT72rHw"   

# -----------------------------
# Function to download files
# -----------------------------
def download_file_from_drive(drive_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, output_path, quiet=False)
        
# Download model and labels if missing
# -----------------------------
download_file_from_drive(MODEL_DRIVE_ID, MODEL_PATH)
download_file_from_drive(LABELS_DRIVE_ID, LABELS_PATH)

# -----------------------------
# Load model
# -----------------------------
try:
    model = keras.models.load_model(MODEL_PATH)
    input_shape = model.input_shape[1:4]  # (height, width, channels)
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# -----------------------------
# Load class labels
# -----------------------------
try:
    with open(LABELS_PATH, "rb") as f:
        class_labels = pickle.load(f)
    
except Exception as e:
    st.error(f"❌ Error loading class labels: {e}")

# -----------------------------
# Prediction function
# -----------------------------
def predict_fish(image: Image.Image):
    # Resize image to model input
    height, width, channels = input_shape
    img = image.convert("RGB").resize((width, height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return class_labels[class_idx], confidence

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🐟 Fish Image Classification")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        try:
            class_name, confidence = predict_fish(image)
            st.success(f"**Prediction:** {class_name}")
            st.info(f"**Confidence:** {confidence:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
