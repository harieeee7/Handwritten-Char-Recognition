import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import os
import urllib.request

# ----------------------------
# Google Drive Model Download
# ----------------------------

MODEL_PATH = "emnist_cnn.pth"
GDRIVE_FILE_ID = "1dp5DCteGewgBB28fp5K402HMQmqkyr5G"  # ✅ Your model
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {e}")

# ----------------------------
# CNN Model Definition
# ----------------------------

class EMNISTModel(nn.Module):
    def __init__(self):
        super(EMNISTModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 26),  # 26 English letters
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# ----------------------------
# Load Model
# ----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMNISTModel().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    st.error("Model file not found. Please check Google Drive link.")

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("✍️ Handwritten Character Recognition")
st.markdown("Upload a **28x28 grayscale** image of a handwritten **English letter (A-Z)**.")

uploaded_file = st.file_uploader("Upload a handwritten character...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Invert and resize
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    # Normalize & convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    if st.button("Predict"):
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, 1).item()
            predicted_letter = chr(pred + 65)  # EMNIST starts from 'A' as index 0
            st.success(f"🧠 Predicted Character: **{predicted_letter}**")
