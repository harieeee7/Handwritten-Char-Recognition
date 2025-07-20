import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# ----------------- Setup -----------------
st.set_page_config(page_title="Handwritten Character Recognition")
st.title("✍️ Handwritten Character Recognition")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model file path and Google Drive download link
MODEL_PATH = "emnist_cnn.pth"
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1dp5DCteGewgBB28fp5K402HMQmqkyr5G"

# Automatically download the model if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)

# ----------------- Model Definition -----------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)  # 26 letters A-Z

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize and load model
model = CNNModel().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))

    model.eval()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------- Helper Functions -----------------
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = 255 - img_array  # Invert colors
    img_array = img_array / 255.0  # Normalize
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

def predict_character(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        label = chr(predicted.item() + ord('A'))
    return label

# ----------------- Streamlit UI -----------------
uploaded_file = st.file_uploader("Upload an image of a handwritten character", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        prediction = predict_character(input_tensor)
        st.success(f"Predicted Character: **{prediction}**")
