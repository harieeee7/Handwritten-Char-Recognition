import streamlit as st
from PIL import Image
import torch
from models.model import EMNISTModel
from utils import get_class_names, get_device
from torchvision import transforms

# Setup
device = get_device()
model = EMNISTModel().to(device)
model.load_state_dict(torch.load("emnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class_names = get_class_names()

# Streamlit UI
st.title("🖋️ Handwritten Character Recognition (A–Z)")
st.markdown("Upload a **28x28 grayscale image** of a handwritten letter (A–Z).")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", width=150)

    # Add Predict button
    if st.button("Predict"):
        img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            predicted_char = class_names[pred.item()]
            st.success(f"🧠 Predicted Character: **{predicted_char}**")
