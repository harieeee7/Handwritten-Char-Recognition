import torch
from PIL import Image
from torchvision import transforms
from models.model import EMNISTModel
from utils import get_device, get_class_names

device = get_device()
class_names = get_class_names()

model = EMNISTModel().to(device)
model.load_state_dict(torch.load("emnist_cnn.pth", map_location=device))
model.eval()

img_path = "samples/A_0.png"  # Your 28x28 grayscale image
img = Image.open(img_path).convert("L")
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    print("Predicted Letter:", class_names[predicted.item()])
