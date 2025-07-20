from datasets.emnist_csv_dataset import EMNISTCSV
from torchvision import transforms
from torch.utils.data import DataLoader
from models.model import EMNISTModel
from utils import get_device
import torch

device = get_device()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = EMNISTCSV('data/emnist-letters-test.csv', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = EMNISTModel().to(device)
model.load_state_dict(torch.load("emnist_cnn.pth", map_location=device))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
