from datasets.emnist_csv_dataset import EMNISTCSV
from torchvision import transforms
from models.model import EMNISTModel
from utils import get_device
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

device = get_device()

transform = transforms.Compose([
    transforms.ToPILImage(),            # Convert NumPy array to PIL
    transforms.ToTensor(),              # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize
])


train_dataset = EMNISTCSV('data/emnist-letters-train.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = EMNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "emnist_cnn.pth")
