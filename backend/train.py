import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.models import resnet18

# Define custom dataset loader for classification
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Define tumor categories
        self.categories = ["glioma", "meningioma", "pituitary", "no_tumor"]
        self.label_map = {category: idx for idx, category in enumerate(self.categories)}

        for category in self.categories:
            img_dir = os.path.join(root_dir, category)
            if not os.path.exists(img_dir):
                continue  # Skip if the folder does not exist

            img_filenames = os.listdir(img_dir)
            for filename in img_filenames:
                self.image_paths.append(os.path.join(img_dir, filename))
                self.labels.append(self.label_map[category])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
train_dataset = BrainTumorDataset("C:/Users/KIIT0001/Desktop/Brain-Tumor-Detection/dataset/training/", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define CNN-based classification model
class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorClassifier, self).__init__()
        base_model = resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify for grayscale
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.fc = nn.Linear(512, num_classes)  # Classification layer

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TumorClassifier().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save trained model
torch.save(model.state_dict(), "tumor_classifier.pth")
print("Model training complete and saved as tumor_classifier.pth")
