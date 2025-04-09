import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image

# Define the Generator (for GAN)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Load Dataset
def get_data_loader(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the GAN
def train_gan(data_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            fake_images = generator(real_images)
            loss = criterion(fake_images, real_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(generator.state_dict(), "mri_generator.pth")
    print("GAN Model Saved!")

# Load MRI data and train GAN
data_loader = get_data_loader("C:/Users/KIIT0001/Desktop/Brain-Tumor-Detection/dataset/training/")
train_gan(data_loader)
