from flask import Flask, request, render_template, send_from_directory
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define Tumor Classification Model
class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorClassifier, self).__init__()
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify for grayscale
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.fc = nn.Linear(512, num_classes)  # Classification layer

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Load Model
model = TumorClassifier()
model.load_state_dict(torch.load("tumor_classifier.pth", map_location="cpu"))
model.eval()

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

def generate_findings(prediction):
    findings = {
        "Glioma": "A Glioma is detected. Gliomas often appear in the frontal or temporal lobes of the brain.",
        "Meningioma": "A Meningioma is detected. These tumors are typically found near the brain's surface and can press against the skull.",
        "Pituitary": "A Pituitary tumor is detected. Pituitary tumors are located near the base of the brain, affecting hormonal balance.",
        "No Tumor": "No tumor detected. The brain scan appears healthy."
    }
    return findings[prediction]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Process the MRI image
        image = Image.open(file_path).convert("L")
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(image_tensor)

        _, predicted_class = torch.max(output, 1)
        tumor_types = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        prediction = tumor_types[predicted_class.item()]
        findings = generate_findings(prediction)

        return render_template("index.html", original=file_path, output=prediction, findings=findings)

    return render_template("index.html", original=None, output=None, findings=None)

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
