import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import ImageFolder
from flask import Flask, render_template, request
from PIL import Image
from featureExtractor import Features

# --------------------- Configuration & Transforms --------------------- #
# This transform should be the same as the one you used during training
transform = Compose([Resize((64, 64)), ToTensor()])

# Use the training dataset to determine class names and number of classes.
# The ImageFolder automatically sorts subfolder names alphabetically.
DATASET_DIR = './dataset'
dataset = ImageFolder(DATASET_DIR, transform=transform)
classes = dataset.classes
num_classes = len(classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features=Features().to(device)

# --------------------- Model Definition --------------------- #
class FingerprintToBloodGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

# --------------------- Load Model --------------------- #
model = FingerprintToBloodGroup().to(device)
# Load saved weights from 'blood_group.pth'
model.load_state_dict(torch.load('./models/best_model.pth', map_location=device))
model.eval()

# --------------------- Flask App Setup --------------------- #
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------- Helper Function --------------------- #
def preprocess_image(image_path):
    """
    Loads an image, applies the transformation, and adds a batch dimension.
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image)  # Resulting shape: [C, H, W]
    image = image.unsqueeze(0)  # Now shape: [1, C, H, W]
    return image.to(device)

# --------------------- Routes --------------------- #
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != "":
            filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Preprocess the image and perform prediction
            img_tensor = preprocess_image(image_path)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, dim=1)
                prediction = classes[pred.item()]

    return render_template('index.html', prediction=prediction, image_path=image_path)

# --------------------- Main --------------------- #
if __name__ == '__main__':
    app.run(debug=True)
