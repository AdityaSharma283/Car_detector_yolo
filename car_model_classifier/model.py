import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# Load label map
with open("car_model_classifier/labels.json", "r") as f:
    label_map = json.load(f)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load pretrained ResNet and replace classifier
def load_model(num_classes=len(label_map)):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model

model = load_model()

# Dummy: Random weights (for now) — training will come later
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_index = str(predicted.item())
    return label_map.get(class_index, "Unknown")
