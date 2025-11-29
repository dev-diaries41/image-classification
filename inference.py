# inference.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import ImageClassifier

def load_image(image_path):
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

def inference(model, device, image_path, class_names):
    model.to(device)
    model.eval()
    image = load_image(image_path)
    image = image.unsqueeze(0).to(device)  # Add batch dimension.
    with torch.no_grad():
        outputs = model(image)
    
    predicted_index = outputs.argmax(dim=1).item()
    predicted_class = class_names[predicted_index]
    return predicted_class