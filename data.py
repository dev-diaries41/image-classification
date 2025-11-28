# data.py
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        image_paths: List of file paths to the screenshot images.
        labels: List of integer labels corresponding to each image.
        transform: Optional torchvision transforms for preprocessing.
        """
        self.image_paths = image_paths
        self.labels = labels
        # If no transform is provided, use a default transform.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return {"image": image, "label": label}
