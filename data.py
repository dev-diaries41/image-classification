import torchvision.transforms as transforms
import os
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from utils import read_dataset_dir

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


def get_dataset(path: str):
    file_paths, labels, class_names = read_dataset_dir(path)
    numbered_labels = [class_names.index(label) for label in labels]
    return ImageDataset(file_paths, numbered_labels)  


def preprocess_cub_dataset(dataset_dir: str):
    train_dir = os.path.join(os.path.dirname(dataset_dir), f"{os.path.basename(dataset_dir)}_train")
    validation_dir = os.path.join(os.path.dirname(dataset_dir), f"{os.path.basename(dataset_dir)}_validation")

    images_dir = os.path.join(dataset_dir, 'images')
    imageids_map_path = os.path.join(dataset_dir, 'images.txt')
    classids_map_path = os.path.join(dataset_dir, 'image_class_labels.txt')
    splits_path = os.path.join(dataset_dir, 'train_test_split.txt')
    bounding_boxes_path = os.path.join(dataset_dir, 'bounding_boxes.txt')

    valid_dataset = all([os.path.isfile(path) for path in [imageids_map_path, classids_map_path, splits_path, bounding_boxes_path]])

    if not valid_dataset:
        raise ValueError("Missing required files in dataset")
    
    id_to_path: dict[int, str] = {}
    id_to_split_type: dict[int, int] = {}
    id_to_boxes: dict[int, tuple[float, float, float, float]] = {}

    with open(imageids_map_path, "r") as f:
        for line in f:
            image_id, image_relative_path = line.strip().split()
            id_to_path[int(image_id)] = os.path.join(images_dir, image_relative_path)

    with open(splits_path, "r") as f:
        for line in f:
            image_id, is_training_image = line.strip().split()
            id_to_split_type[int(image_id)] = int(is_training_image)
    
    with open(bounding_boxes_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            box = [float(i) for i in parts[-4:]]
            id_to_boxes[int(parts[0])] = tuple(box)
    
    
    for image_id, is_training_image in tqdm.tqdm(id_to_split_type.items(), total=len(id_to_split_type)):
        image_path = id_to_path[int(image_id)]
        class_name = os.path.basename(os.path.dirname(image_path))
        filename = os.path.basename(image_path)
        destination_path = os.path.join(train_dir, class_name, filename) if int(is_training_image) == 1 else os.path.join(validation_dir, class_name, filename)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        try:
            x, y, w, h = id_to_boxes[image_id]
            box = (x, y, x + w, y + h)  #            
            cropped = Image.open(image_path).crop(box)
            cropped.save(destination_path)
        except Exception as e:
            print(e)
            continue
    
    