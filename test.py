from utils import read_dataset_dir
file_paths, labels, class_names = read_dataset_dir("assets/dataset")
numbered_labels = [class_names.index(label) for label in labels]

for path, label in zip(file_paths, numbered_labels):
    print(f"Path: {path} | Label: {label}")