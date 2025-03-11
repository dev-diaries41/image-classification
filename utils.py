#!/usr/bin/env python3

import os

def read_dataset_dir(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The provided dataset directory '{dataset_dir}' does not exist or is not a directory.")

    file_paths = []
    labels = []
    # Iterate over all items in the dataset directory
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)  # Construct the full path
        if os.path.isdir(class_dir):
            class_dir_files = os.listdir(class_dir)
            for path in class_dir_files:
                file_paths.append(os.path.join(class_dir, path))
                labels.append(class_name)
    return file_paths, labels


def load_class_names(class_names_path):
    with open(class_names_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    return class_names
