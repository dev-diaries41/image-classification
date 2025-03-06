#!/usr/bin/env python3

import os
import sys
import shutil
import os

def read_dataset_dir(dataset_dir):
    """
    Reads a dataset directory and returns a list of full file paths for all files within it.
    
    Args:
        dataset_dir (str): The path to the dataset directory.
        
    Returns:
        List[str]: A list of full file paths for each file in the dataset directory.
    
    Raises:
        ValueError: If the provided directory does not exist or is not a directory.
    """
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The provided dataset directory '{dataset_dir}' does not exist or is not a directory.")

    file_paths = []
    # Iterate over all items in the dataset directory
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isfile(item_path):
            file_paths.append(item_path)
        elif os.path.isdir(item_path):
            dir_files = read_dataset_dir(item_path)
            for path in dir_files:
                file_paths.append(path)
    return file_paths


def get_labels(image_paths: list):
    labels = []

    for path in image_paths:
        base_name, _ = os.path.splitext(path)
        if base_name.endswith('_1'):
            labels.append(1)
        elif base_name.endswith('_0'):
            labels.append(0)
        else:
            labels.append(2)
    return labels


def load_class_names(class_names_path):
    with open(class_names_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    return class_names


def sort_images():
    # Get the target directory from command-line argument; default to current directory.
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

    # Check if the provided directory exists.
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    # Create the 'twitter' and 'reddit' subdirectories if they do not already exist.
    twitter_dir = os.path.join(target_dir, 'twitter')
    reddit_dir = os.path.join(target_dir, 'reddit')
    os.makedirs(twitter_dir, exist_ok=True)
    os.makedirs(reddit_dir, exist_ok=True)

    # Iterate over all items in the target directory.
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        
        # Process only regular files.
        if os.path.isfile(item_path):
            # Remove file extension before checking the suffix
            base_name, _ = os.path.splitext(item)
            if base_name.endswith('_1'):
                shutil.move(item_path, twitter_dir)
                print(f"Moved '{item}' to '{twitter_dir}/'")
            elif base_name.endswith('_0'):
                shutil.move(item_path, reddit_dir)
                print(f"Moved '{item}' to '{reddit_dir}/'")

if __name__ == '__main__':
    sort_images()
