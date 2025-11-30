import os
import shutil
import random

def read_dataset_dir(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The provided dataset directory '{dataset_dir}' does not exist or is not a directory.")

    file_paths = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name) 
        if os.path.isdir(class_dir):
            class_dir_files = os.listdir(class_dir)
            for filename in class_dir_files:
                file_paths.append(os.path.join(class_dir, filename))
                labels.append(class_name)
    return file_paths, labels, class_names


def get_new_dirname(dir_path: str, prefix: str):
    os.makedirs(dir_path, exist_ok=True)
    highest = -1
    dirs = os.listdir(dir_path)
    for d in dirs:
        if not d.startswith(prefix):
            continue
        try:
            n = int(d[len(prefix):])
            highest = max(highest, n)
        except ValueError:
            continue
    return os.path.join(dir_path, prefix + str(highest + 1))


def get_new_filename(dir_path: str, prefix: str, ext):
    os.makedirs(dir_path, exist_ok=True)
    highest = -1
    for f in os.listdir(dir_path):
        if f.startswith(prefix):
            base = os.path.splitext(f)[0]
            try:
                n = int(base[len(prefix):])
                if n > highest:
                    highest = n
            except ValueError:
                continue
    return os.path.join(dir_path, f"{prefix}{highest + 1}{ext}")


def split_dataset(dataset_dir: str, size: int, dirname: str, seed: int = 42):
    random.seed(seed)
    new_dataset_dir = os.path.join(os.path.dirname(dataset_dir), dirname)
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            class_filenames = os.listdir(class_dir)
            indices = random.sample(range(len(class_filenames)), size)
            selected = [class_filenames[n] for n in indices]
            for filename in selected:
                filepath = os.path.join(class_dir, filename)
                try:
                    destination_dir = os.path.join(new_dataset_dir, class_name)
                    os.makedirs(destination_dir, exist_ok=True)
                    shutil.move(filepath, destination_dir)
                except Exception as e:
                    # print(e)
                    continue

                

