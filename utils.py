import os

def read_dataset_dir(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The provided dataset directory '{dataset_dir}' does not exist or is not a directory.")

    file_paths = []
    labels = []
    class_names = os.listdir(dataset_dir)

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
    if len(dirs) == 0:
        return os.path.join(dir_path, prefix + "0")
    for d in dirs:
        if not d.startswith(prefix):
            continue
        n = int(d.strip(prefix))
        highest = n if n > highest else highest
    return os.path.join(dir_path, prefix + str(highest + 1))


def get_new_filename(dir_path: str, prefix: str, ext):
    os.makedirs(dir_path, exist_ok=True)
    highest = -1
    files = os.listdir(dir_path)
    if len(files) == 0:
        return os.path.join(dir_path,  prefix + "0" + ext)
    for f in files:
        if not f.startswith(prefix):
            continue
        n = int(os.path.splitext(f)[0].strip(prefix))
        highest = n if n > highest else highest
    return os.path.join(dir_path, prefix + str(highest + 1) + ext)
