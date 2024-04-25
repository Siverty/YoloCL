import os
import shutil
import random

input_dir = "input/"

for case_folder in os.listdir(input_dir):
    case_path = os.path.join(input_dir, case_folder)
    output_dir = os.path.join("output", case_folder)

    os.makedirs(os.path.join(output_dir, "train", "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "label"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid", "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid", "label"), exist_ok=True)

    files = os.listdir(case_path)
    random.shuffle(files)
    split_index = int(0.7 * len(files))  # 70% for training, 30% for validation

    train_files = files[:split_index]
    valid_files = files[split_index:]

    for file in train_files:
        if file.endswith(".jpg"):
            shutil.copy(os.path.join(case_path, file), os.path.join(output_dir, "train", "image", file))
        elif file.endswith(".txt"):
            shutil.copy(os.path.join(case_path, file), os.path.join(output_dir, "train", "label", file))

    for file in valid_files:
        if file.endswith(".jpg"):
            shutil.copy(os.path.join(case_path, file), os.path.join(output_dir, "valid", "image", file))
        elif file.endswith(".txt"):
            shutil.copy(os.path.join(case_path, file), os.path.join(output_dir, "valid", "label", file))
