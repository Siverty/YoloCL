#  Description: This script is used to extract and split the data into training and validation sets.

import os
from sklearn.model_selection import train_test_split


def split_data(input_data, prepared_data, train_ratio):
    def gather_files(case_path):
        image_files, label_files = [], []
        for root, dirs, files in os.walk(case_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
                elif file.lower().endswith('.txt'):
                    label_files.append(os.path.join(root, file))
        return image_files, label_files

    def link_files(files, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for file_path in files:
            target_path = os.path.join(dest_dir, os.path.basename(file_path))
            if os.path.exists(target_path):
                os.remove(target_path)
            os.link(file_path, target_path)

    # Iterate over each case in the input root
    for case_dir in os.listdir(input_data):
        full_case_path = os.path.join(input_data, case_dir)
        if os.path.isdir(full_case_path):  # Make sure it's a directory
            image_files, label_files = gather_files(full_case_path)

            # Ensure the images and labels have a one-to-one correspondence
            image_files.sort()
            label_files.sort()

            # Split data into training and validation sets
            train_images, valid_images, train_labels, valid_labels = train_test_split(
                image_files, label_files, train_size=train_ratio, random_state=42)

            # Link training images and labels to the train directory
            train_images_dir = os.path.join(prepared_data, case_dir, 'train', 'images')
            train_labels_dir = os.path.join(prepared_data, case_dir, 'train', 'labels')
            link_files(train_images, train_images_dir)
            link_files(train_labels, train_labels_dir)

            # Link validation images and labels to the valid directory
            valid_images_dir = os.path.join(prepared_data, case_dir, 'valid', 'images')
            valid_labels_dir = os.path.join(prepared_data, case_dir, 'valid', 'labels')
            link_files(valid_images, valid_images_dir)
            link_files(valid_labels, valid_labels_dir)

            print(f"Case {case_dir}: {len(train_images)} training and {len(valid_images)} validation images."
                  f"\n Using a {train_ratio}% training ratio split.")
