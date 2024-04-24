# Description: This script redistributes the dataset into train and validation sets.

import os
import shutil

# Define the source directories
source_dirs = ['case1', 'case2', 'case3']

# Define the destination directories
dest_dirs = {
    'train_images': 'output/train/images',
    'train_labels': 'output/train/labels',
    'validation_images': 'output/validation/images',
    'validation_labels': 'output/validation/labels'
}

# Define the image file extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Iterate over the source directories
for source_dir in source_dirs:
    for filename in os.listdir(source_dir):
        # Get the file extension
        _, file_extension = os.path.splitext(filename)

        # Check if the file is an image
        if file_extension in image_extensions:
            # If it's an image, move it to the images directories
            if 'train' in source_dir:
                shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dirs['train_images'], filename))
            else:
                shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dirs['validation_images'], filename))
        elif file_extension == '.txt':
            # If it's a label, move it to the labels directories
            if 'train' in source_dir:
                shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dirs['train_labels'], filename))
            else:
                shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dirs['validation_labels'], filename))

