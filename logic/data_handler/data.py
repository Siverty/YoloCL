import os
import shutil

# Get the root folder path to add to the input and output directories
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))

# Define the root path of the input and output directories.
input_root = os.path.join(root_dir, 'yolo_input')
output_root = os.path.join(root_dir, 'yolo_output')


# Function to create the necessary output directories.
def create_output_dirs(case_number):
    output_case_dir = os.path.join(output_root, f'case{case_number}')
    os.makedirs(os.path.join(output_case_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_case_dir, 'labels'), exist_ok=True)


# Function to copy files to the output directories instead of moving them.
def copy_files_to_output(case_number, image_files, label_files):
    output_image_dir = os.path.join(output_root, f'case{case_number}', 'images')
    output_label_dir = os.path.join(output_root, f'case{case_number}', 'labels')

    for image_file in image_files:
        shutil.copy(image_file, os.path.join(output_image_dir, os.path.basename(image_file)))
    for label_file in label_files:
        shutil.copy(label_file, os.path.join(output_label_dir, os.path.basename(label_file)))

    print(f"{len(image_files) + len(label_files)} files copied to {output_image_dir} / {output_label_dir}")


# Function to process each case.
def process_case(case_path, case_number):
    print(f"Processing case: {case_number}")
    create_output_dirs(case_number)
    image_files = []
    label_files = []

    # Function to find and pair image-label files.
    def find_and_pair_files(subdir):
        images = [os.path.join(subdir, 'images', f) for f in os.listdir(os.path.join(subdir, 'images')) if
                  f.lower().endswith('.jpg')]
        labels = [os.path.join(subdir, 'labels', f) for f in os.listdir(os.path.join(subdir, 'labels')) if
                  f.lower().endswith('.txt')]
        print(f"Looking in: {os.path.join(subdir, 'images')}")  # Debug print: where it's looking for images
        print(f"Looking in: {os.path.join(subdir, 'labels')}")  # Debug print: where it's looking for labels
        print(f"Found {len(images)} images and {len(labels)} labels in {subdir}")  # Debug print
        paired_images = []
        paired_labels = []
        for img in images:
            lbl = os.path.join(subdir, 'labels', os.path.basename(img).rsplit('.', 1)[0] + '.txt')
            if os.path.exists(lbl):
                paired_images.append(img)
                paired_labels.append(lbl)
            else:
                print(f"Missing label for image: {img}")  # Debug print
        return paired_images, paired_labels

    print(f"Checking folder structure for case {case_number}")
    # Handle the cases with images/labels subfolders directly under the case folder.
    if os.path.exists(os.path.join(case_path, 'images')) and os.path.exists(os.path.join(case_path, 'labels')):
        image_files, label_files = find_and_pair_files(case_path)
    # Handle cases with train/valid subfolders.
    elif os.path.exists(os.path.join(case_path, 'train')) or os.path.exists(os.path.join(case_path, 'valid')):
        for folder in ['train', 'valid']:
            if os.path.exists(os.path.join(case_path, folder)):
                paired_images, paired_labels = find_and_pair_files(os.path.join(case_path, folder))
                image_files.extend(paired_images)
                label_files.extend(paired_labels)
    # Handle the cases with a single data folder.
    elif os.path.exists(os.path.join(case_path, 'data')):
        data_files = [os.path.join(case_path, 'data', f) for f in os.listdir(os.path.join(case_path, 'data'))]
        image_files = [f for f in data_files if f.endswith('.jpg')]
        label_files = [f for f in data_files if f.endswith('.txt')]
    else:
        print(f"Unknown folder structure for case {case_number}. Skipping.")
        return

    # Move the paired files to the output directories.
    copy_files_to_output(case_number, image_files, label_files)


# Process each case.
for case_number in range(1, 4):  # Assuming we have three cases; you can change this range as needed.
    case_path = os.path.join(input_root, f'case{case_number}')
    process_case(case_path, case_number)
