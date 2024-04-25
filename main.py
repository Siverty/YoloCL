# Description: main.py is the entry point for the application.
# It is used to split the data into training and validation sets and to further train the model.

import os


def install_requirements(requirements_file):
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()
        for req in requirements:
            req = req.strip()  # Remove leading/trailing whitespace
            if req.startswith('#'):
                continue  # Skip comments
            os.system(f'pip install {req}')


def main():

    from logic.data_handler.trainValSplit import split_data
    from logic.cl_yolo_train import continue_training

    # Define the root path of the input directory for YOLO data
    yolo_input_root = os.path.join(os.getcwd(), 'yolo_input')

    # Define the path to save the train and validation sets
    train_val_root = os.path.join(os.getcwd(), 'cl_data')

    # Split the data into training and validation sets directly from the input
    split_data(yolo_input_root, train_val_root, 0.7)

    # Set the path to the weights file
    weights_path = 'models/best.pt'

    # Set the path to the data configuration file
    data_yaml = 'datasets/data-chimp.yaml'

    continue_training(weights_path, data_yaml, 640, 16, 1, 0)


if __name__ == "__main__":
    install_requirements('requirements.txt')
    main()
