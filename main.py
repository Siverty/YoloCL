# Description: main.py is the entry point for the application.
# It is used to split the data into training and validation sets and to further train the model.

# Be sure to run this script in a .venv environment with the required packages installed.
# The pytorch && CUDA installations will otherwise throw exotics errors.

# ToDos:
# 1. add testing capabilities
# 2. add logging
# 3. add parameters for continual learning
# 4. -------------------------------------

import os


def install_requirements(requirements_file):
    """
    Install required packages listed in the requirements.txt file, and upgrade pip.
    """
    os.system('python -m pip install --upgrade pip')
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()
        for req in requirements:
            req = req.strip()  # Remove leading/trailing whitespace
            if req.startswith('#'):
                continue  # Skip comments
            os.system(f'pip install {req}')


def train():
    """
    Function to execute the training process on existing weights.
    """
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

    # Continue training the model
    continue_training(weights_path, data_yaml, 640, 16, 50, 0)


if __name__ == "__main__":
    # Install required packages before executing main function
    install_requirements('requirements.txt')
    # Execute train function
    train()
