# Description: main.py is the entry point for the application.
# It is used to split the data into training and validation sets and to further train the model.

# Be sure to run this script in a .venv environment with the required packages installed.
# The pytorch && CUDA installations will otherwise throw exotics errors.

# TODO:
# 1. add testing capabilities
# 2. add logging
# 3. add parameters for continual learning
# 4. run directory add
# 5. make project a parameter that is defined elsewhere

import os

# Select what project to use
project = 'CHILL'
experiment_name = project + '_cl'


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
            os.system(f'pip install --no-cache {req}')


def split_data():
    """
    Split the raw data into training and validation sets.
    """
    from continueTraining.logic.data_handler.trainValSplit import split_data

    # Define the root path of the input directory for YOLO data
    yolo_input_root = os.path.join(os.getcwd(), 'data', project, 'yolo_input')

    # Define the path to save the train and validation sets
    train_val_root = os.path.join(os.getcwd(), 'data', project, 'cl_data')

    # Split the data into training and validation sets directly from the input
    split_data(yolo_input_root, train_val_root, 0.7)


def create_mlflow_experiment():
    """
    Create a new MLFlow experiment if it does not exist.
    """
    import mlflow

    # Set MLFlow tracking URI
    mlflow.set_tracking_uri('http://localhost:5000')

    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Create the experiment if it does not exist
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created.")
    else:
        print(f"Experiment '{experiment_name}' already exists.\nContinuing training...")


def train():
    """
    Function to execute the training process on existing weights.
    """
    from continueTraining.logic.cl_yolo_train import continue_training

    # Set the path to the weights file
    root = os.getcwd()
    weights_dir = os.path.join(root, 'data', project, 'models')
    weights_path_continue_training = os.path.join(weights_dir, 'continue_training')
    weights_path_best = weights_dir

    if os.listdir(weights_path_continue_training):
        # If the continue_training directory has files, use the last modified file as the weights path
        weights_path = max([os.path.join(weights_path_continue_training, f) for f in os.listdir(
            weights_path_continue_training)], key=os.path.getmtime)
        print(f"Using weights from {weights_path}, so will train with this already further trained model.")
    else:
        # If the directory is empty, use best.pt in the main models directory
        weights_path = os.path.join(weights_path_best, 'best.pt')
        print(f"Using weights from {weights_path}, so it will train with the initially provided model.")

    # Set the path to the data configuration file
    data_yaml = os.path.join(root, 'data', project, 'yaml-files', 'data.yaml')

    current_experiment = experiment_name

    # Continue training the model
    continue_training(weights_path, data_yaml, 640, 16, 1, 0, current_experiment)


if __name__ == "__main__":
    # Install required packages before executing main function
    install_requirements('requirements.txt')

    # docker-compose up
    os.system('docker-compose up mlflow-server -d')

    # splitting the crude data into training and validation sets
    split_data()

    # Creating a new mlflow experiment if it does not exist
    create_mlflow_experiment()

    # training the model based on the existing weights with the new data
    train()

    # Wait function to make this continual learning script run indefinitely
    # wait(0.1, 8200)
