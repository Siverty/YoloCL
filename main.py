# Description: main.py is the entry point for the application.
# It is used to split the data into training and validation sets and to further train the model.

# Be sure to run this script in a .venv environment with the required packages installed.
# The pytorch && CUDA installations will otherwise throw exotics errors.

# TODO:
# 1. add testing capabilities
# 2. add logging
# 3. add parameters for continual learning -- check
# 4. run directory add
# 5. make project a parameter that is defined elsewhere

import os
import sys
import mlflow
import docker
from docker.errors import NotFound, DockerException

# Select what project to use
project = 'CHILL'
experiment_name = project + '_cl'

# Set MLFlow tracking URI
mlflow.set_tracking_uri('http://localhost:5000')


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


def is_docker_engine_running():
    """
    Check if the Docker engine is running.
    """
    try:
        client = docker.from_env()
        client.ping()
        return True
    except (DockerException, NotFound):
        return False


def is_container_running(container_name_substring):
    """
    Check if any Docker container with a name containing the given substring is running.
    """
    client = docker.from_env()
    containers = client.containers.list()
    for container in containers:
        if container_name_substring in container.name:
            return True
    return False


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
    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Create the experiment for this dataset/project if it does not exist
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created.")
    else:
        print(f"Experiment '{experiment_name}' already exists.\nContinuing training... ▶️ ▶️ ▶️ ")


def fetch_parent_runs(experiment_id):
    runs = mlflow.search_runs([experiment_id])
    if runs.empty:
        return []
    return [run for run in runs.itertuples() if getattr(run, 'info.parent_run_id', None) is None]


def name_next_parent_run(experiment_id, base_name=f"Parent_{experiment_name}"):
    parent_runs = fetch_parent_runs(experiment_id)
    next_index = len(parent_runs) + 1
    return f"{base_name}_{next_index}", next_index


def train():
    """
    Function to execute the training process on existing weights.
    """
    from continueTraining.logic.cl_yolo_train import continue_training

    # Set the path to the weights file
    root = os.getcwd()
    weights_dir = os.path.join(root, 'data', project, 'models')
    weights_path_continue_training = os.path.join(weights_dir, 'continue_training')

    # Constants
    amount_of_runs = 3
    # --The weights path is determined in the upcoming if-else statement -- #
    data_yaml = os.path.join(root, 'data', project, 'yaml-files', 'data.yaml')
    image_size = 640
    batch_size = 16
    epochs = 2
    checkpoint_interval = 0

    if os.listdir(weights_path_continue_training):
        # If the continue_training directory has files, use the last modified file as the weights path
        weights_path = max([os.path.join(weights_path_continue_training, f) for f in os.listdir(
            weights_path_continue_training)], key=os.path.getmtime)
        print(f"Using weights from {weights_path}, so will train with this already further trained model.⏭")
    else:
        # If the directory is empty, use best.pt in the main models directory
        weights_path = os.path.join(weights_dir, 'best.pt')
        print(f"Using weights from {weights_path}, so it will train with the initially provided model.▶️")

    # Get the experiment ID if it exists, or create a new one if it doesn't
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # Get the number of existing runs within the experiment
    existing_runs = mlflow.search_runs(experiment_ids=[experiment_id])
    num_runs = len(existing_runs) - 1
    name = num_runs + 1

    # dynamically name the parent run
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    next_run_name, parent_index = name_next_parent_run(experiment_id)

    # Start the parent run with child runs nested
    with mlflow.start_run(run_name=next_run_name, experiment_id=experiment_id):
        for i in range(amount_of_runs):
            continue_training(weights_path, data_yaml, image_size, batch_size, epochs, checkpoint_interval,
                              experiment_name, experiment_id, parent_index, i + 1)


if __name__ == "__main__":
    """
    Main function to run the continual learning process.
    """
    # Install required packages before executing main function
    install_requirements('requirements.txt')

    # Check if Docker engine is running
    if not is_docker_engine_running():
        print("Docker engine is not running. Please start Docker Desktop and try again.")
        sys.exit(1)

    # docker-compose up
    os.system('docker-compose up mlflow-server -d')

    # Check if any container with 'mlflow' in its name is running
    if is_container_running('mlflow'):
        # splitting the crude data into training and validation sets
        split_data()
    else:
        print("ERROR: No container with 'mlflow' in its name is running. Please start the container and try again.")

    # Creating a new mlflow experiment if it does not exist
    create_mlflow_experiment()

    # training the model based on the existing weights with the new data
    train()

    # Wait function to make this continual learning script run indefinitely
    # wait(0.1, 8200)
