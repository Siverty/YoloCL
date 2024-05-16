import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import torch
import time

# Set MLFlow tracking URI
mlflow.set_tracking_uri('http://localhost:8282')


def continue_training(weights: str, data_yaml: str, image_size: int, batch_size: int, epochs: int,
                      checkpoint_interval: int, experiment_name: str, experiment_id: str, parent_index: int,
                      child_index: int):
    # Set the root directory dynamically
    root = os.getcwd()

    # Set the MLFlow artifact root directory
    mlflow_artifact_root = os.path.join(root, 'mlflow')
    os.environ["MLFLOW_ARTIFACT_ROOT"] = mlflow_artifact_root

    # Remove '_cl' from the experiment name
    new_name = experiment_name[:-3]

    iteration_name = f"{new_name}_{parent_index}.{child_index}"  # CHILL_3.1, CHILL_3.2, etc.

    with mlflow.start_run(experiment_id=experiment_id, run_name=iteration_name, nested=True):
        mlflow.log_params({
            "Dataset": "CHILL",
            "Weights": weights,
            "Epochs": epochs,
            "Checkpoint": checkpoint_interval,
            "Experiment": experiment_name,
            "Date": time.strftime("%Y-%m-%d-%H:%M"),
            "yaml_file": data_yaml,
            "image_size": image_size,
            "batch_size": batch_size,
        })

        # Use the GPU if available, otherwise default to CPU, with a warning
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('!!!ATTENTION: GPU ❌ not ❌ available. Training on CPU!!!')

        # Define the model and load the weights, if weights are not provided, use the default YOLOv8n weights
        if weights is None:
            model = YOLO(task='train')
            print("❌ ERROR: No weights provided. Training with default YOLOv8n weights.")
        else:
            model = YOLO(weights, task='train')
            print(f"✅ CONFIRMED: Training with weights from {weights}")

        # Set up the training parameters for YOLO
        training_params = {
            'device': device,
            'data': data_yaml,
            'imgsz': image_size,
            'batch': batch_size,
            'epochs': epochs,
            'save_period': checkpoint_interval,
            'val': False,
        }

        # Train the model and log the parameters after each input
        for epoch in range(1):
            results = model.train(**training_params)

            # Assuming the results object has attributes like 'metrics' containing precision, recall, etc.
            if hasattr(results, 'results_dict'):
                # Renaming metrics with invalid characters
                reshaped_results_dict = {metric.replace('(', '').replace(')', ''): value for metric, value in
                                         results.results_dict.items()}
                mlflow.log_metrics(reshaped_results_dict, step=epoch)

        # Save the trained model to the 'continue_training' directory
        save_dir = os.path.join(root, 'data', new_name, 'models', 'continue_training')
        os.makedirs(save_dir, exist_ok=True)
        model_count = len(os.listdir(save_dir))
        new_model_name = f"best_{model_count + 1}.pt"

        # Save the model as 'best_x.pt' in /models/continue_training
        model_path = os.path.join(save_dir, new_model_name)
        model.save(model_path)
        print(f"✅ Model saved as {new_model_name} in {save_dir}")

        # Load the model
        torch_model = torch.load(model_path)

        # Define a new model class that wraps the loaded model
        class ConvertedModel(torch.nn.Module):
            def __init__(self, model):
                super(ConvertedModel, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)

        # Convert the loaded model to a torch.nn.Module
        converted_model = ConvertedModel(torch_model)

        # Log the converted model to MLflow
        mlflow.pytorch.log_model(converted_model, new_model_name)
