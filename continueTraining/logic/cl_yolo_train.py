import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import torch
from datetime import datetime

# Set MLFlow tracking URI
mlflow.set_tracking_uri('http://localhost:5000')


def continue_training(weights: str, model_specifics: str, image_size: int, batch_size: int, epochs: int,
                      checkpoint_interval: int, current_experiment: str):
    # Set the root directory dynamically
    root = os.getcwd()

    # Set the MLFlow artifact root directory
    mlflow_artifact_root = os.path.join(root, 'mlflow')
    os.environ["MLFLOW_ARTIFACT_ROOT"] = mlflow_artifact_root

    # Get the experiment ID if it exists, or create a new one if it doesn't
    experiment = mlflow.get_experiment_by_name(current_experiment)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(current_experiment)

    # Remove the '_cl' extension from the experiment name
    new_name = current_experiment[:-3]

    now_time = datetime.now().strftime("%Y-%m-%d-%H:%M")

    # Get the number of existing runs within the experiment
    existing_runs = mlflow.search_runs(experiment_ids=[experiment_id])
    num_runs = len(existing_runs)
    name = f"{new_name}_{num_runs + 1}"

    # get the desired parameters
    desired_weight = weights.split("models\\", 1)[-1]
    desired_yaml = "CHILL\\" + model_specifics.split("CHILL\\", 1)[-1] if "CHILL\\" in model_specifics else ""

    # Start a mlflow run with a custom run name
    with mlflow.start_run(experiment_id=experiment_id, run_name=name):
        # Log parameters
        mlflow.log_params({
            "weights": desired_weight,
            "yaml_file": desired_yaml,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "checkpoint": checkpoint_interval,
            "experiment": current_experiment,
            "date": now_time
        })

        # Define the model and load the weights, if weights are not provided, use the default YOLOv8n weights
        if weights is None:
            model = YOLO(task='train')
            print("❌ ERROR: No weights provided. Training with default YOLOv8n weights.")
        else:
            model = YOLO(weights, task='train')
            print(f"✅ CONFIRMED: Training with weights from {weights}")

        # Use the GPU if available, otherwise default to CPU, with a warning
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('!!!ATTENTION: GPU ❌not❌ available. Training on CPU!!!')

        # Train the model
        for epoch in range(epochs % 10):
            # Define the model and load the weights, if weights are not provided, use the default YOLOv8n weights
            if weights is None:
                model = YOLO(task='train')
                print("❌ ERROR: No weights provided. Training with default YOLOv8n weights.")
            else:
                model = YOLO(weights, task='train')
                print(f"✅ CONFIRMED: Training with weights from {weights}")

            # Insert training code here, assume some metrics are calculated
            results = model.train(device=device, data=model_specifics, imgsz=image_size, batch=batch_size,
                                  epochs=epochs, save_period=checkpoint_interval)

            if hasattr(results, 'results_dict'):
                # Renaming metrics with invalid characters
                renamed_results_dict = {metric.replace('(', '').replace(')', ''): value for metric, value in
                                        results.results_dict.items()}
                mlflow.log_metrics(renamed_results_dict, step=epoch)

            # Save the trained model to the 'continue_training' directory
            root = os.getcwd()
            save_dir = os.path.join(root, 'data', new_name, 'models', 'continue_training')
            model_count = len(os.listdir(save_dir))
            new_model_name = f"best_{model_count + 1}.pt"

            # Save the model as 'best_x.pt' in /models/continue_training
            model_path = f"{save_dir}/{new_model_name}"

            # Save the model
            model.save(model_path)
            print(f"Model saved as best_{new_model_name} in {save_dir}")

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
            if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
                mlflow.pytorch.log_model(converted_model, f"{new_model_name}")
