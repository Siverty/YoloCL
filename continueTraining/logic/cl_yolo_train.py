import os
import mlflow
import mlflow.pytorch

# Set MLFlow tracking URI
mlflow.set_tracking_uri('http://localhost:5000')


def continue_training(weights: str, model_specifics: str, image_size: int, batch_size: int, epochs: int,
                      checkpoint: int, current_experiment: str):
    from ultralytics import YOLO
    import torch
    from datetime import datetime

    # Get the experiment ID if it exists, or create a new one if it doesn't
    experiment = mlflow.get_experiment_by_name(current_experiment)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(current_experiment)

    # Remove the '_cl' extension from the experiment name
    new_name = current_experiment[:-3]

    now_time = datetime.now().strftime("%Y-%m-%d %H:%M")

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
            "checkpoint": checkpoint,
            "experiment": current_experiment,
            "date": now_time
        })

        # Load the existing model
        model = YOLO(weights, task='train')

        # Use the GPU if available, otherwise default to CPU
        gpu_preference = '0' if torch.cuda.is_available() else ''
        if gpu_preference != '0':
            print('!!!ATTENTION: GPU not available. Training on CPU!!!')

        # Train the model
        for epoch in range(1):
            # Insert training code here, assume some metrics are calculated
            model.train(device=gpu_preference, data=model_specifics,  imgsz=image_size, batch=batch_size, epochs=epochs,
                        save_period=checkpoint, save_dir=0)
            # log loss and accuracy for each epoch (adjust according to actual available metrics)
            mlflow.log_metrics({"loss": 0.01 * epoch, "accuracy": 0.98 - 0.01 * epoch}, step=epoch)

            # Save the trained model
            root = os.getcwd()
            save_dir = os.path.join(root, 'data', new_name, 'models', 'continue_training')
            model_count = len(os.listdir(save_dir))
            new_model_name = f"{model_count + 1}"

            # Save the model as 'best_x.pt' in /models/continue_training
            model_path = f"{save_dir}/best_{new_model_name}.pt"
            model.save(model_path)
            print(f"Model saved as best_{new_model_name}.pt in {save_dir}")

            # Log model as an artifact
            mlflow.log_artifact(model_path)
