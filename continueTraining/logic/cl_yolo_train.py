import os
import mlflow
import mlflow.pytorch

# Set MLFlow tracking URI
mlflow.set_tracking_uri('http://localhost:5000')


def continue_training(weights, model_specifics, image_size, batch_size, epochs, checkpoint, current_experiment):
    from ultralytics import YOLO
    import torch
    from datetime import datetime

    # Get the experiment ID if it exists, or create a new one if it doesn't
    experiment = mlflow.get_experiment_by_name(current_experiment)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(current_experiment)

    # Start a mlflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_params({
            "weights": weights,
            "model_specifics": model_specifics,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "checkpoint": checkpoint
        })

        # Load the existing model
        model = YOLO(weights, task='train')

        # Use the GPU if available, otherwise default to CPU
        gpu_preference = '0' if torch.cuda.is_available() else ''
        if gpu_preference != '0':
            print('!!!ATTENTION: GPU not available. Training on CPU!!!')

        # Train the model
        for epoch in range(epochs):
            # Insert training code here, assume some metrics are calculated
            model.train(device=gpu_preference, data=model_specifics,  imgsz=image_size, batch=batch_size, epochs=epochs,
                        save_period=checkpoint)
            # Example: log loss and accuracy for each epoch (adjust according to actual available metrics)
            mlflow.log_metrics({"loss": 0.01 * epoch, "accuracy": 0.98 - 0.01 * epoch}, step=epoch)

        # Save the trained model
        root = os.getcwd()
        save_dir = os.path.join(root, 'models', 'continue_training')
        model_count = len(os.listdir(save_dir))
        new_model_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
        new_model_name = f"{model_count + 1}_{new_model_name}"

        # Save the model as 'best_x.pt' in /models/continue_training
        model_path = f"{save_dir}/best_{new_model_name}.pt"
        model.save(model_path)
        print(f"Model saved as best_{new_model_name}.pt in {save_dir}")

        # Log model as an artifact
        mlflow.log_artifact(model_path)
