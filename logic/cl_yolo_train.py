# Description: This file contains the logic for continuing training a YOLO model.

import os


def continue_training(weights, model_specifics, image_size, batch_size, epochs, checkpoint):
    """
    Continue training a YOLO model.

    Args:
    - weights (str): Path to the weights file of the existing model.
    - model_specifics (str): Path to the data configuration file.
    - image_size (int): Size of the input images.
    - batch_size (int): Batch size for training.
    - epochs (int): Number of epochs for training.
    - checkpoint (int): Save model after this number of epochs.
    """
    from ultralytics import YOLO
    import torch

    # Load the existing model
    model = YOLO(weights, task='train')

    # Use the GPU if available, otherwise default to CPU
    gpu_preference = '0' if torch.cuda.is_available() else ''

    if gpu_preference != '0':
        print('ATTENTION: GPU not available. Training on CPU.')

    # Train the model
    model.train(data=model_specifics, epochs=epochs, device=gpu_preference,
                imgsz=image_size, batch=batch_size, save_period=checkpoint)

    # Save the trained model
    # Get the root directory
    root = os.getcwd()

    # Define the directory to save the model
    save_dir = os.path.join(root, 'models', 'continue_training')

    # Get the amount of models in the directory
    model_count = len(os.listdir(save_dir))
    new_model_name = model_count + 1

    # Save the model as 'best_x.pt' in /models/continue_training
    model.save(f"{save_dir}/best_{new_model_name}.pt")

    print(f"Model saved as best_{new_model_name}.pt in {save_dir}  \n As best_{new_model_name}.pt")
