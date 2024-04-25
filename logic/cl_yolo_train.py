from ultralytics import YOLO
import torch
import os


# Define the command to continue training
def continue_training(weights, model_specifics, image_size, batch_size, epochs, checkpoint):

    # Load the existing model
    model = YOLO(f"{weights}")

    # use the gpu if available
    use_gpu = '0' if torch.cuda.is_available() else ''

    # Use the model
    model.train(data=model_specifics, epochs=epochs, device=use_gpu,
                imgsz=image_size, batch=batch_size, save_period=checkpoint)

    # Save the model
    # Get root directory
    root = os.getcwd()

    # Define the directory to save the model
    save_dir = os.path.join(root, 'models', 'continue_training')

    # Get the amount of models in the directory
    model_count = len(os.listdir(save_dir))
    new_model_name = model_count + 1

    # save model best.pt in /models/continue_training
    model.save(f"{save_dir}/best_{new_model_name}.pt")
