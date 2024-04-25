import os
import ultralytics
from ultralytics import YOLO
import pandas as pd
import torch
from datetime import datetime


# Define the command to continue training
def continue_training(weights, model_specifics, imageSZ, batchSZ, epochs):
    torch.backends.cudnn.enabled = False
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    # Load the existing model
    model = YOLO(f"{weights}")

    # use the gpu if available
    use_gpu = '0' if torch.cuda.is_available() else ''

    # Use the model
    model.train(data=model_specifics, epochs=epochs, device=use_gpu, imgsz=imageSZ, batch=batchSZ)
    metrics = weights.val()  # evaluate model performance on the validation set
    results = pd.DataFrame(metrics).transpose()  # save metrics to file

    # Get the date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Save the results
    results.to_csv(f"metrics/results-{current_datetime}.csv")
