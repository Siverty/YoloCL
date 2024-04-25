# Description: Test the predictive performance of old and new YOLO models on a test dataset.

import os


def cl_model_test(test_path, cl_weights, old_weights):
    """
    Test the performance of old and new YOLO models on a test dataset.

    Args:
    - test_path (str): Path to the test dataset.
    - cl_weights (str): Path to the weights file of the new model.
    - old_weights (str): Path to the weights file of the old model.

    Returns:
    - df_old (DataFrame): Results of testing the old model.
    - df_new (DataFrame): Results of testing the new model.
    """
    from ultralytics import YOLO
    import pandas as pd
    from datetime import datetime

    # Load the old model
    old_model = YOLO(old_weights)

    # Load the new model
    new_model = YOLO(cl_weights)

    # Test the old model
    results_old = old_model.test(test_path, batch_size=32, show=True)

    # Test the new model
    results_new = new_model.test(test_path, batch_size=32, show=True)

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Save results of testing the old model to a CSV file
    df_old = pd.DataFrame(results_old)
    df_old.to_csv(f"metrics/quantification/results-old{current_time}.csv")

    # Save results of testing the new model to a CSV file
    df_new = pd.DataFrame(results_new)
    df_new.to_csv(f"metrics/quantification/results-new{current_time}.csv")

    return df_old, df_new
