from ultralytics import YOLO
import pandas as pd
from datetime import datetime

def cl_model_test(test_path, cl_weights, old_weights):
    # Load the old model
    old_model = YOLO(f"{old_weights}")

    # Load the new model
    new_model = YOLO(f"{cl_weights}")

    results_old = old_model.test(test_path, batch_size=32, show=True)

    results_new = new_model.test(test_path, batch_size=32, show=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    df_old = pd.DataFrame(results_old)

    df_old.to_csv(f"metrics/quantification/results-old{current_time}.csv")

    df_new = pd.DataFrame(results_new)

    df_old.to_csv(f"metrics/quantification/results-new{current_time}.csv")

    return df_old, df_new


