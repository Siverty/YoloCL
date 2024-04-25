# Description: main.py is the entry point for the application.
# It is used to split the data into training and validation sets and to further train the model.

import os
from logic.data_handler.trainValSplit import split_data


def main():
    # Define the root path of the input directory for YOLO data
    yolo_input_root = os.path.join(os.getcwd(), 'yolo_input')

    # Define the path to save the train and validation sets
    train_val_root = os.path.join(os.getcwd(), 'cl_data')

    # Split the data into training and validation sets directly from the input
    split_data(yolo_input_root, train_val_root, 0.7)


if __name__ == "__main__":
    main()
