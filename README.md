# Yolo Continual Learning

This is a repository to display the proof of concept for training a YOLO model within a continual learning framework. 
The repository is based on the <font color="yellow">YOLOv8</font>
 model from [Ultralytics](https://github.com/ultralytics/ultralytics).

## Installation
To start you need to make a virtual environment for the project and install the requirements. 

For windows:
```powershell
python -m venv .venv
.venv/Scripts/activate.ps1
```
For Linux & Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Only then you can run `main.py` to start the pipeline.

`python main.py`

When started this will take the latest model that is stored in the models or models/continue_training folder and 
start training on the dataset that is stored in the cl_data folder.

## Uses
This project is meant to be implemented within the CHIMP (Continual Hypothesis and Information Mending Pipeline) project
from the [Lectoraat Data Intelligence](https://dilab.nl/).

The project is meant to be used to train a YOLO model on a new input dataset which is stored in the cl_data folder.
Then the latest model is trained with the new dataset that has been split for training and validation. 

The current model has testing capabilities but these are not yet implemented in the pipeline. This will be done in
the frontend of the CHIMP project.

## Future

The project will include a mlflow tracking server to keep track of the training process and the models that are trained.
Currently, the focus of the project is to get the training pipeline up and running.

## License
This project is licensed under the _MIT License_ - see the [LICENSE](LICENSE) file for details.


&copy; 2024 - Lectoraat Data Intelligence, Justin Schots

