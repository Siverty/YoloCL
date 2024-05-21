# Description: This script creates a Flask web application that serves a web page for uploading an image and detecting
# objects in it using a YOLOv8 model. The model is loaded from the latest checkpoint file in the specified project.

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import cv2
import numpy as np
import os
import glob
import torch
from ultralytics import YOLO
import time


# Function to load the latest model from the specified project
def load_latest_model(project_name):
    model_path = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, f"../data/{project_name}/models/continue_training/")

    print(f"Looking for models in: {os.path.abspath(model_dir)}")

    models = glob.glob(os.path.join(model_dir, "best_*.pt"))

    if models:
        model_path = max(models, key=os.path.getctime)
    else:
        model_path = os.path.join(current_dir, f"../data/{project_name}/models/best.pt")

    print(f"Trying to load model from: {os.path.abspath(model_path)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(model_path)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    return model


# Function to warm-up the model by running a dummy input through it, this is done to avoid the overhead of loading the
# model and moving it to the GPU when the first request is made. Makes the first request faster.
def warm_up_model(model):
    dummy_input = torch.zeros(1, 3, 640, 640).to('cuda' if torch.cuda.is_available() else 'cpu')
    _ = model(dummy_input)
    torch.cuda.synchronize()


# Function to create the Flask application
def create_app(project_name):
    app = Flask(__name__, static_folder='.', template_folder='pages')
    # CORS is enabled to allow requests from any origin
    CORS(app, resources={r"/*": {"origins": "*"}})

    model = load_latest_model(project_name)
    print(f"Loaded model: {model}")

    warm_up_model(model)
    print("Model warm-up complete")

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/artefacts/<path:filename>')
    def serve_artefacts(filename):
        return send_from_directory('artefacts', filename)

    @app.route('/styles/<path:filename>')
    def serve_styles(filename):
        return send_from_directory('styles', filename)

    @app.route('/code/<path:filename>')
    def serve_code(filename):
        return send_from_directory('code', filename)

    @app.route('/worker.js')
    def serve_worker():
        return send_from_directory('.', 'misc/worker.js', mimetype='application/javascript')

    @app.route('/detect', methods=['POST'])
    def detect():
        start_time = time.time()

        file = request.files['image']
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        img_resized = cv2.resize(img, (640, 640))
        img_resized = np.transpose(img_resized, (2, 0, 1))
        img_resized = np.expand_dims(img_resized, axis=0)
        img_tensor = torch.from_numpy(img_resized).float() / 255.0

        if torch.cuda.is_available():
            img_tensor = img_tensor.to('cuda')

        preprocessing_time = time.time()
        print(f"Preprocessing Time: {preprocessing_time - start_time} seconds")

        inference_start_time = time.time()
        results = model(img_tensor)
        torch.cuda.synchronize()
        inference_end_time = time.time()
        print(f"Inference Time: {inference_end_time - inference_start_time} seconds")

        detections = []
        if isinstance(results, list):
            for result in results:
                boxes = result.boxes if hasattr(result, 'boxes') else []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = box.cls[0].item()
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        else:
            if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = box.cls[0].item()
                    detections.append([x1, y1, x2, y2, confidence, class_id])

        postprocessing_time = time.time()
        print(f"Postprocessing Time: {postprocessing_time - inference_end_time} seconds")

        total_time = postprocessing_time - start_time
        print(f"Total Time: {total_time} seconds")

        return jsonify(detections)

    return app
