from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO


def load_latest_model(project_name):
    model_path = None
    model_dir = f"data/{project_name}/models/continue_training/"
    models = glob.glob(os.path.join(model_dir, "best_*.pt"))
    if models:
        model_path = max(models, key=os.path.getctime)
    else:
        model_path = f"data/{project_name}/models/best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return YOLO(model_path)


def create_app(project_name):
    app = Flask(__name__, static_folder='interface', template_folder='interface/pages')
    CORS(app)

    # Load the model
    model = load_latest_model(project_name)
    print(f"Loaded model: {model}")

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

    @app.route('/detect', methods=['POST'])
    def detect():
        file = request.files['image']
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Perform inference
        results = model(img)

        # Extract detections
        detections = []
        if isinstance(results, list):
            for result in results:
                boxes = result.boxes if hasattr(result, 'boxes') else []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensors to lists
                    confidence = box.conf[0].item()  # Convert tensor to float
                    class_id = box.cls[0].item()  # Convert tensor to float
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        else:
            if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensors to lists
                    confidence = box.conf[0].item()  # Convert tensor to float
                    class_id = box.cls[0].item()  # Convert tensor to float
                    detections.append([x1, y1, x2, y2, confidence, class_id])

        return jsonify(detections)

    return app
