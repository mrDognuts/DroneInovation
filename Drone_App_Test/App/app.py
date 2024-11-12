import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from io import BytesIO
from PIL import Image
import torch

app = Flask(__name__)

# Load the model once globally
def load_model(weights_path, device="cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.WEIGHTS = weights_path  # Path to your trained model weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class for damage detection
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Confidence threshold for inference
    
    # Set device for inference (CPU or CUDA)
    cfg.MODEL.DEVICE = device 
    
    # Initialize the predictor
    return DefaultPredictor(cfg)

# Path 
weights_path = r"C:\Users\walte\Downloads\0Github\PRJ_FINAL\DroneInovation\Drone_App_Test\App\model_final.pth"
predictor = load_model(weights_path)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    # Load the image from the uploaded file
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Run prediction
    outputs = predictor(img)
    
    # Extract instances and apply confidence and area filtering
    instances = outputs["instances"].to("cpu")
    
    # Set minimum confidence and minimum area thresholds
    min_confidence = 0.5  # Adjust as necessary
    min_area = 1000       # Minimum area in pixels for detections

    # Filter instances based on confidence and area
    filtered_instances = instances[(instances.scores > min_confidence) & 
                                   (instances.pred_boxes.area() > min_area)]
    
    # Visualize predictions
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("car_dataset_train"), scale=0.8)
    v = v.draw_instance_predictions(filtered_instances)
    processed_image = v.get_image()[:, :, ::-1]
    
    # Convert to a format suitable for sending as a response
    processed_pil_image = Image.fromarray(processed_image)
    img_io = BytesIO()
    processed_pil_image.save(img_io, 'JPEG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
