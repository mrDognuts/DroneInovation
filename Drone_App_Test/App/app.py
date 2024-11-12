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
    # Use the Mask R-CNN model with ResNet50 and FPN backbone, 3x schedule
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Set to 1 if it's just "damage" as the single class
    cfg.MODEL.DEVICE = device  # Set to "cpu" to use CPU or "cuda" for GPU
    cfg.MODEL.WEIGHTS = weights_path  # Path to your trained model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Initial prediction threshold (adjustable in code)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Set metadata for visualization
    MetadataCatalog.get("car_dataset").set(thing_classes=["damage"])
    
    return DefaultPredictor(cfg)

# Function to apply custom Non-Maximum Suppression (NMS)
def custom_nms(instances, nms_thresh=0.5):
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    keep = torch.ops.torchvision.nms(boxes, scores, nms_thresh)
    return instances[keep]

# Path to your trained model weights (make sure this path is correct)
weights_path = "C:/Users/walte/Downloads/0Github/DroneBackup/DroneInovation/Drone/model_final.pth"

# Initialize the predictor with the path to weights
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
    
    # Apply Non-Maximum Suppression to filtered instances
    nms_thresh = 0.5  # Adjust NMS threshold if needed
    filtered_instances = custom_nms(filtered_instances, nms_thresh)
    
    # Visualize predictions
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("car_dataset"), scale=0.8)
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
