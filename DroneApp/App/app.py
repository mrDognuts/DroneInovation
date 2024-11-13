import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import warnings
import logging

# Ignore unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
warnings.filterwarnings("ignore")
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

# Define the base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure the uploads folder exists
uploads_dir = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Ensure folders for damaged and undamaged images exist
damaged_folder = os.path.join(uploads_dir, 'damaged')
undamaged_folder = os.path.join(uploads_dir, 'undamaged')

if not os.path.exists(damaged_folder):
    os.makedirs(damaged_folder)

if not os.path.exists(undamaged_folder):
    os.makedirs(undamaged_folder)

# Load the trained .h5 model for damage detection
def load_damage_detection_model(model_path):
    try:
        model = load_model(model_path)
        print("[INFO] Car damage detection model loaded.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

# Define the path for the damage detection model
damage_model_path = os.path.join(BASE_DIR, 'Car_detection.model.h5')
damage_detection_model = load_damage_detection_model(damage_model_path)

damage_classes = ["damaged", "non-damaged"]

def is_car_damaged(image):
    if damage_detection_model is None:
        return "Model not loaded correctly, please check the model path."
    
    try:
        # Open the image and resize it to (224, 224)
        img = Image.open(image)
        img = img.resize((224, 224)) 
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0) 

        predictions = damage_detection_model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        return damage_classes[predicted_class_index]
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "Prediction error."

# Load Detectron2 model
def load_detectron2_model(cfg, weights_path):
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # threshold for detecting objects
    cfg.MODEL.DEVICE = "cpu"  # Set to GPU if needed
    predictor = DefaultPredictor(cfg)
    return predictor

# Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes for damage detection
cfg.MODEL.WEIGHTS = os.path.join(BASE_DIR, 'model_final.pth')  # Path to your trained model

# Load the Detectron2 model
predictor = load_detectron2_model(cfg, cfg.MODEL.WEIGHTS)

# Detectron2 segmentation function
def segment_damage(image):
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(img_cv2)

    # Check for detected instances (damage) on car
    if "instances" in outputs and len(outputs["instances"]) > 0:
        instances = outputs["instances"].to("cpu")
        v = Visualizer(img_cv2[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(instances)
        highlighted_img = v.get_image()[:, :, ::-1]
        segmented_image = Image.fromarray(highlighted_img)
        return segmented_image
    else:
        return None

@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    files = request.files.getlist('image')  # Allow multiple file uploads
    results = []

    for file in files:
        img = Image.open(file.stream).convert('RGB')

        # Generate unique filenames for each image
        unique_filename = f"{uuid.uuid4()}.jpg"
        
        # Convert the image to the format needed for damage detection
        img_path = os.path.join(uploads_dir, unique_filename)
        img.save(img_path)

        # Check if the car is damaged
        damage_result = is_car_damaged(img_path)
        
        # Decide the folder based on the damage result
        folder = 'damaged' if damage_result == "damaged" else 'undamaged'
        folder_path = os.path.join(uploads_dir, folder)
        
        # Check if the file already exists, and if so, overwrite it
        file_path = os.path.join(folder_path, unique_filename)
        img.save(file_path)  # Overwrite the existing image

        # Build the response for the uploaded image
        response = {
            "prediction": damage_result,
            "image_url": url_for('static', filename=f'uploads/{folder}/{unique_filename}'),
            "message": "No damage detected." if damage_result == "non-damaged" else "Damage detected."
        }

        # If the car is damaged, segment the image
        if damage_result == "damaged":
            segmented_img = segment_damage(img)
            if segmented_img:
                processed_filename = f"{uuid.uuid4()}_processed.jpg"
                processed_img_path = os.path.join(folder_path, processed_filename)
                segmented_img.save(processed_img_path)
                response["processed_image_url"] = url_for('static', filename=f'uploads/{folder}/{processed_filename}')
            else:
                response["message"] = "Damage detected, but no segmentation was made."

        results.append(response)

    return jsonify(results) 

if __name__ == "__main__":
    app.run(debug=True)
