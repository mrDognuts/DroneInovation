from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

app = Flask(__name__)

if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Load the trained .h5 model for damage detection
def load_damage_detection_model(model_path):
    try:
        model = load_model(model_path) 
        print("[INFO] Car damage detection model loaded.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

damage_model_path = "./Car_detection.model.h5"  
damage_detection_model = load_damage_detection_model(damage_model_path)

damage_classes = ["damaged", "non-damaged"]

def is_car_damaged(image):
    if damage_detection_model is None:
        return "Model not loaded correctly, please check the model path."
    img = load_img(image, target_size=(224, 224))  
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array) 
    img_array = np.expand_dims(img_array, axis=0)

    try:
        predictions = damage_detection_model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])  
        return damage_classes[predicted_class_index]  
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "Prediction error."


def load_detectron2_model(cfg, weights_path):
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # threshold 
    cfg.MODEL.DEVICE = "cpu"  
    predictor = DefaultPredictor(cfg)
    return predictor

# Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("car_dataset_train",)
cfg.DATASETS.TEST = ("car_dataset_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes for damage detection
cfg.MODEL.WEIGHTS = r"C:\Users\walte\Downloads\0Github\PRJ_FINAL\DroneInovation\Drone_App_Test\App\model_final.pth" #trained model

# Load the model 2
predictor = load_detectron2_model(cfg, cfg.MODEL.WEIGHTS)

# detectron2
def segment_damage(image):
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(img_cv2)

    #Check for damage on car
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

# Define route handle upload/prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    temp_img_path = "./static/uploads/temp_uploaded_image.jpg"
    img.save(temp_img_path)

    # Check if the car is damaged
    damage_result = is_car_damaged(temp_img_path)
    response = {
        "prediction": damage_result,
        "image_url": url_for('static', filename='uploads/temp_uploaded_image.jpg')
    }

    if damage_result == "damaged":
        # Perform segmentation using Detectron2
        segmented_img = segment_damage(img)

        if segmented_img:
            # Save the segmented image
            processed_img_path = './static/uploads/processed_image.jpg'
            segmented_img.save(processed_img_path)
            # Add the segmented image 
            response["processed_image_url"] = url_for('static', filename='uploads/processed_image.jpg')
        else:
            response["message"] = "Damage detected, but no segmentation was made."

    else:
        response["message"] = "No damage detected."
    # Return the JSON response
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
