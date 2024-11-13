import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Load damage detection model
def load_damage_detection_model(model_path):
    try:
        model = load_model(model_path)
        print("[INFO] Car damage detection model loaded.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

# Load Detectron2 model
def load_detectron2_model(cfg, weights_path):
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # confidence threshold
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

# Initialize models
damage_model_path = r"C:\Users\walte\Downloads\0Github\PRJ_FINAL\DroneInovation\DroneApp\Drone\Car_detection.model.h5"
damage_detection_model = load_damage_detection_model(damage_model_path)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("car_dataset_train",)
cfg.DATASETS.TEST = ("car_dataset_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.MODEL.WEIGHTS = r"C:\Users\walte\Downloads\0Github\PRJ_FINAL\DroneInovation\DroneApp\Drone\model_final.pth"

predictor = load_detectron2_model(cfg, cfg.MODEL.WEIGHTS)
damage_classes = ["damaged", "non-damaged"]

def is_car_damaged(image_path):
    if damage_detection_model is None:
        return "Model not loaded correctly, please check the model path."

    img = load_img(image_path, target_size=(224, 224))
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

def segment_damage(image_path):
    """Run segmentation on a damaged car."""
    img = Image.open(image_path).convert('RGB')
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    outputs = predictor(img_cv2)

    if "instances" in outputs and len(outputs["instances"]) > 0:
        instances = outputs["instances"].to("cpu")
        v = Visualizer(img_cv2[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(instances)
        highlighted_img = v.get_image()[:, :, ::-1]
        segmented_image = Image.fromarray(highlighted_img)

        segmented_path = f"static/processed_{os.path.basename(image_path)}"
        segmented_image.save(segmented_path)
        return segmented_path
    else:
        return None

def process_images(image_paths):
    """Process a batch of images for damage detection and segmentation."""
    results = []

    for image_path in image_paths:
        result = {}
        result["original_image_path"] = image_path

        # Detect damage
        damage_result = is_car_damaged(image_path)
        result["prediction"] = damage_result

        # Segment if damaged
        if damage_result == "damaged":
            segmented_image_path = segment_damage(image_path)
            if segmented_image_path:
                result["segmented_image_path"] = segmented_image_path
                result["message"] = "Damage detected and segmented."
            else:
                result["message"] = "Damage detected, but no segmentation was made."
        else:
            result["message"] = "No damage detected."

        results.append(result)

    return results
