import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Load the trained .h5 model for damage detection
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # threshold for detecting objects
    cfg.MODEL.DEVICE = "cpu"  # Set to GPU if needed
    predictor = DefaultPredictor(cfg)
    return predictor

# Detectron2 segmentation function
def segment_damage(image, cfg, predictor):
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

# Initialize Detectron2 configuration
def initialize_detectron2_model(base_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes for damage detection
    cfg.MODEL.WEIGHTS = os.path.join(base_dir, 'model_final.pth')  # Path to your trained model
    return cfg
