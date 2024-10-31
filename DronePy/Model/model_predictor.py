#pip install opencv-python
#pip install 'git+https://github.com/facebookresearch/detectron2.git'

import os
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

def load_model(model_weights="./output/model_final.pth", device="cpu"):
    # Load Detectron2 model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Update this to match your dataset's number of classes
    cfg.MODEL.DEVICE = device  # Use "cpu" or "cuda" for GPU
    cfg.MODEL.WEIGHTS = model_weights  # Path to the trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set a custom threshold for predictions
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return DefaultPredictor(cfg)

def analyze_image(predictor, image_path):
    # Load the image
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    # Visualize predictions
    v = Visualizer(im[:, :, ::-1], 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW)  # Optionally change to IMAGE_BW for segmentation
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_image = v.get_image()[:, :, ::-1]
    
    # Save processed image
    after_dir = "Dataset/after"
    if not os.path.exists(after_dir):
        os.makedirs(after_dir)
    processed_path = os.path.join(after_dir, os.path.basename(image_path))
    
    # Save the processed image only if it doesn't already exist
    if not os.path.exists(processed_path):
        cv2.imwrite(processed_path, processed_image)
        print(f"Processed and saved to {processed_path}")
    else:
        print(f"Image {processed_path} already exists. Skipping.")
    
    return processed_path

# Example usage function (to be called in another file)
def analyze_images_in_directory(before_dir):
    model_weights = "./output/model_final.pth"  # Path to your trained model weights
    predictor = load_model(model_weights=model_weights)
    
    # Analyze all images in the 'before' directory
    for filename in os.listdir(before_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(before_dir, filename)  # Full path to the image
            analyze_image(predictor, image_path)

# This allows the script to be used as a module
if __name__ == "__main__":
    before_dir = "./Dataset/before"  # Directory containing images to analyze
    analyze_images_in_directory(before_dir)
