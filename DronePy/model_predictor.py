#pip install opencv-python
#pip install 'git+https://github.com/facebookresearch/detectron2.git'

import os
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

def load_model(model_weights="./output/model_final.pth", device="cpu"):
    # Load model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.MODEL.DEVICE = device  
    cfg.MODEL.WEIGHTS = model_weights 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
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
                   instance_mode=ColorMode.IMAGE_BW)  
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_image = v.get_image()[:, :, ::-1]
    
    # Save processed image
    after_dir = "Dataset/after"
    if not os.path.exists(after_dir):
        os.makedirs(after_dir)
    processed_path = os.path.join(after_dir, os.path.basename(image_path))
    
    if not os.path.exists(processed_path):
        cv2.imwrite(processed_path, processed_image)
        print(f"Processed and saved to {processed_path}")
    else:
        print(f"Image {processed_path} already exists. Skipping.")
    
    return processed_path

def analyze_images_in_directory(before_dir):
    model_weights = "./output/model_final.pth"  
    predictor = load_model(model_weights=model_weights)
    
    for filename in os.listdir(before_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
            image_path = os.path.join(before_dir, filename)  
            analyze_image(predictor, image_path)


