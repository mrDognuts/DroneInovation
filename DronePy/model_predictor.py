import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

def load_model(model_weights="model_final.pth", device="cpu"):
    # Load Detectron2 model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Adjust this to match the number of classes in your dataset
    cfg.MODEL.DEVICE = device  # Use CPU or "cuda" for GPU
    cfg.MODEL.WEIGHTS = model_weights  # Path to the trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for predictions
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return DefaultPredictor(cfg)

def analyze_image(predictor, image_path):
    # Load the image
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    # Visualize predictions
    v = Visualizer(im[:, :, ::-1], scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_image = v.get_image()[:, :, ::-1]
    
    # Save processed image
    after_dir = "Dataset/after"
    if not os.path.exists(after_dir):
        os.makedirs(after_dir)
    processed_path = os.path.join(after_dir, os.path.basename(image_path))
    cv2.imwrite(processed_path, processed_image)
    
    # Display processed image (only works if your environment supports cv2.imshow)
    cv2.imshow("Prediction", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Processed and saved to {processed_path}")
    return processed_path
