import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

# Load the model once globally
def load_model(weights_path, device="cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.MODEL.DEVICE = device  
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return DefaultPredictor(cfg)

# Initialize the predictor with the path to weights
predictor = load_model("C:/Users/walte/Downloads/0Github/DroneBackup/DroneInovation/Drone/model_final.pth")

def analyze_image(image_path):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    v = Visualizer(im[:, :, ::-1], scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_image = v.get_image()[:, :, ::-1]
    
    after_dir = "Dataset/after"
    if not os.path.exists(after_dir):
        os.makedirs(after_dir)
    processed_path = os.path.join(after_dir, os.path.basename(image_path))
    cv2.imwrite(processed_path, processed_image)
    
    cv2.imshow("Prediction", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Processed and saved to {processed_path}")
    return processed_path
