import cv2
import time
import os
from djitellopy import Tello
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# Global variable to track the car number
car_number = 1

def capture_images(tello, num_images=8, output_dir="Dataset/before"):
    global car_number
    angle_step = 360 // num_images  # 8 images, 45 degrees per step
    images = []

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each angle position
    for i in range(num_images):
        # Rotate the drone and move slightly to the left for a circular path
        tello.rotate_clockwise(angle_step)
        tello.move_left(50)  # Adjust distance if needed

        # Allow stabilization
        time.sleep(2)

        # Capture the image from the current frame
        frame = tello.get_frame_read().frame
        image_name = f"car{car_number}.{i+1}.png"
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, frame)
        images.append(image_path)
        print(f"Captured {image_path}")

    # Increment the car number for the next car
    car_number += 1
    return images

# def analyze_images(images, output_dir="Dataset/after"):
#     # Load the trained model
#     cfg = get_cfg()
#     cfg.MODEL.WEIGHTS = os.path.join("model_final.pth")  # Path to the trained model
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for predictions
#     predictor = DefaultPredictor(cfg)

#     # Ensure the output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Analyze each captured image
#     for image_path in images:
#         im = cv2.imread(image_path)
#         outputs = predictor(im)
#         v = Visualizer(im[:, :, ::-1], metadata=None, scale=1.0)  # Adjust metadata as necessary
#         v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

#         # Save the analyzed image
#         output_image_name = os.path.basename(image_path).replace("car", "analyzed_car")
#         output_image_path = os.path.join(output_dir, output_image_name)
#         cv2.imwrite(output_image_path, v.get_image()[:, :, ::-1])
#         print(f"Analyzed image saved at: {output_image_path}")

def main():
    # Initialize Tello drone
    tello = Tello()
    tello.connect()

    # Start video stream
    tello.streamon()
    time.sleep(2)  # Allow the stream to initialize

    try:
        # Take off and reach the desired height
        tello.takeoff()
        tello.move_up(50)

        # Capture images from different angles around the car
        images = capture_images(tello)

        # Land the drone after capturing images
        tello.land()

        # Analyze the captured images and save them in the "after" folder
#        analyze_images(images)

    finally:
        # Safely close the video stream and windows
        tello.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
