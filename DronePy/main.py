import cv2
import time
from djitellopy import Tello

# Global variable to track the car number
car_number = 1

def capture_images(tello, num_images=8):
    global car_number
    angle_step = 360 // num_images  # Calculate the angle step for 8 images (45 degrees per step)
    images = []

    # Loop through each angle position
    for i in range(num_images):
        # Rotate the drone by the calculated angle step
        tello.rotate_clockwise(angle_step)

        # Sleep for a short duration to allow the drone to stabilize
        time.sleep(2)

        # Capture the image from the current frame
        frame = tello.get_frame_read().frame
        image_name = f"car{car_number}.{i+1}.png"
        cv2.imwrite(image_name, frame)
        images.append(image_name)
        print(f"Captured {image_name}")

    # Increment the car number for the next car
    car_number += 1
    return images

def main():
    # Initialize Tello drone
    tello = Tello()
    tello.connect()

    # Start video stream
    tello.streamon()
    time.sleep(2)  # Allow the stream to initialize

    try:
        # Take off
        tello.takeoff()
        tello.move_up(50)  # Adjust the height to an appropriate level for capturing images

        # Capture images from different angles
        capture_images(tello)

        # Land the drone
        tello.land()
        
    finally:
        # Safely close everything
        tello.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
