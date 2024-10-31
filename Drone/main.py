import cv2
import time
import os
from djitellopy import Tello
from pynput import keyboard as kp
from model_predictor import load_model, analyze_image  # Ensure this is correctly set up

# Global variables
car_number = 1
capture_flag = False
current_keys = set()
captured_images = []  # Store captured image paths
logged_status = False  # Flag to track if status has been logged

# Initialize the predictor by loading the model
predictor = load_model("model_final.pth", "cpu")  # Adjust path and device as needed

# File paths for logs
drone_log_file = "drone_status_log.txt"
keypress_log_file = "keypress_log.txt"

def log_key_press(key):
    """Log each key press to keypress_log.txt."""
    with open(keypress_log_file, "a") as log:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Key pressed: {key}\n")

def log_drone_status(tello):
    """Log the battery and temperature status."""
    battery = tello.get_battery()
    temperature = tello.get_temperature()
    with open(drone_log_file, "a") as log:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Battery: {battery}%, Temperature: {temperature}°C\n")

def capture_and_save_image(frame, output_dir="Dataset/before"):
    """Capture and save the image with an incremental filename."""
    global car_number
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_name = f"car{car_number}.1.png"
    image_path = os.path.join(output_dir, image_name)
    cv2.imwrite(image_path, frame)
    print(f"Captured {image_path}")

    captured_images.append(image_path)  # Store the image path
    car_number += 1  # Increment the car number after capturing the image

def on_press(key):
    """Handle key press events."""
    global capture_flag
    try:
        current_keys.add(key.char)
        if key.char == 'c':  # Press 'c' to capture an image
            capture_flag = True
        log_key_press(key)
    except AttributeError:
        current_keys.add(key)

def on_release(key):
    """Handle key release events."""
    try:
        current_keys.remove(key.char)
    except KeyError:
        pass
    if key == kp.Key.esc:  # Stop listener
        return False

def getKeyboardInput(tello):
    """Control drone movement using keyboard input and return movement values."""
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if 'h' in current_keys:  # Move left
        lr = -speed
    if 'k' in current_keys:  # Move right
        lr = speed
    if 'u' in current_keys:  # Move forward
        fb = speed
    if 'j' in current_keys:  # Move backward
        fb = -speed
    if 'w' in current_keys:  # Move up
        ud = speed
    if 's' in current_keys:  # Move down
        ud = -speed
    if 'a' in current_keys:  # Rotate left
        yv = -speed
    if 'd' in current_keys:  # Rotate right
        yv = speed
    if 'q' in current_keys:  # Land
        tello.land()
        time.sleep(3)
    if 'e' in current_keys:  # Take off
        tello.takeoff()
        return True  # Indicate that the drone has taken off

    return [lr, fb, ud, yv]

def main():
    global capture_flag, logged_status
    tello = Tello()

    try:
        tello.connect()
        print("Connected to Tello...")

        # Start video streaming
        tello.streamon()
        time.sleep(2)
        print("Started streaming...")

        # Start listener for key presses
        listener = kp.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while True:
            frame = tello.get_frame_read().frame
            cv2.imshow("Tello Live Feed", frame)

            # Log drone status only once after takeoff
            if not logged_status and 'e' in current_keys:
                log_drone_status(tello)
                logged_status = True

            # Get keyboard input for drone control
            vals = getKeyboardInput(tello)
            if vals is True:  # If drone has taken off, reset the logged status
                logged_status = False
            else:
                tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

            if capture_flag:
                capture_and_save_image(frame)  # Capture image
                capture_flag = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Process captured images after exiting the main loop
        for image_path in captured_images:
            analyze_image(predictor, image_path)  # Analyze images after all captures
        tello.streamoff()
        cv2.destroyAllWindows()
        listener.stop()

if __name__ == "__main__":
    main()
