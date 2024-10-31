import cv2
import time
import os
from djitellopy import Tello
from pynput import keyboard as kp
from model_predictor import load_model, analyze_image  # Import functions from model_predictor

# Global variables
car_number = 1
capture_flag = False

# Initialize the predictor by loading the model
predictor = load_model("model_final.pth", "cpu")  # Adjust path and device as needed

# File paths for logs
drone_log_file = "drone_status_log.txt"
keypress_log_file = "keypress_log.txt"

def log_key_press(key):
    """Log each key press to keypress_log.txt."""
    with open(keypress_log_file, "a") as log:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Key pressed: {key}\n")

def capture_and_save_image(frame, output_dir="Dataset/before"):
    """Capture and save the image with an incremental filename."""
    global car_number
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_name = f"car{car_number}.1.png"
    image_path = os.path.join(output_dir, image_name)
    cv2.imwrite(image_path, frame)
    print(f"Captured {image_path}")

    car_number += 1  # Increment the car number after capturing the image
    return image_path

def on_press(key):
    """Handle key press events."""
    global capture_flag
    try:
        if key.char == 'c':  # Press 'c' to capture an image
            capture_flag = True
        log_key_press(key)
    except AttributeError:
        pass

def get_drone_status(tello):
    """Get and log drone status including battery, time, temperature, and height."""
    battery_level = tello.get_battery()
    flight_time = tello.get_flight_time()
    temperature = tello.get_temperature()
    status_info = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - "
        f"Battery: {battery_level}% | Time: {flight_time}s | Temp: {temperature}°C\n"
    )
    print(status_info)
    with open(drone_log_file, "a") as log:
        log.write(status_info)

def getKeyboardInput(tello):
    """Control drone movement using keyboard input and return movement values."""
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.is_pressed("h"):  # Move left
        lr = -speed
    elif kp.is_pressed("k"):  # Move right
        lr = speed
    if kp.is_pressed("u"):  # Move forward
        fb = speed
    elif kp.is_pressed("j"):  # Move backward
        fb = -speed
    if kp.is_pressed("w"):  # Move up
        ud = speed
    elif kp.is_pressed("s"):  # Move down
        ud = -speed
    if kp.is_pressed("a"):  # Rotate left
        yv = -speed
    elif kp.is_pressed("d"):  # Rotate right
        yv = speed
    if kp.is_pressed("q"):  # Land
        tello.land()
        time.sleep(3)
    if kp.is_pressed("e"):  # Take off
        tello.takeoff()
    
    return [lr, fb, ud, yv]

def main():
    global capture_flag
    tello = Tello()

    try:
        tello.connect()
        print("Connected to Tello...")

        # Log initial drone status
        get_drone_status(tello)
        tello.streamon()
        time.sleep(2)
        print("Started streaming...")

        # Start listener for key presses
        listener = kp.Listener(on_press=on_press)
        listener.start()

        while True:
            frame = tello.get_frame_read().frame
            cv2.imshow("Tello Live Feed", frame)

            # Get keyboard input for drone control
            vals = getKeyboardInput(tello)
            tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

            if capture_flag:
                image_path = capture_and_save_image(frame)
                analyze_image(predictor, image_path)  # Use analyze_image with the loaded predictor
                capture_flag = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        tello.land()

    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        listener.stop()

if __name__ == "__main__":
    main()
