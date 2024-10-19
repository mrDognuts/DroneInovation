import cv2
import time
from djitellopy import Tello

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)
#tello.takeoff()
cv2.imwrite("picture.png", frame_read.frame)

try:
    while True:
        # Get the current frame from the Tello drone
        frame = tello.get_frame_read().frame

        # Resize the frame if necessary (optional)
        frame = cv2.resize(frame, (640, 480))

        # Display the frame using OpenCV
        cv2.imshow("Tello Live Feed", frame)

        # Press 'q' to break out of the loop and stop the feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Safely close everything
    tello.streamoff()  # Stop the video stream
    cv2.destroyAllWindows()  # Close the OpenCV window
#.land()