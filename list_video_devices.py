import os

# Suppress OpenCV warnings/errors
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2

def count_cameras():
    max_tested = 10
    available_cameras = []

    print(f"Scanning first {max_tested} indices for video devices...")
    
    for i in range(max_tested):
        # Try default backend first
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found Video Source at Index: {i}")
            
            # Try to get resolution
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  - Default Resolution: {int(w)}x{int(h)}")
            
            available_cameras.append(i)
            cap.release()
        else:
            cap.release()
    
    if not available_cameras:
        print("No cameras found.")
    else:
        print("\nAvailable Source IDs:", available_cameras)
        print("Use these IDs with the --source argument.")

if __name__ == "__main__":
    count_cameras()
