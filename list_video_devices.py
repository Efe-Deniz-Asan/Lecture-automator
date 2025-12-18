# ------------------------------------------------------------------------------
#  Copyright (c) 2025 Efe Deniz Asan
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

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
