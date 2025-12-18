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

import cv2
import numpy as np

def nothing(x):
    pass

def main():
    print("Starting Color Tuner...")
    print("Adjust the sliders until your board boundaries are WHITE and everything else is BLACK.")
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow('Result')
    cv2.namedWindow('Settings')
    
    # Create Trackbars
    # Starting values for "Red" roughly
    cv2.createTrackbar('Low H', 'Settings', 0, 179, nothing)
    cv2.createTrackbar('High H', 'Settings', 179, 179, nothing)
    
    cv2.createTrackbar('Low S', 'Settings', 70, 255, nothing)
    cv2.createTrackbar('High S', 'Settings', 255, 255, nothing)
    
    cv2.createTrackbar('Low V', 'Settings', 50, 255, nothing)
    cv2.createTrackbar('High V', 'Settings', 255, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Get current positions of trackbars
        lh = cv2.getTrackbarPos('Low H', 'Settings')
        uh = cv2.getTrackbarPos('High H', 'Settings')
        ls = cv2.getTrackbarPos('Low S', 'Settings')
        us = cv2.getTrackbarPos('High S', 'Settings')
        lv = cv2.getTrackbarPos('Low V', 'Settings')
        uv = cv2.getTrackbarPos('High V', 'Settings')
        
        # Determine arrays
        lower = np.array([lh, ls, lv])
        upper = np.array([uh, us, uv])
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Show result
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Stack images for easier viewing (Downscale frame)
        scale = 0.5
        small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        small_res = cv2.resize(result, (0,0), fx=scale, fy=scale)
        small_mask = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (0,0), fx=scale, fy=scale)
        
        # Concatenate
        combined = np.hstack((small_frame, small_mask, small_res))
        
        cv2.imshow('Result', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nFinal Settings:")
            print(f"Lower: [{lh}, {ls}, {lv}]")
            print(f"Upper: [{uh}, {us}, {uv}]")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
