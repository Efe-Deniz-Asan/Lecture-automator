import cv2
import numpy as np
from ultralytics import YOLO

class ROIDetector:
    def __init__(self, color_mode="neon-green", use_yolo=False, yolo_model="yolov8n.pt"):
        self.use_yolo = use_yolo
        self.yolo_model = None
        if self.use_yolo:
             print(f"Loading YOLO model for Board Detection: {yolo_model}")
             try:
                self.yolo_model = YOLO(yolo_model)
             except Exception as e:
                print(f"WARNING: Failed to load YOLO model: {e}")
                self.use_yolo = False

        # HSV Ranges
        # Neon Green: Bright, High Saturation
        if color_mode == "neon-green":
            self.lower = np.array([40, 100, 100], dtype=np.uint8)
            self.upper = np.array([80, 255, 255], dtype=np.uint8)
        
        # Chalk Green: Similar Hue, but potentially lower saturation/value
        elif color_mode == "chalk-green":
            self.lower = np.array([35, 50, 50], dtype=np.uint8)
            self.upper = np.array([85, 255, 255], dtype=np.uint8)
            
        # Chalk Blue: Hue around 100-120
        elif color_mode == "chalk-blue":
            self.lower = np.array([90, 50, 50], dtype=np.uint8)
            self.upper = np.array([130, 255, 255], dtype=np.uint8)
            
        # Chalk Red: User Calibrated Values
        elif color_mode == "chalk-red":
            self.lower = np.array([170, 61, 137], dtype=np.uint8)
            self.upper = np.array([179, 160, 243], dtype=np.uint8)
            # Second range set identical to first since user calibration didn't find the 0-10 range
            self.lower_2 = np.array([170, 61, 137], dtype=np.uint8)
            self.upper_2 = np.array([179, 160, 243], dtype=np.uint8)
            
        else:
            # Default to neon
            print(f"Unknown color mode '{color_mode}'. Defaulting to neon-green.")
            self.lower = np.array([40, 100, 100], dtype=np.uint8)
            self.upper = np.array([80, 255, 255], dtype=np.uint8)
            
        self.color_mode = color_mode

    def find_boards(self, frame):
        """
        Detects rectangular boards using Hybrid Approach (YOLO Region + Color Tape).
        Returns a list of bounding boxes (x, y, w, h).
        """
        candidates = []
        
        # 1. YOLO DETECTION (Direct Candidate)
        if self.use_yolo and self.yolo_model:
            # Classes: 62=TV, 63=Laptop (COCO)
            results = self.yolo_model(frame, verbose=False, classes=[62, 63])
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    # Add as candidate
                    print(f"DEBUG: YOLO Found Board-like Object ({w}x{h})")
                    candidates.append((x1, y1, w, h, area))

        # 2. COLOR DETECTION (Tape/Chalk)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        
        if self.color_mode == "chalk-red":
             mask2 = cv2.inRange(hsv, self.lower_2, self.upper_2)
             mask = cv2.bitwise_or(mask, mask2)
        
        # Morphological ops
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            
            if w > 50 and h > 50 and area > 2000: 
                candidates.append((x, y, w, h, area))
        
        # 3. MERGE & FILTER DUPLICATES (Non-Max Suppression-ish)
        # Sort by area (largest first)
        candidates.sort(key=lambda c: c[4], reverse=True)
        
        final_boards = []
        for c in candidates:
            cx, cy, cw, ch, carea = c
            is_new = True
            for existing in final_boards:
                ex, ey, ew, eh, earea = existing
                
                # Check Intersection
                ix = max(cx, ex)
                iy = max(cy, ey)
                iw = min(cx+cw, ex+ew) - ix
                ih = min(cy+ch, ey+eh) - iy
                
                if iw > 0 and ih > 0:
                    intersection = iw * ih
                    # If high overlap with a larger (already added) box, ignore this one
                    # Use IoU or Intersection/SmallestArea
                    iou = intersection / min(carea, earea)
                    if iou > 0.5: # 50% overlap implies same object
                        is_new = False
                        break
            
            if is_new:
                # Add (x, y, w, h) only
                final_boards.append((cx, cy, cw, ch, carea))
        
        # Return top 5, stripped of area
        return [b[:4] for b in final_boards[:5]]


class TeacherDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.locked_id = None # ID of the tracked teacher
        self.ref_hist = None  # Visual Memory (Histogram)
        self.missing_frames = 0
        
    def set_reference_teacher(self, frame, box):
        """
        Locks onto the ID AND Appearance of the person closest to the provided box.
        """
        # We need to run tracking on this frame to get IDs
        results = self.model.track(frame, verbose=False, classes=[0], persist=True)
        
        target_cx = (box[0] + box[2]) / 2
        target_cy = (box[1] + box[3]) / 2
        
        best_id = None
        best_box = None
        min_dist = float('inf')
        
        for result in results:
            if not result.boxes or not result.boxes.id: continue
            
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            
            for i, b in enumerate(boxes):
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                dist = np.sqrt((cx - target_cx)**2 + (cy - target_cy)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_id = ids[i]
                    best_box = map(int, b)
        
        if min_dist < 200 and best_box is not None:
            self.locked_id = best_id
            self.missing_frames = 0
            
            # --- CAPTURE VISUAL MEMORY ---
            bx1, by1, bx2, by2 = best_box
            person_roi = frame[by1:by2, bx1:bx2]
            if person_roi.size > 0:
                 hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                 hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
                 cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                 self.ref_hist = hist
                 print(f"Teacher Locked! ID: #{self.locked_id} | Visual Memory: Saved")
            else:
                 print(f"Teacher Locked! ID: #{self.locked_id} | Visual Memory: FAILED (Empty ROI)")
            
            return True
        else:
            print("Failed to match Teacher ID.")
            return False

    def detect_teacher(self, frame, board_rois=None):
        """
        Returns bounding box [x1, y1, x2, y2] of the teacher.
        Uses model.track() to maintain ID across frames.
        """
        # Run Tracking
        results = self.model.track(frame, verbose=False, classes=[0], persist=True)
        
        candidates = []
        for result in results:
            if not result.boxes: continue
            
            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
            else:
                ids = [-1] * len(result.boxes)
            
            boxes = result.boxes.xyxy.cpu().tolist()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                area = w * h
                tid = ids[i]
                candidates.append((x1, y1, x2, y2, area, tid))
        
        # --- MODE 1: ID LOCK ---
        if self.locked_id is not None:
            for cand in candidates:
                if cand[5] == self.locked_id:
                    self.missing_frames = 0 # Reset counter
                    return cand[:4]
            
            # Locked ID not found
            self.missing_frames += 1
            if self.missing_frames > 150: # 5s lost
                print(f"Teacher #{self.locked_id} lost. Switching to VISUAL SEARCH.")
                self.locked_id = None
                self.missing_frames = 0
                # Fall through to Visual Search (ref_hist is still set)
            else:
                return None # Waiting for ID to return
        
        # --- MODE 2: VISUAL SEARCH (Re-Acquire) ---
        if self.ref_hist is not None:
            # We have a memory, but no active ID. Look for a visual match.
            best_person = None
            max_similarity = -1.0
            best_id_match = None
            
            for person in candidates:
                px1, py1, px2, py2, p_area, pid = person
                if p_area < 2000: continue
                
                cand_roi = frame[py1:py2, px1:px2]
                if cand_roi.size == 0: continue
                
                hsv_cand = cv2.cvtColor(cand_roi, cv2.COLOR_BGR2HSV)
                cand_hist = cv2.calcHist([hsv_cand], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(cand_hist, cand_hist, 0, 255, cv2.NORM_MINMAX)
                
                similarity = cv2.compareHist(self.ref_hist, cand_hist, cv2.HISTCMP_CORREL)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_person = (px1, py1, px2, py2)
                    best_id_match = pid
            
            if max_similarity > 0.5: # Strict visual match
                print(f"Teacher RE-IDENTIFIED! New ID: #{best_id_match} (Sim: {max_similarity:.2f})")
                self.locked_id = best_id_match
                return best_person
            
            # If we have a memory but no match, DO NOT FALLBACK to spatial.
            # We want to ignore students walking by until the REAL teacher returns.
            return None
            
        # --- MODE 2: SPATIAL HEURISTIC (Default) ---
        if not candidates: return None

        if board_rois and len(board_rois) > 0:
            best_person = None
            min_dist_score = float('inf')
            
            board_centers_y = [b[1] + b[3]/2 for b in board_rois]
            avg_board_y = sum(board_centers_y) / len(board_centers_y)
            
            for person in candidates:
                px1, py1, px2, py2, p_area, tid = person
                p_cy = py1 + (py2 - py1) / 2
                
                total_intersection = 0
                for (bx, by, bw, bh) in board_rois:
                    ix = max(px1, bx)
                    iy = max(py1, by)
                    iw = min(px2, bx+bw) - ix
                    ih = min(py2, by+bh) - iy
                    if iw > 0 and ih > 0:
                        total_intersection += (iw * ih)
                
                if total_intersection > 0:
                    score = -total_intersection 
                else:
                    dist_y = abs(p_cy - avg_board_y)
                    score = dist_y
                
                if score < min_dist_score:
                    min_dist_score = score
                    best_person = (px1, py1, px2, py2)
            
            return best_person
            
        else:
            # Fallback: Largest + Central
            frame_h = frame.shape[0]
            valid_candidates = []
            for p in candidates:
                cy = p[1] + (p[3] - p[1]) / 2
                if cy < frame_h * 0.85: 
                    valid_candidates.append(p)
            
            if not valid_candidates: return None
            valid_candidates.sort(key=lambda x: x[4], reverse=True)
            return valid_candidates[0][:4]

class BoardMonitor:
    def __init__(self, board_roi):
        """
        board_roi: (x, y, w, h)
        """
        self.x, self.y, self.w, self.h = board_roi
        self.state = "Empty" # Empty, Writing, Left, Full
        self.last_status = "Empty"
        
    def calculate_iou(self, teacher_box):
        if not teacher_box:
            return 0.0
            
        t_x1, t_y1, t_x2, t_y2 = teacher_box
        b_x1, b_y1, b_x2, b_y2 = self.x, self.y, self.x + self.w, self.y + self.h
        
        # Intersection coordinates
        x_left = max(t_x1, b_x1)
        y_top = max(t_y1, b_y1)
        x_right = min(t_x2, b_x2)
        y_bottom = min(t_y2, b_y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # We generally care about how much the TEACHER overlaps the BOARD
        # But standard IoU is Intersection / Union. The prompt asks for Overlap.
        # "Calculate how much the Teacher's box overlaps with the Board's ROI."
        # This usually means Interference. Let's use Intersection / TeacherArea or Intersection / Union.
        # Let's stick to standard IoU or Intersection / BoardArea to see if board is blocked.
        # Prompt says: "IF Overlap > 30% -> Status = Writing".
        # I'll use Intersection / TeacherArea to see if the teacher is "on" the board.
        
        teacher_area = (t_x2 - t_x1) * (t_y2 - t_y1)
        if teacher_area == 0: return 0
        
        return intersection_area / teacher_area

    def check_fullness(self, frame):
        # Placeholder for Custom CNN
        # Use edge detection as a heuristic for now
        roi = frame[self.y:self.y+self.h, self.x:self.x+self.w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (self.w * self.h)
        
        # HEURISTIC: precise value depends on distance, resolution, etc.
        # Returns True if "Full", False otherwise.
        print(f"DEBUG: Board Edge Density: {edge_density:.2f}") # Debug info
        return edge_density > 2.0 # Lowered from 20 for higher sensitivity

    def update(self, frame, teacher_box):
        overlap = self.calculate_iou(teacher_box)
        
        current_status = "Empty"
        if overlap > 0.3:
            current_status = "Writing"
        else:
            current_status = "No Overlap"
            
        trigger_snapshot = False
        
        # State Transition Logic
        # IF Status changes from "Writing" to "No Overlap" -> Check Fullness
        if self.last_status == "Writing" and current_status == "No Overlap":
            # Teacher just left the board
            val = self.check_fullness(frame)
            if val:
                 trigger_snapshot = True
                 self.state = "Left_Full"
            else:
                 self.state = "Left_NotFull"
                 
        self.last_status = current_status
        return trigger_snapshot

