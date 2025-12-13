import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

class ROIDetector:
    def __init__(self, color_mode="neon-green", use_yolo=False, yolo_model="yolov8n.pt"):
        self.use_yolo = use_yolo
        self.yolo_model = None
        if self.use_yolo:
             logger.info(f"Loading YOLO model for Board Detection: {yolo_model}")
             try:
                self.yolo_model = YOLO(yolo_model)
             except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}")
                self.use_yolo = False
        
        # Device Check
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        if self.use_yolo and self.device == 0:
            logger.info(f"ROIDetector using GPU: {torch.cuda.get_device_name(0)}")

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
            results = self.yolo_model(frame, verbose=False, classes=[62, 63], device=self.device)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    # Add as candidate
                    # print(f"DEBUG: YOLO Found Board-like Object ({w}x{h})")
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
        self.ref_height = None # Reference Height (pixels) for validation
        self.missing_frames = 0
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        if self.device == 0:
             logger.info(f"TeacherDetector using GPU: {torch.cuda.get_device_name(0)}")
        
    def set_reference_teacher(self, frame, box):
        """
        Locks onto the ID AND Appearance of the person closest to the provided box.
        """
        # We need to run tracking on this frame to get IDs
        results = self.model.track(frame, verbose=False, classes=[0], persist=True, device=self.device)
        
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
                    best_box = map(int, b) # x1, y1, x2, y2
        
        # Relaxed threshold for initial lock
        if min_dist < 500 and best_box is not None:
            self.locked_id = best_id
            self.missing_frames = 0
            
            # --- CAPTURE VISUAL & SPATIAL MEMORY ---
            bx1, by1, bx2, by2 = list(best_box)
            self.ref_height = by2 - by1 # Store Height
            
            person_roi = frame[by1:by2, bx1:bx2]
            if person_roi.size > 0:
                 hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                 hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
                 cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                 self.ref_hist = hist
                 logger.info(f"Teacher Locked! ID: #{self.locked_id} | H: {self.ref_height}px | Visual Memory: Saved")
            else:
                 logger.warning(f"Teacher Locked! ID: #{self.locked_id} | Visual Memory: FAILED (Empty ROI)")
            
            return True
        else:
            logger.warning(f"Lock Failed. Min Dist: {min_dist:.1f}")
            return False

    def detect_teacher(self, frame, board_rois=None):
        # Backward compatibility wrapper
        box, _ = self.detect_teacher_with_debug(frame, board_rois)
        return box

    def detect_teacher_with_debug(self, frame, board_rois=None):
        """
        Returns (bounding box [x1, y1, x2, y2], debug_info_dict)
        """
        # Run Tracking
        results = self.model.track(frame, verbose=False, classes=[0], persist=True, device=self.device)
        
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
                candidates.append((x1, y1, w, h, area, tid))
        
        debug_info = {"candidates": candidates}
        
        # --- MODE 1: ID LOCK ---
        if self.locked_id is not None:
            for cand in candidates:
                if cand[5] == self.locked_id:
                    self.missing_frames = 0
                    x, y, w, h = cand[:4]
                    # Update ref_height dynamically if matched? No, keep original as baseline or moving avg?
                    # Let's keep original to avoid drift to seated students.
                    return [x, y, x+w, y+h], debug_info
            
            # Locked ID not found
            self.missing_frames += 1
            if self.missing_frames > config.vision.locked_teacher_timeout_frames:
                logger.info(f"Teacher #{self.locked_id} lost. Switching to VISUAL SEARCH.")
                self.locked_id = None
                self.missing_frames = 0
            else:
                return None, debug_info
        
        # --- MODE 2: VISUAL + HEIGHT SEARCH (Re-Acquire or Debug) ---
        # Even if we have a lock, we might want to see the scores for debugging
        debug_scores = [] 
        
        if self.ref_hist is not None:
            best_person = None
            max_similarity = -1.0
            best_id_match = None
            
            for i, person in enumerate(candidates):
                px, py, pw, ph, p_area, pid = person
                
                # Height Ratio
                h_ratio = 0.0
                if self.ref_height:
                    h_ratio = ph / self.ref_height

                if p_area < 2000: 
                    debug_scores.append(f"Small")
                    continue
                
                # Bounds check
                px_c = max(0, px); py_c = max(0, py)
                px2 = min(frame.shape[1], px+pw)
                py2 = min(frame.shape[0], py+ph)
                
                cand_roi = frame[py_c:py2, px_c:px2]
                if cand_roi.size == 0: 
                    debug_scores.append("Empty")
                    continue
                
                hsv_cand = cv2.cvtColor(cand_roi, cv2.COLOR_BGR2HSV)
                cand_hist = cv2.calcHist([hsv_cand], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(cand_hist, cand_hist, 0, 255, cv2.NORM_MINMAX)
                
                similarity = cv2.compareHist(self.ref_hist, cand_hist, cv2.HISTCMP_CORREL)
                
                # Penalize Lower Frame (Audience)
                penalty = 0.0
                p_cy = py + ph/2
                if p_cy > frame.shape[0] * (1 - config.vision.upper_frame_percentage):
                     penalty = config.vision.audience_penalty
                     similarity -= penalty
                
                # Update candidate tuple with Score for manager to draw: (..., pid, score, h_ratio)
                # We need to modify the candidates list structure in debug_info
                # Current: (x, y, w, h, area, tid) -> New: (x,y,w,h,area,tid, sim, h_ratio)
                candidates[i] = (px, py, pw, ph, p_area, pid, similarity, h_ratio)

                # Logic only if we are searching (Mode 2)
                if self.locked_id is None:
                    # Height Filtering
                    if self.ref_height and ph < (self.ref_height * config.vision.height_ratio_threshold):
                        continue
                        
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_person = [px, py, px+pw, py+ph]
                        best_id_match = pid
            
            # Use threshold
            if self.locked_id is None and max_similarity > config.vision.similarity_threshold:
                logger.info(f"Teacher RE-IDENTIFIED! New ID: #{best_id_match} (Sim: {max_similarity:.2f})")
                self.locked_id = best_id_match
                # We return here, but debug_info is updated by reference in list? No, local list.
                # Update debug info with enriched candidates
                
        debug_info["candidates"] = candidates
        
        # --- MODE 1 RETURN (After scoring) ---
        if self.locked_id is not None:
             for cand in candidates:
                 # Check ID (cand[5] is still ID)
                 if len(cand) > 5 and cand[5] == self.locked_id:
                     self.missing_frames = 0
                     x, y, w, h = cand[:4]
                     return [x, y, x+w, y+h], debug_info
        
        # --- MODE 2 RETURN ---
        if self.locked_id is None and self.ref_hist is not None:
            if best_person: # Found via visual match loop above
                 return best_person, debug_info
            return None, debug_info

        # --- MODE 3: SPATIAL HEURISTIC (No Lock, No Memory) ---
        if not candidates: return None, debug_info
        
        # Priority: Proximity to Boards > Upper Frame > Largest
        valid_candidates = [c for c in candidates if (c[1] + c[3]/2) < frame.shape[0] * 0.85] # Upper 85%
        
        if not valid_candidates: 
            return None, debug_info
            
        valid_candidates.sort(key=lambda x: x[4], reverse=True) # Sort by Area
        top = valid_candidates[0]
        return [top[0], top[1], top[0]+top[2], top[1]+top[3]], debug_info

class BoardMonitor:
    def __init__(self, board_roi):
        """
        board_roi: (x, y, w, h)
        """
        self.x, self.y, self.w, self.h = board_roi
        self.state = "Empty" # Empty, Writing, Left, Full
        self.last_status = "Empty"
        self.intersection_start_time = None  # Track when teacher started writing
        self.is_valid_session = False  # Track if minimum duration met
        self.last_snapshot_time = 0  # Track periodic snapshots during continuous writing
        
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
        # print(f"DEBUG: Board Edge Density: {edge_density:.2f}") # Debug info
        return edge_density > 2.0 # Lowered from 20 for higher sensitivity

    def update(self, frame, teacher_box):
        overlap = self.calculate_iou(teacher_box)
        
        current_status = "Empty"
        if overlap > 0.3:
            current_status = "Writing"
        else:
            current_status = "No Overlap"
            
        trigger_snapshot = False
        
        # Start timing when teacher begins writing
        if current_status == "Writing" and self.last_status != "Writing":
            # Teacher just started intersecting
            self.intersection_start_time = time.monotonic()  # Use monotonic for clock-independent timing
            self.is_valid_session = False
            logger.debug(f"Board intersection started")
        
        # Check if minimum duration has been met
        if current_status == "Writing" and self.intersection_start_time is not None:
            duration = time.monotonic() - self.intersection_start_time
            if duration >= config.recording.min_intersection_duration_seconds:
                if not self.is_valid_session:  # Only log once when threshold is reached
                    logger.info(f"Valid writing session detected (duration: {duration:.1f}s >= {config.recording.min_intersection_duration_seconds}s)")
                self.is_valid_session = True
                
                # Fix #1: Periodic snapshot for continuous writing (every 5 minutes)
                time_since_snapshot = time.monotonic() - self.last_snapshot_time
                if time_since_snapshot >= 300:  # 5 minutes
                    if self.check_fullness(frame):
                        trigger_snapshot = True
                        self.last_snapshot_time = time.monotonic()
                        self.state = "Writing_Snapshot"
                        logger.info(f"Periodic snapshot triggered (continuous writing for {time_since_snapshot:.0f}s)")
        
        # State Transition Logic
        # IF Status changes from "Writing" to "No Overlap" -> Check Fullness (if valid session)
        if self.last_status == "Writing" and current_status == "No Overlap":
            # Teacher just left the board
            if self.is_valid_session:
                # Valid writing session - check fullness and potentially trigger snapshot
                val = self.check_fullness(frame)
                if val:
                    trigger_snapshot = True
                    self.state = "Left_Full"
                    logger.info("Snapshot triggered after valid writing session")
                else:
                    self.state = "Left_NotFull"
                    logger.debug("Valid session but board not full enough")
            else:
                # Brief intersection - ignore
                duration = time.monotonic() - self.intersection_start_time if self.intersection_start_time else 0
                self.state = "Left_TooShort"
                logger.debug(f"Brief intersection ignored (duration: {duration:.1f}s < {config.recording.min_intersection_duration_seconds}s)")
            
            # Reset timer
            self.intersection_start_time = None
            self.is_valid_session = False
                 
        self.last_status = current_status
        return trigger_snapshot

