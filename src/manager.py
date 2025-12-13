import cv2
import json
import time
import os
import shutil
from .vision import ROIDetector, TeacherDetector, BoardMonitor
from .audio import AudioRecorder
from .logger import get_logger
from .config import config
from .state_manager import state_manager

logger = get_logger(__name__)

class LectureManager:
    def __init__(self, output_dir="output", color_mode="neon-green", audio_device_index=None, use_yolo=False):
        self.output_dir = output_dir
        self.audio_device_index = audio_device_index
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize Audio Recorder with selected device
        self.audio_recorder = AudioRecorder(
            output_filename=os.path.join(output_dir, "master_audio.wav"),
            input_device_index=self.audio_device_index
        )
        
        self.roi_detector = ROIDetector(color_mode=color_mode, use_yolo=use_yolo)
        self.teacher_detector = TeacherDetector()
        
        self.board_monitors = []
        self.manifest = []
        self.last_snapshot_time = 0
        self.start_time = 0
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        
    def calibrate_rois(self, frame_source=0):
        """
        Runs a brief loop to detect boards using tape.
        Press 'c' to capture calibration.
        """
        cap = cv2.VideoCapture(frame_source)
        
        # Try multiple resolutions (highest first)
        resolutions = [
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (640, 480)     # VGA (fallback)
        ]
        
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify it worked
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_w == width and actual_h == height:
                logger.info(f"Camera resolution set to {width}x{height}")
                break
            else:
                logger.warning(f"Failed to set {width}x{height}, got {actual_w}x{actual_h}")
        else:
            # None worked, use whatever we got
            logger.warning(f"Using default camera resolution: {actual_w}x{actual_h}")
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 1920, 1080)

        print("Starting Calibration. Point camera at the board with Neon Tape.")
        print("Press 'c' to lock in ROIs and start recording.")
        
        paused = False
        last_boards = []
        last_frame = None

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                
                # Show detected boards
                boards = self.roi_detector.find_boards(frame)
                last_boards = boards
                last_frame = frame.copy()
            else:
                # Keep showing the frozen frame
                frame = last_frame
                boards = last_boards
            
            debug_frame = frame.copy()
            for i, (x, y, w, h) in enumerate(boards):
                # Use Green for #1 (Default), Yellow for others
                color = (0, 255, 0) if i == 0 else (0, 255, 255)
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
                # Label
                cv2.putText(debug_frame, f"#{i+1}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            status_text = "PAUSED (Select now)" if paused else "Running (Press 'p' to Freeze)"
            cv2.putText(debug_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if paused else (255, 255, 255), 2)
            cv2.putText(debug_frame, "Press '1'-'5' to SELECT. 'a' for ALL.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            cv2.imshow("Calibration", debug_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Toggle Pause
            if key == ord('p') or key == ord(' '): # 'p' or Space
                paused = not paused
            
            # Select ALL
            elif key == ord('a') and len(boards) > 0:
                print(f"Calibration successful. {len(boards)} boards detected (ALL).")
                for board in boards:
                    self.board_monitors.append(BoardMonitor(board))
                break
            
            # Select Specific ID (1-5)
            elif key >= ord('1') and key <= ord('5'):
                idx = key - ord('1') # 0-indexed
                if idx < len(boards):
                     selected = boards[idx]
                     print(f"Calibration successful. Selected Board #{idx+1} ONLY.")
                     self.board_monitors.append(BoardMonitor(selected))
                     break
        
        cap.release()
        cv2.destroyAllWindows()
        return len(self.board_monitors) > 0
    
    def _check_disk_space(self, min_gb=1.0):
        """Check if sufficient disk space available for recording"""
        try:
            usage = shutil.disk_usage(self.output_dir)
            free_gb = usage.free / (1024**3)
            if free_gb < min_gb:
                print(f"\n⚠️ WARNING: Low disk space ({free_gb:.1f}GB free)")
                print(f"   Recommended: At least {min_gb}GB for recording")
                return False
            logger.info(f"Disk space OK: {free_gb:.1f}GB available")
            return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Proceed anyway if check fails

    def start_session(self, frame_source=0):
        # Check disk space before starting
        if not self._check_disk_space(min_gb=1.0):
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Recording cancelled due to low disk space.")
                return
        
        # 1. Start Audio - WAITING FOR USER INPUT
        # self.audio_recorder.start_recording()
        # self.start_time = time.time()
        
        cap = cv2.VideoCapture(frame_source)
        
        # Try multiple resolutions (highest first)
        resolutions = [
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (640, 480)     # VGA (fallback)
        ]
        
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify it worked
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_w == width and actual_h == height:
                logger.info(f"Camera resolution set to {width}x{height}")
                break
            else:
                logger.warning(f"Failed to set {width}x{height}, got {actual_w}x{actual_h}")
        else:
            # None worked, use whatever we got
            logger.warning(f"Using default camera resolution: {actual_w}x{actual_h}")
        
        cv2.namedWindow("Lecture Automator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lecture Automator", 1920, 1080)

        print("Session Ready. Press 'r' to START Recording, 'p' to PAUSE/RESUME, 'q' to STOP.")
        
        recording_active = False
        is_paused = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret: 
                    print("Error: Camera disconnected or EOF.")
                    break
                
                # --- State Handling ---
                if not recording_active:
                    # WAITING STATE
                    cv2.putText(frame, "READY: 'r' start | 'l' lock teacher | 'q' quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                else:
                    # RECORDING STATE
                    if is_paused:
                        # PAUSED
                         cv2.putText(frame, "PAUSED (Press 'p' or 'r' to Resume)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                         # Add Yellow Border
                         cv2.rectangle(frame, (0,0), (1920, 1080), (0, 255, 255), 10)
                    else:
                        # ACTIVE RECORDING
                        # Blinking Red Dot
                        if (time.time() % 1) < 0.5:
                            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 2. Detect Teacher (Prioritize those near boards)
                        board_rois = [(m.x, m.y, m.w, m.h) for m in self.board_monitors]
                        teacher_box, debug_info = self.teacher_detector.detect_teacher_with_debug(frame, board_rois)
                        
                        # DRAW ALL CANDIDATES (Gray) to help user understand who is who
                        if debug_info and "candidates" in debug_info:
                            for c in debug_info["candidates"]:
                                sim_txt = ""
                                if len(c) >= 8:
                                    cx, cy, cw, ch, _, cid, score, h_ratio = c
                                    sim_txt = f"|S:{score:.2f}|H:{h_ratio:.2f}"
                                else:
                                    cx, cy, cw, ch, _, cid = c[:6]

                                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (100, 100, 100), 1)
                                cv2.putText(frame, f"ID:{cid}{sim_txt}", (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                        # DRAW LOCKED TEACHER (Red/Green)
                        if teacher_box:
                            x1, y1, x2, y2 = teacher_box
                            color = (0, 0, 255) # Red (Default)
                            status_text = "TEACHER"
                            
                            if self.teacher_detector.locked_id is not None:
                                color = (0, 255, 0) # Green (Locked)
                                status_text = f"LOCKED #{self.teacher_detector.locked_id}"
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, status_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw Board Regions (Always visible)    
                for monitor in self.board_monitors:
                    color = (255, 0, 0) # Blue for empty
                    if monitor.state == "Writing": color = (0, 255, 255) # Yellow
                    elif monitor.state == "Left_Full": color = (0, 255, 0) # Green (Snapshot taken)
                    
                    cv2.rectangle(frame, (monitor.x, monitor.y), (monitor.x+monitor.w, monitor.y+monitor.h), color, 2)
                    cv2.putText(frame, monitor.state, (monitor.x, monitor.y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow("Lecture Automator", frame)
                key = cv2.waitKey(1) & 0xFF
                
                # --- Controls ---
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not recording_active:
                        print("User STARTED Recording.")
                        self.audio_recorder.start_recording()
                        self.start_time = time.time()
                        recording_active = True
                    elif is_paused:
                        print("User RESUMED Recording.")
                        self.audio_recorder.resume_recording()
                        is_paused = False
                        
                elif key == ord('p'):
                    if recording_active:
                        if is_paused:
                            print("User RESUMED Recording.")
                            self.audio_recorder.resume_recording()
                            is_paused = False
                        else:
                            print("User PAUSED Recording.")
                            self.audio_recorder.pause_recording()
                            is_paused = True
                            
                elif key == ord('s') and recording_active and not is_paused:
                    print("Manual Snapshot Triggered!")
                    for idx, monitor in enumerate(self.board_monitors):
                        self._save_snapshot(frame, idx, monitor)

                elif key == ord('l') or key == ord('L'):
                    # Lock onto current detected teacher (works with both lowercase and uppercase)
                    print(f"DEBUG: 'l' or 'L' pressed.")
                    board_rois = [(m.x, m.y, m.w, m.h) for m in self.board_monitors]
                    teacher_box = self.teacher_detector.detect_teacher(frame, board_rois)
                    
                    if teacher_box:
                        success = self.teacher_detector.set_reference_teacher(frame, teacher_box)
                        if success:
                            print(f"TEACHER LOCKED! (ID + Visual Memory). Tracking ID #{self.teacher_detector.locked_id}.")
                        else:
                             print("Lock Failed: Could not match ID.")
                    else:
                        print("Cannot lock: No teacher detected.")
                
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"\nCRITICAL ERROR: {e}")
            with open(os.path.join(self.output_dir, "crash_log.txt"), "a") as log:
                log.write(f"\n[{time.ctime()}] CRASH:\n{error_msg}\n")
                
        finally:
            self.stop_session()
            cap.release()
            cv2.destroyAllWindows()

    def _save_snapshot(self, frame, board_idx, monitor):
        # Save Image
        timestamp = time.time() - self.start_time
        img_filename = f"board_{board_idx+1}_{int(timestamp)}.jpg"
        img_path = os.path.join(self.output_dir, img_filename)
        
        # Crop to board or save full frame? Prompt implies board logic, but full frame is safer for context.
        # "Snapshot triggered" -> usually means capturing the board content.
        # Let's crop to the board + some padding.
        pad = 20
        h, w, c = frame.shape
        y1 = max(0, monitor.y - pad)
        y2 = min(h, monitor.y + monitor.h + pad)
        x1 = max(0, monitor.x - pad)
        x2 = min(w, monitor.x + monitor.w + pad)
        
        snapshot = frame[y1:y2, x1:x2]
        cv2.imwrite(img_path, snapshot)
        
        # Audio Segment Logic
        # [Previous Snapshot Time, Current Snapshot Time]
        # First snapshot: [0, current]
        # Subsequent: [last, current]
        
        segment_start = self.last_snapshot_time
        segment_end = timestamp
        self.last_snapshot_time = segment_end
        
        entry = {
            "board_id": board_idx + 1,
            "image_path": img_filename,
            "timestamp": timestamp,
            "audio_segment": [segment_start, segment_end]
        }
        self.manifest.append(entry)
        
        # Update JSON file incrementally
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=4)
            
    def stop_session(self):
        self.audio_recorder.stop_recording()
        print("Session Ended. Manifest saved.")
