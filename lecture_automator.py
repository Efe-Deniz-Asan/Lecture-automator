import argparse
import sys
from src.manager import LectureManager
from src.generator import ContentGenerator

def main():
    # AUTO-FIX: Add FFmpeg to PATH if missing (Common issue with Winget installs needing restart)
    import shutil
    import os
    if not shutil.which("ffmpeg"):
        # Try to find it in the standard Winget location
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        possible_path = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe", "ffmpeg-8.0.1-full_build", "bin")
        if os.path.exists(possible_path):
             print(f"DEBUG: Found FFmpeg locally at {possible_path}. Adding to PATH.")
             os.environ["PATH"] += os.pathsep + possible_path

    parser = argparse.ArgumentParser(description="Lecture Automator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Calibrate Command
    parser_calib = subparsers.add_parser("calibrate", help="Calibrate ROI Detection using Neon Tape or Chalk")
    parser_calib.add_argument("--source", type=str, default="0", help="Video source ID (default: 0) or camera name (e.g., 'Logitech Webcam C920')")
    parser_calib.add_argument("--color", type=str, default="neon-green", choices=["neon-green", "chalk-green", "chalk-blue", "chalk-red"], 
                              help="Color of the boundary markers (default: neon-green)")
    parser_calib.add_argument("--use-yolo", action="store_true", help="Enable Hybrid YOLO+Color detection for Boards")

    
    # Record Command
    parser_rec = subparsers.add_parser("record", help="Start recording a lecture session")
import argparse
import sys
from src.manager import LectureManager
from src.generator import ContentGenerator

def main():
    # AUTO-FIX: Load .env file for API Keys
    from dotenv import load_dotenv
    load_dotenv()
    
    # AUTO-FIX: Add FFmpeg to PATH if missing (Common issue with Winget installs needing restart)
    import shutil
    import os
    if not shutil.which("ffmpeg"):
        # Try to find it in the standard Winget location
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        possible_path = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe", "ffmpeg-8.0.1-full_build", "bin")
        if os.path.exists(possible_path):
             print(f"DEBUG: Found FFmpeg locally at {possible_path}. Adding to PATH.")
             os.environ["PATH"] += os.pathsep + possible_path

    parser = argparse.ArgumentParser(description="Lecture Automator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Calibrate Command
    parser_calib = subparsers.add_parser("calibrate", help="Calibrate ROI Detection using Neon Tape or Chalk")
    parser_calib.add_argument("--source", type=str, default="0", help="Video source ID (default: 0) or camera name (e.g., 'Logitech Webcam C920')")
    parser_calib.add_argument("--color", type=str, default="neon-green", choices=["neon-green", "chalk-green", "chalk-blue", "chalk-red"], 
                              help="Color of the boundary markers (default: neon-green)")
    parser_calib.add_argument("--use-yolo", action="store_true", help="Enable Hybrid YOLO+Color detection for Boards")

    
    # Record Command
    parser_rec = subparsers.add_parser("record", help="Start recording a lecture session")
    parser_rec.add_argument("--source", type=str, default="0", help="Video source ID (default: 0) or camera name (e.g., 'Logitech Webcam C920')")
    parser_rec.add_argument("--output", type=str, default="output", help="Output directory")
    parser_rec.add_argument("--color", type=str, default="neon-green", choices=["neon-green", "chalk-green", "chalk-blue", "chalk-red"], 
                              help="Color of the boundary markers (default: neon-green)")
    parser_rec.add_argument("--audio-device", type=int, default=None, help="Audio Input Device ID (Use list_audio_devices.py to find ID)")
    parser_rec.add_argument("--use-yolo", action="store_true", help="Enable Hybrid YOLO+Color detection for Boards (More reliable)")
    
    # Generate Command
    parser_gen = subparsers.add_parser("generate", help="Generate transcript and study materials from recorded session")
    parser_gen.add_argument("--output", type=str, default="output", help="Output directory containing manifest.json")
    parser_gen.add_argument("--model", type=str, default="turbo", help="Whisper model size (tiny, base, small, medium, large-v3, turbo)")
    parser_gen.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for inference (default: auto)")
    parser_gen.add_argument("--compute-type", type=str, default="default", help="Compute type (default, float16, int8_float16, int8)")
    parser_gen.add_argument("--language", type=str, default=None, help="Language code (default: Auto-detect)")
    parser_gen.add_argument("--openai-key", type=str, help="OpenAI API Key for Summary/Questions")
    parser_gen.add_argument("--gemini-key", type=str, help="Google Gemini API Key for Summary/Questions")
    parser_gen.add_argument("--skip-transcription", action="store_true", help="Skip Whisper and use existing transcript.txt")
    parser_gen.add_argument("--latest", action="store_true", help="Automatically process the most recent Lecture folder inside output/")
    
    args = parser.parse_args()
    
    import datetime
    import glob
    
    try:
        if args.command == "calibrate":
            # Simple Source Handling
            try:
                camera_source = int(args.source)
            except ValueError:
                camera_source = args.source

            manager = LectureManager(color_mode=args.color, use_yolo=args.use_yolo)
            manager.calibrate_rois(frame_source=camera_source)
            
        elif args.command == "record":
            # Simple Source Handling
            try:
                camera_source = int(args.source)
            except ValueError:
                camera_source = args.source

            # AUTO-FOLDER LOGIC:
            # If output is 'output', we treat it as ROOT and create 'output/Lecture-N-Date-Time'
            root_dir = args.output
            
            # Simple check: If user specified a nested path explicitly, respect it? 
            # Logic: If args.output is 'output', we auto-create subfolder.
            # If args.output is 'output/MyLecture', we use that.
            
            target_dir = root_dir
            if os.path.basename(os.path.normpath(root_dir)) == "output" or root_dir == ".": 
                # Create subfolder
                if not os.path.exists(root_dir): os.makedirs(root_dir)
                
                # Scan for existing Lecture-N folders to increment ID
                existing_folders = glob.glob(os.path.join(root_dir, "Lecture-*"))
                max_id = 0
                for f in existing_folders:
                    try:
                        name = os.path.basename(f)
                        parts = name.split("-")
                        # Expected: Lecture-1-2023...
                        if len(parts) >= 2 and parts[1].isdigit():
                            lid = int(parts[1])
                            if lid > max_id: max_id = lid
                    except: pass
                
                next_id = max_id + 1
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                folder_name = f"Lecture-{next_id}-{timestamp}"
                target_dir = os.path.join(root_dir, folder_name)
                print(f"Auto-Detected Session Mode: Saving to '{target_dir}'")
            
            # Resolve Audio Device
            audio_id = args.audio_device
            if audio_id is None:
                env_id = os.getenv("AUDIO_DEVICE_ID")
                if env_id and env_id.strip():
                    try:
                        audio_id = int(env_id)
                        print(f"Using Audio Device from .env: {audio_id}")
                    except ValueError:
                        print(f"Warning: Invalid AUDIO_DEVICE_ID in .env: {env_id}")
            
            manager = LectureManager(output_dir=target_dir, color_mode=args.color, audio_device_index=audio_id, use_yolo=args.use_yolo)
            
            if manager.calibrate_rois(frame_source=camera_source):
                manager.start_session(frame_source=camera_source)
            else:
                print("Calibration failed or cancelled. Exiting.")
                
        elif args.command == "generate":
            target_dir = args.output
            
            # Auto-Find Latest Logic
            # Condition: If 'manifest.json' NOT in target_dir, OR --latest flag used
            manifest_exists = os.path.exists(os.path.join(target_dir, "manifest.json"))
            
            if args.latest or (not manifest_exists and os.path.exists(target_dir)):
                 # Try to find subfolders
                 subfolders = [f.path for f in os.scandir(target_dir) if f.is_dir()]
                 if subfolders:
                     # Sort by modification time (creation time approx)
                     latest_folder = max(subfolders, key=os.path.getmtime)
                     print(f"Auto-Selected Latest Session: {latest_folder}")
                     target_dir = latest_folder
                 else:
                     if not manifest_exists:
                         print(f"Error: No manifest.json found in '{target_dir}' and no subfolders found.")
                         sys.exit(1)

            generator = ContentGenerator(output_dir=target_dir, model_size=args.model, 
                                         device=args.device, compute_type=args.compute_type,
                                         openai_key=args.openai_key, gemini_key=args.gemini_key,
                                         language=args.language, skip_transcription=args.skip_transcription)
            generator.process_session()
            
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting safely...")
        sys.exit(0)


if __name__ == "__main__":
    main()