# Lecture Automator

An automated system that generates lecture transcripts, summaries, asks reinforcing questions,gives recommendations on reading matterials associated with the lecture, and captures board content using Computer Vision and Audio Processing.

## Features

- **Smart Board Capture**: Uses Neon Tape ROI detection and Person Detection (YOLOv8) to capture the board only when the teacher moves away.
- **Audio Sync**: Continuous audio recording synchronized with board snapshots.
- **Auto-Generation**: Generates transcripts (using Faster-Whisper) and comprehensive Study Guides (using Google Gemini 2.0 Flash Exp).
- **Multi-Language**: Auto-detects lecture language and generates study materials in the matching language.

## Prerequisites

- **Python 3.10+**
- **FFmpeg**: Required for audio processing.
  - *Windows*: `winget install Gyan.FFmpeg`
- **Webcam**: For board capture.
- **Microphone**: For audio recording.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Efe-Deniz-Asan/Lecture-automator.git
   cd Lecture-automator
   ```

2. **Create a Virtual Environment (Recommended)**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## API Key Configuration

This tool requires a **Google Gemini API Key** to generate study guides and analyze board content.

### Setting the Environment Variable

**Windows (PowerShell)**:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

*To make it permanent, search for "Edit the system environment variables" in Windows Settings.*

**Linux/Mac**:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Usage

### 1. Start Recording

Run the recording script. This will open the camera feed.

```bash
.\start_recording.bat
# OR
python lecture_automator.py record
```

- **Calibration**: follow on-screen prompts to select the board area (ROI).
- **Recording**: Press `q` to stop recording.

### 2. Generate Notes

After recording, generate the study guide.

```bash
.\generate_notes.bat
# OR
python lecture_automator.py generate --latest
```

### 3. Fast Mode (Skip Transcription)

If you already have a transcript and just want to re-generate the study guide:

```bash
.\generate_notes_FAST.bat
```

## Troubleshooting

- **FFmpeg not found**: Ensure FFmpeg is installed and added to your system PATH. Restart your terminal after installing.
- **Camera issues**: Verify your camera source index (default is 0). You can change it with `--source 1`.

- ---

## License

Copyright (C) 2025 Efe Deniz Asan

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
