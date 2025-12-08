# Lecture Automator

An automated system that generates lecture transcripts, summaries, asks reinforcing questions, and captures board content using Computer Vision and Audio Processing.

## Features
- **Smart Board Capture**: Uses Neon Tape ROI detection and Person Detection (YOLOv8) to capture the board only when the teacher moves away.
- **Audio Sync**: continuous audio recording synchronized with board snapshots.
- **Auto-Generation**: Generates transcripts (Whisper) and Study Guides (LLM).

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. **Record**: `python lecture_automator.py record`
2. **Generate**: `python lecture_automator.py generate`

See `walkthrough.md` for details.
