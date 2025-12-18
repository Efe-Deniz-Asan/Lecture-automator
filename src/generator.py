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

import json
import os
from faster_whisper import WhisperModel
from openai import OpenAI
import PIL.Image
from src.ocr import BoardOCR
from src.config import config

class ContentGenerator:
    def __init__(self, output_dir="output", model_size="medium", device="auto", compute_type="default",
                 openai_key=None, gemini_key=None, language=None, skip_transcription=False):
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.audio_path = os.path.join(output_dir, "master_audio.wav")
        self.language = language
        self.skip_transcription = skip_transcription
        self.model_size = model_size
        
        # Initialize clients
        self.openai_key = openai_key if openai_key else os.environ.get("OPENAI_API_KEY")
        self.gemini_key = gemini_key if gemini_key else os.environ.get("GEMINI_API_KEY")
        
        self.openai_client = None
        self.gemini_model = None

        if self.openai_key:
            self.openai_client = OpenAI(api_key=self.openai_key)
            print("OpenAI API Key detected.")
        
        if self.gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            
            # --- Model: The Professor (Gemini 2.0 Flash Exp) ---
            # Reverting to single model mode without search tools (for stability).
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            print("Gemini API Key detected. Model: Gemini 2.0 Flash Exp (No Search)")

        if not self.skip_transcription:
             print(f"Loading Faster-Whisper model: {model_size} on {device} ({compute_type})...")
             self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        else:
             print("Skipping Whisper model load (Gen-Only Mode).")
            
        if not self.openai_key and not self.gemini_key:
            print("WARNING: No API keys.")
        
        # Initialize OCR if enabled
        self.ocr = None
        if config.ocr.enabled:
            try:
                print("Initializing OCR for board text and equation extraction...")
                self.ocr = BoardOCR()
                print("OCR initialized successfully")
            except Exception as e:
                print(f"Warning: OCR initialization failed: {e}")
                print("Continuing without OCR (Gemini will still process images)")

    def process_session(self):
        if not os.path.exists(self.manifest_path):
            print(f"ERROR: Manifest not found: {self.manifest_path}")
            print("Please run 'record' command first to create a session.")
            return False
        
        self.renumber_boards_chronologically()
        return True

    def renumber_boards_chronologically(self):
        """
        Renames Board #1, #2... based on who was written on FIRST.
        """
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            
            if not manifest: return
            
            # (Renumbering logic condensed for brevity - functionality remains same)
            # 1. Find First Appearance Time for each Board ID
            first_appearances = {}
            for entry in manifest:
                bid = entry["board_id"]
                ts = entry["timestamp"]
                if bid not in first_appearances:
                    first_appearances[bid] = ts
                else:
                    if ts < first_appearances[bid]:
                        first_appearances[bid] = ts
            
            # 2. Sort Board IDs by Time
            sorted_unique_ids = sorted(first_appearances.keys(), key=lambda k: first_appearances[k])
            
            # 3. Create Mapping
            id_map = {old_id: rank+1 for rank, old_id in enumerate(sorted_unique_ids)}
            
            # 4. Rename Files on Disk (Using TEMP logic)
            import shutil
            for filename in os.listdir(self.output_dir):
                if filename.startswith("board_") and filename.endswith(".jpg"):
                    try:
                        parts = filename.split("_")
                        old_id = int(parts[1])
                        timestamp_part = parts[2]
                        if old_id in id_map:
                            new_id = id_map[old_id]
                            temp_name = f"TEMP_board_{new_id}_{timestamp_part}"
                            os.rename(os.path.join(self.output_dir, filename), os.path.join(self.output_dir, temp_name))
                    except: continue
            
            for filename in os.listdir(self.output_dir):
                if filename.startswith("TEMP_board_"):
                    final_name = filename.replace("TEMP_", "")
                    os.rename(os.path.join(self.output_dir, filename), os.path.join(self.output_dir, final_name))
            
            # 5. Update Manifest
            new_manifest = []
            for entry in manifest:
                old_id = entry["board_id"]
                if old_id in id_map:
                    entry["board_id"] = id_map[old_id]
                    if "image_path" in entry:
                         old_img = entry["image_path"]
                         parts = old_img.split("_")
                         if len(parts) >= 3:
                             entry["image_path"] = f"board_{id_map[old_id]}_{parts[2]}"
                new_manifest.append(entry)
            
            with open(self.manifest_path, 'w') as f:
                json.dump(new_manifest, f, indent=4)
                
            print("Board renumbering complete.")
        except Exception as e:
            print(f"Warning: Board renumbering failed: {e}") 
        
        # --- NEW: Process Board Images if Gemini is available ---
        board_transcriptions = ""
        if self.gemini_model:
             board_transcriptions = self.transcribe_board_images()

        # Transcribe or Load
        if self.skip_transcription:
            print("Skipping Transcription. Loading existing 'transcript.txt'...")
            transcript_path = os.path.join(self.output_dir, "transcript.txt")
            if not os.path.exists(transcript_path):
                print(f"ERROR: {transcript_path} not found.")
                return
            with open(transcript_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            detected_language = self.language if self.language else "tr"
            
        else:
            # Validate audio file exists and has content
            if not os.path.exists(self.audio_path):
                print(f"ERROR: Audio file not found: {self.audio_path}")
                print("Recording may have failed. Cannot transcribe.")
                return
            
            audio_size = os.path.getsize(self.audio_path)
            if audio_size < 1000:  # Less than 1KB is likely corrupt/empty
                print(f"ERROR: Audio file too small ({audio_size} bytes), may be corrupt")
                return
            
            print(f"Audio file OK: {audio_size / (1024*1024):.1f} MB")
            
            lang_msg = self.language if self.language else "Auto-Detect"
            print(f"Transcribing master audio file (Language: {lang_msg})...")
            
            segments, info = self.whisper_model.transcribe(
                self.audio_path, 
                beam_size=5, 
                language=self.language,
                condition_on_previous_text=False,
                vad_filter=True 
            )
            
            print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            detected_language = info.language
            
            full_text_parts = []
            print("Processing segments...")
            for segment in segments:
                full_text_parts.append(segment.text)
            
            full_text = " ".join(full_text_parts)
            
            with open(os.path.join(self.output_dir, "transcript.txt"), "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"Transcript saved to {os.path.join(self.output_dir, 'transcript.txt')}")
            
        # Generate Summary and Questions (Merging Text + Board Notes)
        if self.openai_client or self.gemini_model:
            # Use detected language
            final_lang = self.language if self.language else detected_language
            self.generate_llm_content(full_text, board_transcriptions, language_code=final_lang)
        else:
            print("Skipping AI generation (No API Keys).")

    def transcribe_board_images(self):
        """
        Iterates through manifest images and uses Gemini Vision to transcribe them.
        Returns a formatted markdown string.
        """
        print("Analyzing Board Images with Gemini Vision...")
        
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
        except: return ""

        results = []
        
        # Sort by timestamp to tell a story
        manifest.sort(key=lambda x: x["timestamp"])
        
        for i, entry in enumerate(manifest):
            img_filename = entry.get("image_path")
            board_id = entry.get("board_id")
            timestamp = entry.get("timestamp")
            
            if not img_filename: continue
            
            img_path = os.path.join(self.output_dir, img_filename)
            if not os.path.exists(img_path): continue
            
            print(f"  - Processing {img_filename} (Board #{board_id})...")
            
            try:
                img = PIL.Image.open(img_path)
                
                # Step 1: Try OCR extraction if available
                ocr_data = None
                if self.ocr:
                    try:
                        print(f"    Running OCR on {img_filename}...")
                        ocr_data = self.ocr.process_board_image(img_path)
                        if ocr_data and ocr_data.get('equations'):
                            print(f"    Extracted {len(ocr_data['equations'])} equations via OCR")
                    except Exception as e:
                        print(f"    OCR warning: {e}")
                
                # Step 2: Build enhanced prompt with OCR data
                if ocr_data and (ocr_data.get('text') or ocr_data.get('equations')):
                    # OCR found content - give Gemini structured data
                    prompt = f"""
                    Analyze this whiteboard image. OCR has extracted the following:
                    
                    TEXT CONTENT:
                    {ocr_data.get('text', 'N/A')}
                    
                    EQUATIONS (LaTeX):
                    {chr(10).join(['$$' + eq + '$$' for eq in ocr_data.get('equations', [])])}
                    
                    Your task:
                    1. Verify the OCR extraction is correct (fix any errors)
                    2. Add any missing content from the image
                    3. Provide context and explanations for the equations
                    4. Format everything in clean Markdown with LaTeX
                    
                    Use $...$ for inline math, $$...$$ for block math.
                    If there are diagrams, describe them in [italics].
                    """
                else:
                    # No OCR or OCR failed - standard prompt
                    prompt = """
                    Analyze this whiteboard image. 
                    Transcribe all handwritten text, formulas, and diagrams into LaTeX and Markdown.
                    - Use $...$ for inline math.
                    - Use $$...$$ for block math.
                    - If there is a diagram, describe it briefly in [italics].
                    - Do not add conversational filler, just give the content.
                    """
                
                response = self.gemini_model.generate_content([prompt, img])
                
                # Handle safety blocks and empty responses
                if not response.parts:
                    block_reason = getattr(response, 'prompt_feedback', 'Unknown')
                    print(f"    Warning: Content blocked ({block_reason})")
                    transcription = "[Image could not be processed]"
                else:
                    transcription = response.text
                    if not transcription or not transcription.strip():
                        transcription = "[No content extracted from this image]"
                
                results.append(f"### Board #{board_id} (Time: {int(timestamp)}s)\n\n{transcription}\n")
                
            except Exception as e:
                print(f"    Error processing image: {e}")
                
        return "\n".join(results)

    def generate_llm_content(self, text, board_notes, language_code="en"):
        
        # Append Board Notes to the prompt Context
        combined_context = f"TRANSCRIPT:\n{text[:15000]}\n\nBOARD NOTES (Handwriting Analysis):\n{board_notes}"
        
        if language_code == "tr" or language_code == "turkish":
             prompt = f"""
            Sen bir eğitim asistanısın. Aşağıdaki ders materyallerine (Konuşma metni ve Tahta notları) dayanarak:
            
            {combined_context}
            
            Lütfen şunları sağla:
            1. Kısa bir Ders Özeti (Türkçe).
            2. Öğrencilerin anlamasını test etmek için 5 Örnek Soru (Cevapları ile).
            3. Tahtadaki formülleri veya önemli notları temiz bir "Ders Notları" başlığı altında LaTeX formatında düzenle.
            4. Konuyla ilgili Önerilen Okuma Materyalleri:
               - Spesifik Kitap İsimleri ve Yazarları
               - İlgili Akademik Makale Konuları
            """
        else:
            prompt = f"""
            You are an educational assistant. Based on the following lecture materials (Transcript and Board Notes):
            
            {combined_context}
            
            Please provide:
            1. A concise Lecture Summary.
            2. 5 Example Questions to test student understanding.
            3. Cleaned up "Lecture Notes" section derived from the Board Analysis (use LaTeX for math).
            4. Suggested Reading Materials:
               - Specific Book Titles and Authors
               - Relevant Academic Article Topics
            """
        
        try:
            content = ""
            if self.openai_client:
                print("Generating content using OpenAI GPT-3.5...")
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                
            elif self.gemini_model:
                print("Generating content using Google Gemini...")
                response = self.gemini_model.generate_content(prompt)
                # Handle safety blocks
                if not response.parts:
                    print("Warning: Study guide generation blocked by safety filter")
                    content = "[Content generation was blocked]"
                else:
                    content = response.text
            
            # Combine Raw Board Output + AI Summary
            final_output = f"{content}\n\n---\n\n## Raw Board Transcriptions\n\n{board_notes}"
            
            with open(os.path.join(self.output_dir, "study_guide.md"), "w", encoding="utf-8") as f:
                f.write(final_output)
            print("Study Guide generated: study_guide.md")
            
        except Exception as e:
            print(f"Error generating LLM content: {e}")
