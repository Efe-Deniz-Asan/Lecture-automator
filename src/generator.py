import json
import os
from faster_whisper import WhisperModel
from openai import OpenAI
import PIL.Image

class ContentGenerator:
    def __init__(self, output_dir="output", model_size="base", openai_key=None, gemini_key=None, language=None, skip_transcription=False):
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
             print(f"Loading Faster-Whisper model: {model_size} (Int8)...")
             # Run on CPU with Int8 quantization for speed
             self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
             print("Skipping Whisper model load (Gen-Only Mode).")
            
        if not self.openai_key and not self.gemini_key:
            print("WARNING: No API keys.")

    def process_session(self):
        if not os.path.exists(self.manifest_path):
             pass 
        else:
             self.renumber_boards_chronologically()

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
                
                prompt = """
                Analyze this whiteboard image. 
                Transcribe all handwritten text, formulas, and diagrams into LaTeX and Markdown.
                - Use $...$ for inline math.
                - Use $$...$$ for block math.
                - If there is a diagram, describe it briefly in [italics].
                - Do not add conversational filler, just give the content.
                """
                
                response = self.gemini_model.generate_content([prompt, img])
                transcription = response.text
                
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
                content = response.text
            
            # Combine Raw Board Output + AI Summary
            final_output = f"{content}\n\n---\n\n## Raw Board Transcriptions\n\n{board_notes}"
            
            with open(os.path.join(self.output_dir, "study_guide.md"), "w", encoding="utf-8") as f:
                f.write(final_output)
            print("Study Guide generated: study_guide.md")
            
        except Exception as e:
            print(f"Error generating LLM content: {e}")
