"""
OCR module for extracting text and mathematical equations from board images.
Combines Tesseract (for text) and Pix2Tex (for LaTeX equations).
"""
import os
from typing import Dict, List, Optional
from pathlib import Path
import json

try:
    import pytesseract
    from PIL import Image, ImageEnhance
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from pix2tex.cli import LatexOCR
    PIX2TEX_AVAILABLE = True
except ImportError:
    PIX2TEX_AVAILABLE = False

from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class BoardOCR:
    """Extracts text and equations from whiteboard/blackboard images"""
    
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.pix2tex_available = PIX2TEX_AVAILABLE
        
        # Initialize Pix2Tex if available
        if self.pix2tex_available and config.ocr.enabled:
            try:
                logger.info("Initializing Pix2Tex for equation extraction...")
                self.latex_ocr = LatexOCR()
                logger.info("Pix2Tex initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Pix2Tex: {e}")
                self.pix2tex_available = False
        else:
            self.latex_ocr = None
        
        if not self.tesseract_available:
            logger.warning("Tesseract not available. Text extraction disabled.")
        if not self.pix2tex_available:
            logger.warning("Pix2Tex not available. Equation extraction disabled.")
    
    def _preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed PIL Image
        """
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if config.ocr.enhance_contrast:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
        
        if config.ocr.denoise:
            # Denoise (simple approach)
            from PIL import ImageFilter
            img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        if not self.tesseract_available:
            logger.warning("Tesseract not available for text extraction")
            return ""
        
        try:
            img = self._preprocess_image(image_path)
            
            # Run Tesseract
            text = pytesseract.image_to_string(
                img,
                lang=config.ocr.tesseract_lang,
                config=config.ocr.tesseract_config
            )
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Tesseract OCR failed for {image_path}: {e}")
            return ""
    
    def extract_equations(self, image_path: str) -> List[str]:
        """
        Extract LaTeX equations from image using Pix2Tex.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of LaTeX equation strings
        """
        if not self.pix2tex_available or not self.latex_ocr:
            logger.warning("Pix2Tex not available for equation extraction")
            return []
        
        try:
            # Pix2Tex processes the entire image as one equation
            # For boards with multiple equations, we return the result
            latex = self.latex_ocr(image_path)
            
            if latex and latex.strip():
                return [latex.strip()]
            else:
                return []
        
        except Exception as e:
            logger.error(f"Pix2Tex OCR failed for {image_path}: {e}")
            return []
    
    def process_board_image(self, image_path: str) -> Dict[str, any]:
        """
        Process board image to extract both text and equations.
        
        Args:
            image_path: Path to board snapshot image
            
        Returns:
            Dictionary with extracted data:
            {
                "text": "Regular text found",
                "equations": ["\\frac{x}{2}", "E=mc^2", ...],
                "combined": "Text with embedded LaTeX"
            }
        """
        if not config.ocr.enabled:
            logger.debug(f"OCR disabled, skipping {image_path}")
            return {
                "text": "",
                "equations": [],
                "combined": ""
            }
        
        logger.info(f"Processing {os.path.basename(image_path)} with OCR...")
        
        # Extract text
        text = self.extract_text(image_path)
        
        # Extract equations
        equations = self.extract_equations(image_path)
        
        # Combine results
        combined = text
        if equations:
            combined += "\n\n**Equations Detected:**\n"
            for eq in equations:
                combined += f"$${{eq}}$$\n"
        
        result = {
            "text": text,
            "equations": equations,
            "combined": combined,
            "ocr_success": bool(text or equations)
        }
        
        logger.info(f"  Extracted {len(text)} chars of text, {len(equations)} equations")
        
        return result
    
    def process_multiple_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images, optionally in parallel.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of OCR results (one per image)
        """
        results = []
        
        if config.ocr.parallel_workers > 1:
            # Parallel processing
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=config.ocr.parallel_workers) as executor:
                results = list(executor.map(self.process_board_image, image_paths))
        else:
            # Sequential processing
            for img_path in image_paths:
                results.append(self.process_board_image(img_path))
        
        return results
    
    def save_ocr_cache(self, image_path: str, ocr_result: Dict):
        """Save OCR result to cache file"""
        if not config.ocr.cache_results:
            return
        
        cache_path = Path(image_path).with_suffix('.ocr.json')
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_result, f, indent=2)
            logger.debug(f"Cached OCR result to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache OCR result: {e}")
    
    def load_ocr_cache(self, image_path: str) -> Optional[Dict]:
        """Load cached OCR result if available"""
        if not config.ocr.cache_results:
            return None
        
        cache_path = Path(image_path).with_suffix('.ocr.json')
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            logger.debug(f"Loaded cached OCR from {cache_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load OCR cache: {e}")
            return None
