"""
Configuration loader and validator for Lecture Automator.
Loads settings from config.yaml and provides defaults.
"""
import yaml
from pathlib import Path
from typing import Any, Dict
from src.logger import get_logger

logger = get_logger(__name__)


class ConfigSection:
    """Base class for configuration sections"""
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class VisionConfig(ConfigSection):
    """Vision system configuration"""
    locked_teacher_timeout_frames: int = 150
    similarity_threshold: float = 0.60
    height_ratio_threshold: float = 0.75
    upper_frame_percentage: float = 0.85
    audience_penalty: float = 0.2


class AudioConfig(ConfigSection):
    """Audio recording configuration"""
    sample_rate: int = 48000
    chunk_size: int = 4096
    format: str = "paInt16"


class RecordingConfig(ConfigSection):
    """Recording session configuration"""
    auto_restart_on_crash: bool = True
    snapshot_cooldown_seconds: float = 2.0
    state_save_interval_seconds: float = 10.0


class LoggingConfig(ConfigSection):
    """Logging configuration"""
    level: str = "INFO"
    file: str = "logs/lecture_automator.log"
    max_bytes: int = 10485760
    backup_count: int = 3


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.version = "2.0.0"
        
        # Load configuration
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                data = {}
        else:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            data = {}
        
        # Initialize sections with defaults
        self.vision = VisionConfig(data.get('vision', {}))
        self.audio = AudioConfig(data.get('audio', {}))
        self.recording = RecordingConfig(data.get('recording', {}))
        self.logging = LoggingConfig(data.get('logging', {}))
        
        # Update version if available
        if 'version' in data:
            self.version = data['version']
    
    def save(self):
        """Save current configuration to file"""
        data = {
            'version': self.version,
            'vision': self.vision.__dict__,
            'audio': self.audio.__dict__,
            'recording': self.recording.__dict__,
            'logging': self.logging.__dict__
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")


# Global configuration instance
config = Config()
