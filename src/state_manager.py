# ------------------------------------------------------------------------------
#  Copyright (c) 2025 Efe Deniz Asan <asan.efe.deniz@gmail.com>
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of
#  Efe Deniz Asan. The intellectual and technical concepts contained herein
#  are proprietary to Efe Deniz Asan and are protected by trade secret or
#  copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained
#  from Efe Deniz Asan or via email at <asan.efe.deniz@gmail.com>.
# ------------------------------------------------------------------------------

"""
State manager for crash recovery.
Saves and restores recording session state.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from src.logger import get_logger

logger = get_logger(__name__)


class StateManager:
    """Manages session state for crash recovery"""
    
    def __init__(self, state_file: str = "state.json"):
        self.state_file = Path(state_file)
        self._state: Dict[str, Any] = {}
        self._last_save_time = 0
    
    def save(self, state: Dict[str, Any], force: bool = False):
        """
        Save state to file.
        
        Args:
            state: State dictionary to save
            force: Force save even if recently saved
        """
        # Throttle saves to avoid excessive disk I/O
        current_time = time.time()
        if not force and (current_time - self._last_save_time) < 1.0:
            return
        
        self._state = state
        self._last_save_time = current_time
        
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load state from file.
        
        Returns:
            State dictionary if found, None otherwise
        """
        if not self.state_file.exists():
            logger.debug("No state file found")
            return None
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info(f"State loaded from {self.state_file}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def clear(self):
        """Delete the state file"""
        if self.state_file.exists():
            try:
                self.state_file.unlink()
                logger.info("State file cleared")
            except Exception as e:
                logger.error(f"Failed to clear state: {e}")
        self._state = {}
    
    def exists(self) -> bool:
        """Check if state file exists"""
        return self.state_file.exists()
    
    def create_session_state(
        self,
        output_dir: str,
        locked_teacher_id: Optional[int] = None,
        ref_height: Optional[int] = None,
        board_rois: Optional[list] = None,
        audio_device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a session state dictionary.
        
        Returns:
            State dictionary ready to save
        """
        return {
            "session_active": True,
            "output_dir": output_dir,
            "start_time": time.time(),
            "locked_teacher_id": locked_teacher_id,
            "ref_height": ref_height,
            "board_rois": board_rois or [],
            "audio_device_id": audio_device_id,
            "last_update": time.time()
        }


# Global state manager instance
state_manager = StateManager()
