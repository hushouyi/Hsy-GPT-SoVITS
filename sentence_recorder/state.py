"""
Global application state with mutual exclusion.

Mutual exclusion rules:
  training=True  → recording, inference blocked
  inference_api=True → training blocked
  recording=True → project switching blocked, other recording blocked
"""

import threading

_lock = threading.Lock()

# Global state variables (accessed via getter/setter for thread safety)
_training = False
_inference_api = False
_recording = False
_current_project = "default"


class AppState:
    """Global application state manager."""

    @staticmethod
    def get_training() -> bool:
        with _lock:
            return _training

    @staticmethod
    def set_training(val: bool) -> None:
        with _lock:
            global _training
            _training = val

    @staticmethod
    def get_inference_api() -> bool:
        with _lock:
            return _inference_api

    @staticmethod
    def set_inference_api(val: bool) -> None:
        with _lock:
            global _inference_api
            _inference_api = val

    @staticmethod
    def get_recording() -> bool:
        with _lock:
            return _recording

    @staticmethod
    def set_recording(val: bool) -> None:
        with _lock:
            global _recording
            _recording = val

    @staticmethod
    def get_current_project() -> str:
        with _lock:
            return _current_project

    @staticmethod
    def set_current_project(val: str) -> None:
        with _lock:
            global _current_project
            _current_project = val

    @staticmethod
    def get_status_text() -> str:
        """Return human-readable status text for global status bar."""
        with _lock:
            if _training:
                return "[TRAIN] Training in progress"
            if _recording:
                return "[REC] Recording..."
            if _inference_api:
                return "[INFER] Inference API running"
            return "[OK] Idle"
