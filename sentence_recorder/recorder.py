"""
Audio recording manager using sounddevice.

Singleton pattern: only one recording can happen at a time.
Records to 24000Hz mono WAV files.
"""

import os
import time
import wave
import threading
import numpy as np
import sounddevice as sd

from .state import AppState


class RecordingManager:
    """Global recording manager (singleton)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self._stream = None
        self._audio_data = []
        self._save_path = None
        self._start_time = None
        self._recording = False

    def start_recording(self, save_path: str, sample_rate: int = 24000) -> bool:
        """Start recording to the given file path. Returns True on success."""
        if self._recording:
            print(f"[WARN] Already recording, stop first")
            return False

        if AppState.get_training():
            print(f"[WARN] Cannot record while training")
            return False

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            def callback(indata, frames, time_info, status):
                if status:
                    print(f"[WARN] Recording status: {status}")
                self._audio_data.append(indata.copy())

            self._audio_data = []
            self._save_path = save_path
            self._sample_rate = sample_rate
            self._stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                callback=callback
            )
            self._stream.start()
            self._start_time = time.time()
            self._recording = True
            AppState.set_recording(True)
            print(f"[OK] Recording started -> {save_path}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to start recording: {e}")
            self._cleanup()
            return False

    def stop_recording(self) -> dict:
        """Stop recording and save WAV file. Returns recording info dict."""
        if not self._recording:
            return {"path": "", "duration": 0, "sample_rate": 0}

        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()

            duration = time.time() - self._start_time
            recorded_at = time.strftime("%Y%m%d_%H%M%S")

            if self._audio_data and len(self._audio_data) > 0:
                audio_array = np.concatenate(self._audio_data, axis=0)
                self._save_wav(self._save_path, audio_array, self._sample_rate)
                print(f"[OK] Recording saved: {self._save_path} ({duration:.1f}s)")
            else:
                print(f"[WARN] No audio data captured")
                return {"path": "", "duration": 0, "sample_rate": 0}

            result = {
                "path": self._save_path,
                "duration": round(duration, 2),
                "sample_rate": self._sample_rate,
                "recorded_at": recorded_at,
            }
            return result
        except Exception as e:
            print(f"[WARN] Failed to stop recording: {e}")
            return {"path": "", "duration": 0, "sample_rate": 0}
        finally:
            self._cleanup()

    def _save_wav(self, path: str, data: np.ndarray, sample_rate: int):
        """Save int16 numpy array as WAV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())

    def _cleanup(self):
        self._stream = None
        self._audio_data = []
        self._save_path = None
        self._start_time = None
        self._recording = False
        AppState.set_recording(False)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_save_path(self) -> str:
        return self._save_path
