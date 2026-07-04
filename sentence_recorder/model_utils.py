"""
Model utilities: scan weight directories, manage inference API, manage weight.json.
"""

import json
import os
import re
import subprocess
import time
import requests
from typing import List, Optional

# Paths relative to project root
UPSTREAM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "GPT-SoVITS-v2pro-20250604-nvidia50")


def get_upstream_path(*parts: str) -> str:
    """Get absolute path inside upstream directory."""
    return os.path.join(UPSTREAM_DIR, *parts)


def scan_gpt_weights(version: str = "v2Pro") -> List[str]:
    """Scan GPT_weights_{version}/ for .ckpt files."""
    dir_name = f"GPT_weights_{version}"
    weights_dir = get_upstream_path(dir_name)
    return _scan_files(weights_dir, ['.ckpt'])


def scan_sovits_weights(version: str = "v2Pro") -> List[str]:
    """Scan SoVITS_weights_{version}/ for .pth files."""
    dir_name = f"SoVITS_weights_{version}"
    weights_dir = get_upstream_path(dir_name)
    return _scan_files(weights_dir, ['.pth'])


def _scan_files(directory: str, extensions: List[str]) -> List[str]:
    """Scan directory for files with given extensions."""
    if not os.path.exists(directory):
        return []
    files = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in extensions:
            files.append(f)
    return files


# Inference API management
_api_process = None


def start_inference_api(port: int = 9880) -> bool:
    """Start the inference API subprocess on given port. Returns True if started."""
    global _api_process
    if is_api_running(port):
        print(f"[OK] Inference API already running on port {port}")
        return True

    api_path = get_upstream_path("api_v2.py")
    config_path = get_upstream_path("GPT_SoVITS", "configs", "tts_infer.yaml")
    runtime_python = get_upstream_path("runtime", "python.exe")

    if not os.path.exists(api_path):
        print(f"[WARN] Inference API not found: {api_path}")
        return False

    python_exe = runtime_python if os.path.exists(runtime_python) else "python"

    cmd = (
        f'"{python_exe}" "{api_path}" '
        f'-a 127.0.0.1 -p {port} '
        f'-c "{config_path}"'
    )

    try:
        _api_process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Wait for API to be ready
        for i in range(30):
            time.sleep(1)
            if is_api_running(port):
                print(f"[OK] Inference API started on port {port}")
                return True
        print(f"[WARN] Inference API did not become ready within 30s")
        return False
    except Exception as e:
        print(f"[WARN] Failed to start inference API: {e}")
        return False


def stop_inference_api(port: int = 9880) -> None:
    """Stop the inference API."""
    global _api_process
    # Try graceful shutdown first
    try:
        requests.get(f"http://127.0.0.1:{port}/control?cmd=exit", timeout=2)
    except Exception:
        pass

    # Kill process
    if _api_process and _api_process.poll() is None:
        _api_process.kill()
        _api_process = None

    # Also find and kill any remaining processes on port
    _kill_process_on_port(port)
    print(f"[OK] Inference API stopped")


def is_api_running(port: int = 9880) -> bool:
    """Check if inference API is running on given port."""
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/control", timeout=1)
        return resp.status_code == 200
    except Exception:
        return False


def _kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given port."""
    try:
        result = subprocess.run(
            f'netstat -ano | findstr ":{port} "',
            capture_output=True, text=True, shell=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 5 and 'LISTENING' in line:
                pid = parts[-1]
                subprocess.run(f'taskkill /f /pid {pid}', shell=True, capture_output=True)
    except Exception:
        pass


def cleanup_all_ports() -> None:
    """Clean up all ports used by this application."""
    for port in [7860, 9880, 17860]:
        _kill_process_on_port(port)


# weight.json management

def get_weight_json_path() -> str:
    return get_upstream_path("weight.json")


def read_weight_json() -> dict:
    """Read current weight.json. Returns empty dict if not found."""
    path = get_weight_json_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def update_weight_json(gpt_path: str, sovits_path: str, version: str = "v2") -> None:
    """Update weight.json with new model paths."""
    path = get_weight_json_path()
    data = read_weight_json()
    if "GPT" not in data:
        data["GPT"] = {}
    if "SoVITS" not in data:
        data["SoVITS"] = {}
    data["GPT"][version] = gpt_path
    data["SoVITS"][version] = sovits_path
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] weight.json updated")
    except Exception as e:
        print(f"[WARN] Failed to update weight.json: {e}")


def set_inference_weights(gpt_path: str, sovits_path: str, port: int = 9880) -> bool:
    """Send weight paths to running inference API."""
    if not is_api_running(port):
        return False
    try:
        requests.get(f"http://127.0.0.1:{port}/set_gpt_weights", params={
            "weights_path": get_upstream_path(gpt_path)
        }, timeout=10)
        requests.get(f"http://127.0.0.1:{port}/set_sovits_weights", params={
            "weights_path": get_upstream_path(sovits_path)
        }, timeout=10)
        return True
    except Exception as e:
        print(f"[WARN] Failed to set inference weights: {e}")
        return False


def set_refer_audio(audio_path: str, prompt_text: str, prompt_lang: str = "zh",
                    port: int = 9880) -> bool:
    """Set reference audio for inference API."""
    if not is_api_running(port):
        return False
    try:
        requests.get(f"http://127.0.0.1:{port}/set_refer_audio", params={
            "ref_audio_path": audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang
        }, timeout=10)
        return True
    except Exception as e:
        print(f"[WARN] Failed to set reference audio: {e}")
        return False
