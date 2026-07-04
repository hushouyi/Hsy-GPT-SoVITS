"""
Training pipeline orchestrator.

Runs the complete training process:
  Step 0: Collect confirmed recordings, generate train.list
  Step 1: Dataset preparation (3-4 subprocess scripts)
  Step 2: S2 (SoVITS) training
  Step 3: S1 (GPT) training
  Step 4: Complete - save model info, update weight.json
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import yaml
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .state import AppState
from .mapping import MappingManager
from .model_utils import (get_upstream_path, update_weight_json,
                         scan_gpt_weights, scan_sovits_weights)


# Progress callback type
ProgressCallback = Callable[[float, str, str], None]
# Args: (percent 0-100, phase_name, message)


class TrainingConfig:
    """Configuration for a training run."""
    def __init__(self, **kwargs):
        self.exp_name = kwargs.get("exp_name", "my_voice")
        self.model_version = kwargs.get("model_version", "v2Pro")
        self.data_dir = kwargs.get("data_dir", "")  # dir with train.list + recorded/
        self.s1_epochs = kwargs.get("s1_epochs", 15)
        self.s2_epochs = kwargs.get("s2_epochs", 5)
        self.batch_size = kwargs.get("batch_size", 8)
        self.text_low_lr_rate = kwargs.get("text_low_lr_rate", 0.4)
        self.precision = kwargs.get("precision", "16-mixed")
        self.save_every_epoch = kwargs.get("save_every_epoch", 1)
        self.if_save_every_weights = kwargs.get("if_save_every_weights", True)
        self.if_save_latest = kwargs.get("if_save_latest", True)
        self.if_dpo = kwargs.get("if_dpo", False)
        self.gpu_numbers = kwargs.get("gpu_numbers", "0")
        self.data_sources = kwargs.get("data_sources", [])  # list of project names


class TrainingPipeline:
    """Orchestrates the complete training pipeline."""

    # Version → config template mapping
    S2_CONFIG_TEMPLATES = {
        "v1": "GPT_SoVITS/configs/s2.json",
        "v2": "GPT_SoVITS/configs/s2.json",
        "v3": "GPT_SoVITS/configs/s2v2Pro.json",
        "v4": "GPT_SoVITS/configs/s2v2Pro.json",
        "v2Pro": "GPT_SoVITS/configs/s2v2Pro.json",
        "v2ProPlus": "GPT_SoVITS/configs/s2v2ProPlus.json",
    }
    S1_CONFIG_TEMPLATES = {
        "v1": "GPT_SoVITS/configs/s1.yaml",
        "v2": "GPT_SoVITS/configs/s1longer-v2.yaml",
        "v3": "GPT_SoVITS/configs/s1longer-v2.yaml",
        "v4": "GPT_SoVITS/configs/s1longer-v2.yaml",
        "v2Pro": "GPT_SoVITS/configs/s1longer-v2.yaml",
        "v2ProPlus": "GPT_SoVITS/configs/s1longer-v2.yaml",
    }
    S2_SCRIPTS = {
        "v1": "GPT_SoVITS/s2_train.py",
        "v2": "GPT_SoVITS/s2_train.py",
        "v3": "GPT_SoVITS/s2_train_v3_lora.py",
        "v4": "GPT_SoVITS/s2_train_v3_lora.py",
        "v2Pro": "GPT_SoVITS/s2_train.py",
        "v2ProPlus": "GPT_SoVITS/s2_train.py",
    }
    WEIGHT_DIRS = {
        "v1": ("SoVITS_weights", "GPT_weights"),
        "v2": ("SoVITS_weights_v2", "GPT_weights_v2"),
        "v3": ("SoVITS_weights_v3", "GPT_weights_v3"),
        "v4": ("SoVITS_weights_v4", "GPT_weights_v4"),
        "v2Pro": ("SoVITS_weights_v2Pro", "GPT_weights_v2Pro"),
        "v2ProPlus": ("SoVITS_weights_v2ProPlus", "GPT_weights_v2ProPlus"),
    }

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._process = None
        self._cancel_flag = False
        self._running = False

    def run(self, progress_callback: ProgressCallback = None) -> Dict:
        """Execute the full training pipeline. Returns result dict."""
        if self._running:
            return {"success": False, "error": "Training already running"}

        self._running = True
        self._cancel_flag = False
        AppState.set_training(True)

        if progress_callback is None:
            progress_callback = lambda p, ph, msg: None

        exp_name = self.config.exp_name
        version = self.config.model_version
        exp_dir = self._get_exp_dir(exp_name)
        tmp_dir = os.path.join(os.path.dirname(exp_dir), "TEMP")
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            # ===== Step 0: Data collection (0-5%) =====
            progress_callback(0, "准备", "Starting training pipeline...")
            progress_callback(1, "准备", f"Experiment: {exp_name}, Version: {version}")

            # Collect recordings from data sources
            train_list_path = os.path.join(exp_dir, "train.list")
            recorded_dir = os.path.join(exp_dir, "recorded")
            os.makedirs(recorded_dir, exist_ok=True)

            total_recordings = self._collect_recordings(train_list_path, recorded_dir, exp_name)
            if total_recordings == 0:
                raise RuntimeError("No confirmed recordings found for training")

            progress_callback(5, "准备", f"Collected {total_recordings} recordings -> {train_list_path}")

            # ===== Step 1: Dataset preparation (5-20%) =====
            python_exe = self._get_python_exe()

            # 1-get-text
            progress_callback(7, "数据准备", "Step 1/4: Extracting text features...")
            self._run_script(python_exe, [
                get_upstream_path("GPT_SoVITS", "prepare_datasets", "1-get-text.py"),
                "--list_path", train_list_path,
                "--opt_dir", exp_dir,
            ])
            self._check_cancel()
            progress_callback(10, "数据准备", "Step 2/4: Extracting HuBERT features...")

            # 2-get-hubert
            self._run_script(python_exe, [
                get_upstream_path("GPT_SoVITS", "prepare_datasets", "2-get-hubert-wav32k.py"),
                "--list_path", train_list_path,
                "--opt_dir", exp_dir,
            ])
            self._check_cancel()
            progress_callback(13, "数据准备", "Step 2b/4: Extracting speaker verification...")

            # 2-get-sv (v2Pro+ only)
            if version in ("v2Pro", "v2ProPlus"):
                self._run_script(python_exe, [
                    get_upstream_path("GPT_SoVITS", "prepare_datasets", "2-get-sv.py"),
                    "--list_path", train_list_path,
                    "--opt_dir", exp_dir,
                ])

            progress_callback(16, "数据准备", "Step 3/4: Extracting semantic tokens...")

            # 3-get-semantic
            pretrained_s2G = self._get_pretrained_path(version, "s2G")
            self._run_script(python_exe, [
                get_upstream_path("GPT_SoVITS", "prepare_datasets", "3-get-semantic.py"),
                "--list_path", train_list_path,
                "--opt_dir", exp_dir,
                "--pretrained_s2G", pretrained_s2G,
            ])
            self._check_cancel()
            progress_callback(20, "数据准备", "Dataset preparation complete")

            # ===== Step 2: S2 Training (20-70%) =====
            progress_callback(25, "S2训练", f"Starting S2 ({version}) training...")
            s2_config_tmp = os.path.join(tmp_dir, f"tmp_s2_{exp_name}.json")
            self._prepare_s2_config(s2_config_tmp, exp_dir, exp_name)

            self._run_script(python_exe, [
                "-s",
                get_upstream_path("GPT_SoVITS", "s2_train.py" if version in ("v1","v2","v2Pro","v2ProPlus") else "s2_train_v3_lora.py"),
                "--config", s2_config_tmp,
            ], progress_callback=lambda phase, pct, msg: progress_callback(
                20 + int(pct * 0.5), f"S2训练", msg
            ) if phase == "general" else None)
            self._check_cancel()
            progress_callback(70, "S2训练", "S2 training complete")

            # ===== Step 3: S1 Training (70-95%) =====
            progress_callback(72, "S1训练", "Starting S1 training...")
            s1_config_tmp = os.path.join(tmp_dir, f"tmp_s1_{exp_name}.yaml")
            self._prepare_s1_config(s1_config_tmp, exp_dir, exp_name)

            self._run_script(python_exe, [
                "-s",
                get_upstream_path("GPT_SoVITS", "s1_train.py"),
                "--config_file", s1_config_tmp,
            ], progress_callback=lambda phase, pct, msg: progress_callback(
                70 + int(pct * 0.25), f"S1训练", msg
            ) if phase == "general" else None)
            self._check_cancel()
            progress_callback(95, "S1训练", "S1 training complete")

            # ===== Step 4: Complete (95-100%) =====
            progress_callback(96, "完成", "Finalizing model...")

            # Find trained weights
            sovits_weight_dir, gpt_weight_dir = self.WEIGHT_DIRS[version]
            sovits_path = self._find_latest_weight(
                get_upstream_path(sovits_weight_dir), exp_name, '.pth')
            gpt_path = self._find_latest_weight(
                get_upstream_path(gpt_weight_dir), exp_name, '.ckpt')

            if not sovits_path or not gpt_path:
                raise RuntimeError("Could not find trained weights")

            # Copy to models/ directory
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models"
            )
            model_subdir = os.path.join(models_dir, f"{exp_name}_e{self.config.s1_epochs}_s{self.config.s2_epochs}")
            os.makedirs(model_subdir, exist_ok=True)

            gpt_src = get_upstream_path(gpt_weight_dir, gpt_path)
            sovits_src = get_upstream_path(sovits_weight_dir, sovits_path)
            shutil.copy2(gpt_src, os.path.join(model_subdir, "gpt.ckpt"))
            shutil.copy2(sovits_src, os.path.join(model_subdir, "sovits.pth"))

            # Save meta info
            data_sources_info = [{"project": p, "recordings": total_recordings}
                                 for p in self.config.data_sources] if self.config.data_sources else []
            meta = {
                "exp_name": exp_name,
                "model_version": version,
                "s1_epochs": self.config.s1_epochs,
                "s2_epochs": self.config.s2_epochs,
                "batch_size": self.config.batch_size,
                "data_source": data_sources_info,
                "total_recordings": total_recordings,
                "trained_at": datetime.now().isoformat(),
                "gpt_weight": gpt_path,
                "sovits_weight": sovits_path,
            }
            with open(os.path.join(model_subdir, "meta.json"), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Update weight.json
            update_weight_json(
                os.path.join(gpt_weight_dir, gpt_path),
                os.path.join(sovits_weight_dir, sovits_path),
                version
            )

            progress_callback(100, "完成", f"Training complete! Model saved to {model_subdir}")

            return {
                "success": True,
                "exp_name": exp_name,
                "model_dir": model_subdir,
                "gpt_path": gpt_path,
                "sovits_path": sovits_path,
                "total_recordings": total_recordings,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[WARN] Training failed: {error_msg}")
            progress_callback(0, "错误", f"Training failed: {error_msg}")
            return {"success": False, "error": error_msg}

        finally:
            self._running = False
            AppState.set_training(False)

    def cancel(self) -> None:
        """Cancel the running training."""
        self._cancel_flag = True
        if self._process and self._process.poll() is None:
            self._process.kill()
        AppState.set_training(False)

    def _check_cancel(self):
        if self._cancel_flag:
            raise RuntimeError("Training cancelled by user")

    def _get_exp_dir(self, exp_name: str) -> str:
        upstream_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        upstream_dir = os.path.join(upstream_dir, "GPT-SoVITS-v2pro-20250604-nvidia50")
        return os.path.join(upstream_dir, "logs", exp_name)

    def _get_python_exe(self) -> str:
        runtime_py = get_upstream_path("runtime", "python.exe")
        if os.path.exists(runtime_py):
            return runtime_py
        return "python"

    def _get_pretrained_path(self, version: str, model: str) -> str:
        """Get path to pretrained model."""
        base = f"GPT_SoVITS/pretrained_models"
        if version == "v2Pro":
            return get_upstream_path(base, "v2Pro", f"{model}v2Pro.pth")
        elif version == "v2ProPlus":
            return get_upstream_path(base, "v2ProPlus", f"{model}v2ProPlus.pth")
        else:
            return get_upstream_path(base, f"{model}.pth")

    def _collect_recordings(self, train_list_path: str, recorded_dir: str,
                           exp_name: str) -> int:
        """Collect confirmed recordings from data sources into exp_dir."""
        # Import mapping manager here to avoid circular imports
        from .mapping import MappingManager
        from .project_manager import ProjectManager

        total = 0
        projects_dir = os.path.dirname(ProjectManager.get_project_dir("dummy"))
        lines = []

        sources = self.config.data_sources or [AppState.get_current_project()]

        for proj_name in sources:
            mapping_path = ProjectManager.get_mapping_path(proj_name)
            if not os.path.exists(mapping_path):
                continue

            mm = MappingManager()
            mm.load(mapping_path)
            entries = mm.get_confirmed()

            for entry in entries:
                src_wav = os.path.join(projects_dir, proj_name, entry.wav_path)
                if not os.path.exists(src_wav):
                    print(f"[WARN] Missing WAV: {src_wav}")
                    continue

                # Copy with unique name
                dest_name = f"wav_{entry.idx:04d}_{proj_name}.wav"
                dest_path = os.path.join(recorded_dir, dest_name)
                shutil.copy2(src_wav, dest_path)

                lines.append(f"{dest_path}|speaker001|zh|{entry.text}")
                total += 1

        with open(train_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return total

    def _prepare_s2_config(self, config_path: str, exp_dir: str, exp_name: str):
        """Prepare S2 training config JSON with user parameters."""
        template = self.S2_CONFIG_TEMPLATES.get(self.config.model_version, "GPT_SoVITS/configs/s2v2Pro.json")
        template_path = get_upstream_path(template)

        with open(template_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        version = self.config.model_version
        sovits_weight_dir = self.WEIGHT_DIRS[version][0]

        cfg["train"]["batch_size"] = self.config.batch_size
        cfg["train"]["epochs"] = self.config.s2_epochs
        cfg["train"]["text_low_lr_rate"] = self.config.text_low_lr_rate
        cfg["train"]["pretrained_s2G"] = self._get_pretrained_path(version, "s2G")
        cfg["train"]["pretrained_s2D"] = self._get_pretrained_path(version, "s2D")
        cfg["train"]["if_save_latest"] = self.config.if_save_latest
        cfg["train"]["if_save_every_weights"] = self.config.if_save_every_weights
        cfg["train"]["save_every_epoch"] = self.config.save_every_epoch
        cfg["train"]["gpu_numbers"] = self.config.gpu_numbers
        cfg["model"]["version"] = version
        cfg["data"]["exp_dir"] = exp_dir
        cfg["s2_ckpt_dir"] = exp_dir
        cfg["save_weight_dir"] = get_upstream_path(sovits_weight_dir)
        cfg["name"] = exp_name

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def _prepare_s1_config(self, config_path: str, exp_dir: str, exp_name: str):
        """Prepare S1 training config YAML with user parameters."""
        template = self.S1_CONFIG_TEMPLATES.get(self.config.model_version, "GPT_SoVITS/configs/s1longer-v2.yaml")
        template_path = get_upstream_path(template)

        with open(template_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        version = self.config.model_version
        gpt_weight_dir = self.WEIGHT_DIRS[version][1]

        if not isinstance(cfg, dict):
            cfg = {}

        if "train" not in cfg:
            cfg["train"] = {}
        cfg["train"]["batch_size"] = self.config.batch_size
        cfg["train"]["epochs"] = self.config.s1_epochs
        cfg["train"]["precision"] = self.config.precision
        cfg["train"]["save_every_n_epoch"] = self.config.save_every_epoch
        cfg["train"]["if_save_every_weights"] = self.config.if_save_every_weights
        cfg["train"]["if_save_latest"] = self.config.if_save_latest
        cfg["train"]["if_dpo"] = self.config.if_dpo
        cfg["train"]["half_weights_save_dir"] = get_upstream_path(gpt_weight_dir)
        cfg["train"]["exp_name"] = exp_name
        cfg["pretrained_s1"] = self._get_pretrained_path(version, "s1")
        cfg["train_semantic_path"] = os.path.join(exp_dir, "6-name2semantic.tsv")
        cfg["train_phoneme_path"] = os.path.join(exp_dir, "2-name2text.txt")
        cfg["output_dir"] = os.path.join(exp_dir, f"logs_s1_{version}")

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    def _run_script(self, python_exe: str, args: List[str],
                    progress_callback: Callable = None) -> None:
        """Run a subprocess script and capture output for progress."""
        cmd = f'"{python_exe}" ' + ' '.join(f'"{a}"' if ' ' in a else a for a in args)

        self._process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Monitor output
        epoch_pattern = re.compile(r'[Ee]poch\s*(\d+)\s*[/,:]\s*(\d+)')
        loss_pattern = re.compile(r'[Ll]oss[_:]?\s*[Gg]?[:\s]*([\d.]+)')

        for line in iter(self._process.stdout.readline, ''):
            self._check_cancel()
            line = line.strip()
            if line:
                print(f"[TRAIN] {line[:200]}")  # Truncate long lines for console

                # Try to extract progress info
                if progress_callback:
                    epoch_match = epoch_pattern.search(line)
                    loss_match = loss_pattern.search(line)
                    if epoch_match:
                        current = int(epoch_match.group(1))
                        total = int(epoch_match.group(2))
                        if total > 0:
                            pct = current / total * 100
                            progress_callback("general", pct, line[:100])

        self._process.wait()
        if self._process.returncode != 0 and not self._cancel_flag:
            raise RuntimeError(f"Script failed with exit code {self._process.returncode}")

    def _find_latest_weight(self, directory: str, exp_name: str, ext: str) -> Optional[str]:
        """Find the latest trained weight file matching exp_name."""
        if not os.path.exists(directory):
            return None
        candidates = []
        for f in os.listdir(directory):
            if f.startswith(exp_name) and f.endswith(ext):
                candidates.append(f)
        if not candidates:
            return None
        return sorted(candidates)[-1]
