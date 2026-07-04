"""
Project management system.

Each project is a directory under projects/ containing:
  - script.txt    : sentence definitions
  - mapping.txt   : recording status
  - recorded/     : WAV files
  - .locked       : lock file (optional)
"""

import os
import shutil
from typing import List


PROJECTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")
REFERENCE_SCRIPT = os.path.join(os.path.dirname(PROJECTS_DIR), "reference.txt")


class ProjectManager:
    """Manage recording projects."""

    @staticmethod
    def list_projects() -> List[str]:
        """Return sorted list of project names."""
        if not os.path.exists(PROJECTS_DIR):
            return []
        projects = []
        for name in os.listdir(PROJECTS_DIR):
            proj_dir = os.path.join(PROJECTS_DIR, name)
            if os.path.isdir(proj_dir) and os.path.exists(os.path.join(proj_dir, "mapping.txt")):
                projects.append(name)
        return sorted(projects)

    @staticmethod
    def get_project_dir(name: str) -> str:
        return os.path.join(PROJECTS_DIR, name)

    @staticmethod
    def get_script_path(name: str) -> str:
        return os.path.join(PROJECTS_DIR, name, "script.txt")

    @staticmethod
    def get_mapping_path(name: str) -> str:
        return os.path.join(PROJECTS_DIR, name, "mapping.txt")

    @staticmethod
    def get_recorded_dir(name: str) -> str:
        return os.path.join(PROJECTS_DIR, name, "recorded")

    @staticmethod
    def get_lock_path(name: str) -> str:
        return os.path.join(PROJECTS_DIR, name, ".locked")

    @staticmethod
    def exists(name: str) -> bool:
        return os.path.exists(ProjectManager.get_mapping_path(name))

    @staticmethod
    def create(name: str, script_source: str = "default") -> bool:
        """
        Create a new project.
        script_source: "default" (use reference.txt), or a file path.
        """
        if ProjectManager.exists(name):
            print(f"[WARN] Project '{name}' already exists")
            return False

        if not name or not name.strip():
            return False

        # Sanitize name - allow Chinese chars, letters, digits, underscore
        # Just check it's not empty and doesn't contain path separators
        if any(c in name for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            print(f"[WARN] Invalid project name: {name}")
            return False

        proj_dir = ProjectManager.get_project_dir(name)
        recorded_dir = ProjectManager.get_recorded_dir(name)
        os.makedirs(recorded_dir, exist_ok=True)

        # Copy script
        script_path = ProjectManager.get_script_path(name)
        if script_source == "default" and os.path.exists(REFERENCE_SCRIPT):
            shutil.copy2(REFERENCE_SCRIPT, script_path)
        elif os.path.exists(script_source):
            shutil.copy2(script_source, script_path)
        else:
            # Create empty script
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write("")

        # Create empty mapping
        mapping_path = ProjectManager.get_mapping_path(name)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("idx|text|wav_path|confirmed|duration_sec|recorded_at\n")

        print(f"[OK] Project '{name}' created")
        return True

    @staticmethod
    def init_default() -> str:
        """Ensure default project exists. Returns project name."""
        if not ProjectManager.exists("default"):
            ProjectManager.create("default", "default")
        return "default"

    @staticmethod
    def delete(name: str) -> bool:
        """Delete a project directory. Returns False if locked or doesn't exist."""
        if not ProjectManager.exists(name):
            return False
        if ProjectManager.is_locked(name):
            print(f"[WARN] Project '{name}' is locked, cannot delete")
            return False
        proj_dir = ProjectManager.get_project_dir(name)
        try:
            shutil.rmtree(proj_dir)
            print(f"[OK] Project '{name}' deleted")
            return True
        except Exception as e:
            print(f"[WARN] Failed to delete project '{name}': {e}")
            return False

    @staticmethod
    def lock(name: str) -> bool:
        """Lock a project (create .locked file)."""
        lock_path = ProjectManager.get_lock_path(name)
        try:
            with open(lock_path, 'w') as f:
                f.write("locked")
            return True
        except Exception:
            return False

    @staticmethod
    def unlock(name: str) -> bool:
        """Unlock a project (remove .locked file)."""
        lock_path = ProjectManager.get_lock_path(name)
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
            return True
        except Exception:
            return False

    @staticmethod
    def is_locked(name: str) -> bool:
        return os.path.exists(ProjectManager.get_lock_path(name))

    @staticmethod
    def import_project(src_path: str) -> str:
        """
        Import a project from an external directory.
        Returns the new project name, or empty string on failure.
        """
        if not os.path.exists(src_path) or not os.path.isdir(src_path):
            return ""

        # Check it has mapping.txt
        mapping_src = os.path.join(src_path, "mapping.txt")
        if not os.path.exists(mapping_src):
            return ""

        name = os.path.basename(src_path.rstrip('/\\'))
        target = ProjectManager.get_project_dir(name)

        # If name collision, add suffix
        if os.path.exists(target):
            suffix = 1
            while os.path.exists(ProjectManager.get_project_dir(f"{name}_{suffix}")):
                suffix += 1
            name = f"{name}_{suffix}"
            target = ProjectManager.get_project_dir(name)

        try:
            shutil.copytree(src_path, target)
            print(f"[OK] Project imported as '{name}'")
            return name
        except Exception as e:
            print(f"[WARN] Failed to import project: {e}")
            return ""
